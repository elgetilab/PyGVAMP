import torch
import torch.nn as nn
from typing import Literal, Optional, Union, List


def _has_constant_row_sum(x: torch.Tensor, atol: float = 1e-4) -> bool:
    """True if every row of ``x`` sums to the same value (e.g. a softmax output).

    For such features the all-ones direction is EXACTLY in the null space of the
    mean-removed covariance: if ``sum_j x_ij = c`` for all i, then each centred
    row is orthogonal to ``1``, so ``C00 @ 1 = 0``. See ``_constant_complement_basis``.
    """
    if x.dim() != 2 or x.shape[1] < 2:
        return False
    sums = x.sum(dim=1)
    return bool(torch.all(torch.abs(sums - sums[0]) <= atol).item())


def _constant_complement_basis(k: int, device, dtype) -> torch.Tensor:
    """Orthonormal basis (k, k-1) of the complement of the all-ones direction.

    Built from a single Householder reflector that maps ``1/sqrt(k)`` onto e_0;
    columns 1..k-1 of that reflector then span ``1^perp`` exactly, so no
    thresholding or numerical rank decision is involved.
    """
    ones = torch.ones(k, device=device, dtype=dtype) / (k ** 0.5)
    e0 = torch.zeros(k, device=device, dtype=dtype)
    e0[0] = 1.0
    v = ones - e0
    nrm = torch.linalg.norm(v)
    if nrm < 1e-12:  # already aligned; the complement is just e_1..e_{k-1}
        householder = torch.eye(k, device=device, dtype=dtype)
    else:
        v = v / nrm
        householder = torch.eye(k, device=device, dtype=dtype) - 2.0 * torch.outer(v, v)
    return householder[:, 1:]


class VAMPScore(nn.Module):
    def __init__(
            self,
            method: Literal['VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE'] = 'VAMP2',
            epsilon: float = 1e-6,
            mode: Literal['trunc', 'regularize'] = 'trunc',
            project_constant_direction: bool = True
    ):
        """
        VAMP Score module for evaluating time-lagged embeddings.

        Parameters
        ----------
        method : str, default='VAMP2'
            VAMP scoring method:
            - 'VAMP1': Nuclear norm of Koopman matrix
            - 'VAMP2': Squared Frobenius norm of Koopman matrix
            - 'VAMPE': Eigendecomposition-based score
            - 'VAMPCE': Custom scoring with cross-entropy
        epsilon : float, default=1e-6
            Cutoff parameter for small eigenvalues or regularization parameter
        mode : str, default='trunc'
            Mode for handling small eigenvalues:
            - 'trunc': Truncate eigenvalues smaller than epsilon
            - 'regularize': Add epsilon to diagonal for regularization
        project_constant_direction : bool, default=True
            When the inputs have a constant row sum (softmax χ, the usual case),
            project onto the complement of the all-ones direction before
            whitening. That direction is EXACTLY in the null space of the
            mean-removed covariance, so this is lossless — but leaving it in
            means its numerically-nonzero eigenvalue (~1e-6 at validation scale
            in float32) can survive the ``epsilon`` cutoff and get amplified by
            ``1/sqrt(epsilon) ≈ 1000``, producing singular values > 1 and scores
            above the k-state ceiling. See ``tests/test_vamp_score_ceiling.py``
            and ``experiments/revgraphvamp_repro.md`` (Aβ42, 2026-07-28).
            Set False to restore the pre-fix behaviour.
        """
        super(VAMPScore, self).__init__()

        self.method = method
        self.epsilon = epsilon
        self.mode = mode
        self.project_constant_direction = project_constant_direction

        # Validate parameters
        valid_methods = ['VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE']
        valid_modes = ['trunc', 'regularize']

        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}', supported are {valid_methods}")

        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}', supported are {valid_modes}")

    def _project_constant(self, data: torch.Tensor, data_lagged: torch.Tensor):
        """Drop the structural null direction when the inputs live on a simplex.

        Returns the pair unchanged unless ``project_constant_direction`` is on AND
        both tensors have a constant row sum. The map is an orthogonal change of
        basis restricted to ``1^perp``; since ``1`` carries no variance after mean
        removal, every whitened-Koopman singular value — and hence the score — is
        mathematically unchanged. What it removes is the 0/0 division that the
        eigenvalue cutoff would otherwise have to adjudicate.
        """
        if not self.project_constant_direction:
            return data, data_lagged
        if not (_has_constant_row_sum(data) and _has_constant_row_sum(data_lagged)):
            return data, data_lagged
        basis = _constant_complement_basis(data.shape[1], data.device, data.dtype)
        return data @ basis, data_lagged @ basis

    def forward(self, data: torch.Tensor, data_lagged: torch.Tensor) -> torch.Tensor:
        """
        Compute the VAMP score based on data and corresponding time-shifted data.

        Parameters
        ----------
        data : torch.Tensor
            (N, d)-dimensional torch tensor containing instantaneous data
        data_lagged : torch.Tensor
            (N, d)-dimensional torch tensor containing time-lagged data

        Returns
        -------
        torch.Tensor
            Computed VAMP score. Includes +1 contribution from constant singular function.
        """
        # TODO: This is an experimental code snippet modifying epsilon, remove if this is behaving weirdly
        # Adjust regularization based on batch size
        #batch_size = data.size(0)
        #n_states = data.size(1)
        #effective_epsilon = self.epsilon * (n_states / batch_size)

        # Temporarily override epsilon for this forward pass
        #original_epsilon = self.epsilon
        #self.epsilon = effective_epsilon

        # Validate inputs
        if not torch.is_tensor(data) or not torch.is_tensor(data_lagged):
            raise TypeError("Data inputs must be torch.Tensor objects")

        if data.shape != data_lagged.shape:
            raise ValueError(f"Data shapes must match but were {data.shape} and {data_lagged.shape}")

        # Compute score based on method
        if self.method in ['VAMP1', 'VAMP2']:
            koopman = self._koopman_matrix(data, data_lagged)

            if self.method == 'VAMP1':
                score = torch.norm(koopman, p='nuc')
            else:  # VAMP2
                score = torch.pow(torch.norm(koopman, p='fro'), 2)

        elif self.method == 'VAMPE':
            # Drop the structural null direction first (see _project_constant);
            # VAMP1/VAMP2 get this inside _koopman_matrix.
            data, data_lagged = self._project_constant(data, data_lagged)

            # Compute Covariance
            c00, c0t, ctt = self._covariances(data, data_lagged, remove_mean=True)

            # Compute inverse square roots
            c00_sqrt_inv = self._sym_inverse(c00, return_sqrt=True)
            ctt_sqrt_inv = self._sym_inverse(ctt, return_sqrt=True)

            # Compute Koopman operator
            koopman = self._multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

            # SVD decomposition
            u, s, v = torch.svd(koopman)
            mask = s > self.epsilon

            # Apply mask and compute transformed matrices
            u = torch.mm(c00_sqrt_inv, u[:, mask])
            v = torch.mm(ctt_sqrt_inv, v[:, mask])
            s = s[mask]

            # Compute score
            u_t = u.t()
            v_t = v.t()
            s = torch.diag(s)
            score = torch.trace(
                2. * self._multi_dot([s, u_t, c0t, v]) -
                self._multi_dot([s, u_t, c00, u, s, v_t, ctt, v])
            )

        elif self.method == 'VAMPCE':
            score = torch.trace(data[0])
            score = -1.0 * score
            return score

        # Add the +1 contribution from constant singular function
        final_score = 1 + score

        return final_score

    def loss(self, data: torch.Tensor, data_lagged: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Calculate the VAMP loss (negative VAMP score).

        Parameters
        ----------
        data : torch.Tensor
            (N, d)-dimensional torch tensor containing instantaneous data
        data_lagged : torch.Tensor
            (N, d)-dimensional torch tensor containing time-lagged data
        k : Optional[int], default=None
            Number of singular values to consider (not used in this implementation)

        Returns
        -------
        torch.Tensor
            VAMP loss
        """
        # The k parameter is not used in this implementation but kept for API compatibility
        return -self.forward(data, data_lagged)

    def _koopman_matrix(self, data: torch.Tensor, data_lagged: torch.Tensor,
                        epsilon: float = None, mode: str = None) -> torch.Tensor:
        """
        Compute the Koopman matrix from data and time-lagged data.

        Parameters
        ----------
        data : torch.Tensor
            Instantaneous data.
        data_lagged : torch.Tensor
            Time-lagged data.
        epsilon : float, optional
            Cutoff parameter for small eigenvalues. Defaults to self.epsilon if None.
        mode : str, optional
            Regularization mode for Hermitian inverse. Defaults to self.mode if None.

        Returns
        -------
        torch.Tensor
            Koopman matrix.
        """
        # Use class defaults if not provided
        if epsilon is None:
            epsilon = self.epsilon
        if mode is None:
            mode = self.mode

        # Drop the structural null direction before whitening (see
        # _project_constant); no-op unless the inputs have a constant row sum.
        data, data_lagged = self._project_constant(data, data_lagged)

        # Compute covariances
        c00, c0t, ctt = self._covariances(data, data_lagged, remove_mean=True)

        # Compute inverse square roots
        c00_sqrt_inv = self._sym_inverse(c00, return_sqrt=True, epsilon=epsilon, mode=mode)
        ctt_sqrt_inv = self._sym_inverse(ctt, return_sqrt=True, epsilon=epsilon, mode=mode)

        # Compute Koopman matrix: C00^(-1/2) @ C0t @ Ctt^(-1/2)
        # Note the transpose at the end to match deeptime's implementation
        koopman = self._multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

        return koopman

    def _covariances(self, data: torch.Tensor, data_lagged: torch.Tensor, remove_mean: bool = True) -> tuple:
        """
        Compute instantaneous and time-lagged covariances matrices.

        Parameters
        ----------
        data : torch.Tensor, shape (T, n)
            Instantaneous data.
        data_lagged : torch.Tensor, shape (T, n)
            Time-lagged data.
        remove_mean: bool, default=True
            Whether to remove the mean of the data.

        Returns
        -------
        cov_00 : torch.Tensor, shape (n, n)
            Auto-covariance matrix of instantaneous data.
        cov_0t : torch.Tensor, shape (n, n)
            Cross-covariance matrix of instantaneous and time-lagged data.
        cov_tt : torch.Tensor, shape (n, n)
            Auto-covariance matrix of time-lagged data.
        """
        assert data.shape == data_lagged.shape, "data and data_lagged must be of same shape"
        batch_size = data.shape[0]

        if remove_mean:
            data = data - data.mean(dim=0, keepdim=True)
            data_lagged = data_lagged - data_lagged.mean(dim=0, keepdim=True)

        # Calculate the cross-covariance and auto-covariances
        data_t = data.transpose(0, 1)  # Transpose for matrix multiplication
        data_lagged_t = data_lagged.transpose(0, 1)

        c00 = 1 / (batch_size - 1) * torch.matmul(data_t, data)  # Instantaneous auto-covariance
        c0t = 1 / (batch_size - 1) * torch.matmul(data_t, data_lagged)  # Cross-covariance
        ctt = 1 / (batch_size - 1) * torch.matmul(data_lagged_t, data_lagged)  # Time-lagged auto-covariance

        # Add regularization to diagonal
        epsilon_eye = self.epsilon * torch.eye(c00.size(0), device=c00.device)
        c00 = c00 + epsilon_eye
        ctt = ctt + epsilon_eye

        return c00, c0t, ctt

    def _sym_inverse(self, mat, epsilon=None, return_sqrt=False, mode=None):
        """
        Utility function that returns the inverse of a matrix, with the
        option to return the square root of the inverse matrix.

        Parameters
        ----------
        mat: torch.Tensor with shape [m,m]
            Matrix to be inverted.
        epsilon : float, optional
            Cutoff for eigenvalues. Defaults to self.epsilon if None.
        return_sqrt: bool, default=False
            If True, the square root of the inverse matrix is returned instead
        mode: str, optional
            Regularization mode. Defaults to self.mode if None.
            'trunc': Truncate eigenvalues smaller than epsilon.
            'regularize': Add epsilon to diagonal before decomposition.
            'clamp': Clamp eigenvalues to be at least epsilon.

        Returns
        -------
        x_inv: torch.Tensor with shape [m,m]
            Inverse of the original matrix
        """
        # Use class defaults if not provided
        if epsilon is None:
            epsilon = self.epsilon
        if mode is None:
            mode = self.mode

        # Perform eigendecomposition
        eigval, eigvec = self._symeig_reg(mat, epsilon, mode)

        # Build the diagonal matrix with the filtered eigenvalues or square
        # root of the filtered eigenvalues according to the parameter
        if return_sqrt:
            diag = torch.diag(torch.sqrt(1. / eigval))
        else:
            diag = torch.diag(1. / eigval)

        return self._multi_dot([eigvec, diag, eigvec.t()])

    def _symeig_reg(self, mat, epsilon=None, mode=None):
        """
        Symmetric eigenvalue decomposition with regularization options.
        """
        # Use class defaults if not provided
        if epsilon is None:
            epsilon = self.epsilon
        if mode is None:
            mode = self.mode

        valid_modes = ('trunc', 'regularize', 'clamp')
        assert mode in valid_modes, f"Invalid mode {mode}, supported are {valid_modes}"

        if mode == 'regularize':
            # Add epsilon to diagonal before decomposition
            identity = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
            mat = mat + epsilon * identity

        # Calculate eigenvalues and eigenvectors
        eigval, eigvec = torch.linalg.eigh(mat)

        # Apply regularization based on mode
        if mode == 'trunc':
            # Filter out eigenvalues below threshold
            mask = eigval > epsilon

            # Important: Check if mask is not empty to avoid dimension issues
            if not torch.any(mask):
                # If all eigenvalues would be filtered out, use smallest eigenvalue
                # and set it to epsilon to avoid empty tensor
                min_idx = torch.argmin(eigval)
                mask = torch.zeros_like(eigval, dtype=torch.bool)
                mask[min_idx] = True
                eigval = torch.tensor([epsilon], device=eigval.device)
            else:
                eigval = eigval[mask]

            eigvec = eigvec[:, mask]  # Select corresponding eigenvectors
        elif mode == 'regularize':
            # Take absolute values of eigenvalues
            eigval = torch.abs(eigval)
        elif mode == 'clamp':
            # Clamp eigenvalues to be at least epsilon
            eigval = torch.clamp_min(eigval, min=epsilon)

        return eigval, eigvec

    def _multi_dot(self, matrices):
        """
        Compute the dot product of multiple matrices efficiently.

        Parameters
        ----------
        matrices : list of torch.Tensor
            List of matrices to multiply.

        Returns
        -------
        torch.Tensor
            Product of all matrices.
        """
        #result = matrices[0]
        result = torch.linalg.multi_dot(matrices)
        #for matrix in matrices[1:]:
        #    result = result @ matrix
        return result

