"""
PaiNN encoder configuration
"""
from dataclasses import dataclass
from typing import Optional
from ..base_config import BaseConfig


@dataclass
class PaiNNConfig(BaseConfig):
    """Configuration specific to the PaiNN (equivariant) encoder.

    PaiNN carries a scalar AND a 3-vector per feature channel per node, so at
    equal ``hidden_dim`` it is NOT capacity-matched to SchNet. Match parameter
    counts explicitly when running a comparison arm.
    """
    encoder_type: str = "painn"

    # Generic dims are used when the painn_* overrides are left at None,
    # which keeps a single-variable encoder swap possible.
    node_dim: int = 16
    edge_dim: int = 16
    hidden_dim: int = 128
    output_dim: int = 64
    n_interactions: int = 3

    # PaiNN-specific
    painn_hidden_dim: Optional[int] = None
    painn_output_dim: Optional[int] = None
    painn_n_interactions: Optional[int] = None
    painn_activation: str = "silu"          # paper default
    painn_cutoff: Optional[float] = None    # None = no cutoff (k-NN graphs)
    painn_shared_interactions: bool = False
