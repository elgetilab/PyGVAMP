"""
The torch_scatter fallback must export everything the encoders import.

Why this exists (2026-08-25). `pygv/utils/alternative_torch_scatter` is the
substitute used when `torch_scatter` is not installed — which is the case in the
deployed environment. It previously exported only a subset of what the Meta and
MetaAtt encoders actually call, so those encoders imported cleanly, stayed
selectable from the CLI, and then raised on the first forward pass:

    NameError: name 'scatter_add' is not defined            (meta.py:61)
    ModuleNotFoundError: No module named 'torch_scatter'    (meta_att.py:89)

Worse, `scatter_min`/`scatter_max` in the fallback called
`torch.ops.torch_scatter.*` — the fallback required the very C++ extension it was
meant to replace.

The existing encoder tests only cover this because torch_scatter happens to be
absent here. These tests pin the contract directly, so the gap cannot reopen on a
machine where torch_scatter IS installed and the fallback goes unexercised.
"""

import re
from pathlib import Path

import pytest
import torch

from pygv.utils import alternative_torch_scatter as fb

ENCODERS = ['pygv/encoder/meta.py', 'pygv/encoder/meta_att.py']


def _repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.mark.parametrize('rel', ENCODERS)
def test_fallback_exports_every_symbol_the_encoder_imports(rel):
    """Whatever an encoder imports from torch_scatter must exist in the fallback."""
    src = (_repo_root() / rel).read_text()
    wanted = set()
    for m in re.finditer(r'from torch_scatter import ([^\n]+)', src):
        wanted |= {n.strip() for n in m.group(1).split(',')}
    assert wanted, f"{rel}: no torch_scatter import found — test needs updating"
    missing = [n for n in sorted(wanted) if not hasattr(fb, n)]
    assert not missing, (
        f"{rel} imports {missing} from torch_scatter, but the fallback does not "
        "provide them — this encoder will raise on its first forward pass"
    )


@pytest.mark.parametrize('rel', ENCODERS)
def test_no_unguarded_torch_scatter_import(rel):
    """Every torch_scatter import must sit inside a try/except ImportError."""
    src = (_repo_root() / rel).read_text()
    for line_no, line in enumerate(src.splitlines(), 1):
        if 'from torch_scatter import' in line:
            indent = len(line) - len(line.lstrip())
            assert indent > 0, (
                f"{rel}:{line_no} imports torch_scatter at module level with no "
                "try/except — it will fail wherever torch_scatter is absent"
            )


def test_fallback_needs_no_torch_scatter_extension():
    """The fallback must be pure torch — no torch.ops.torch_scatter CALLS.

    Checked via AST rather than grep: the module legitimately *mentions*
    torch.ops.torch_scatter in docstrings explaining why it must not use it.
    """
    import ast

    tree = ast.parse(Path(fb.__file__).read_text())

    def dotted(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts))

    offenders = [
        f"line {n.lineno}: {dotted(n)}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and dotted(n).startswith('torch.ops.torch_scatter')
    ]
    assert not offenders, (
        "the fallback calls the torch_scatter C++ extension, so it is not a "
        f"fallback at all: {offenders}"
    )


def test_scatter_softmax_normalises_per_group():
    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    idx = torch.tensor([0, 0, 1, 1, 1])
    out = fb.scatter_softmax(src, idx, dim=0)
    assert torch.allclose(out[:2].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(out[2:].sum(), torch.tensor(1.0), atol=1e-6)
    assert (out > 0).all()


def test_scatter_softmax_is_numerically_stable():
    """Large logits must not overflow — the per-group max must be subtracted."""
    src = torch.tensor([1e4, 1e4 + 1.0, -1e4])
    idx = torch.tensor([0, 0, 1])
    out = fb.scatter_softmax(src, idx, dim=0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:2].sum(), torch.tensor(1.0), atol=1e-6)


def test_scatter_max_matches_a_reference_loop():
    src = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    idx = torch.tensor([0, 0, 1, 1, 1])
    val, arg = fb.scatter_max(src, idx, dim=0)
    assert val.tolist() == [3.0, 5.0]
    assert arg.tolist() == [0, 4]


def test_scatter_add_matches_a_reference_loop():
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    idx = torch.tensor([0, 1, 0, 1])
    assert fb.scatter_add(src, idx, dim=0).tolist() == [4.0, 6.0]
