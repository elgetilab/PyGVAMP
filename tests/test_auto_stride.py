"""
Tests for Phase 2 — auto-stride.

Covers:
  * formula correctness across (tau, frame_dt) combinations
  * never-zero guarantee
  * per-lag-time differentiation
  * constant-within-lag (retrains reuse stride)
  * off-by-default (backward compat)
  * preprocessing-stride floor (auto_stride never goes below cache density)
"""

import argparse
import math
import pytest
from pathlib import Path

from pygv.config.base_config import BaseConfig
from pygv.pipe.master_pipeline import PipelineOrchestrator


def _make_orchestrator(
    *,
    auto_stride: bool,
    frame_dt_ps: float | None = 200.0,
    prep_stride: int = 1,
    stride: int = 1,
    timestep_ns: float | None = None,
) -> PipelineOrchestrator:
    """Build an orchestrator with only the fields needed by _compute_auto_stride / _create_train_args."""
    cfg = BaseConfig()
    cfg.auto_stride = auto_stride
    cfg.stride = stride
    cfg.timestep = timestep_ns
    cfg.cache = False
    orch = PipelineOrchestrator(cfg)
    orch._frame_dt_ps = frame_dt_ps
    orch._prep_stride = prep_stride
    return orch


def _make_train_args(orch: PipelineOrchestrator, lag_time_ns: float) -> argparse.Namespace:
    """Call _create_train_args with minimal filler to exercise the stride path."""
    dirs = {"cache": None}
    exp_dir = "/tmp/test_auto_stride_dummy"
    return orch._create_train_args(
        dirs=dirs,
        dataset_path=None,
        lag_time=lag_time_ns,
        n_states=5,
        exp_dir=exp_dir,
    )


# ---------------------------------------------------------------------------
# 1. formula correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lag_ns, frame_dt_ps, prep_stride, expected",
    [
        # From the plan:  frame_dt=200ps, tau=20ns -> stride=10
        (20.0, 200.0, 1, 10),
        # ab42_red spec from tracker: frame_dt=250ps
        (1.0,   250.0, 1, 1),    # floor(1000/2500)=0 -> max(1,.)=1
        (2.5,   250.0, 1, 1),    # floor(2500/2500)=1
        (5.0,   250.0, 1, 2),    # floor(5000/2500)=2
        (10.0,  250.0, 1, 4),    # floor(10000/2500)=4
        (20.0,  250.0, 1, 8),
        (25.0,  250.0, 1, 10),
        (50.0,  250.0, 1, 20),
        # Preprocessing stride widens the cache frame_dt
        (10.0, 100.0, 5, 2),     # cache_dt=500ps, floor(10000/5000)=2
        # Very small tau -> stride floor kicks in
        (0.01, 200.0, 1, 1),
        # Very large tau -> big stride
        (1000.0, 100.0, 1, 1000),
    ],
)
def test_stride_computation_correctness(lag_ns, frame_dt_ps, prep_stride, expected):
    orch = _make_orchestrator(
        auto_stride=True, frame_dt_ps=frame_dt_ps, prep_stride=prep_stride
    )
    assert orch._compute_auto_stride(lag_ns) == expected


# ---------------------------------------------------------------------------
# 2. minimum-one guarantee
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lag_ns", [1e-6, 1e-3, 0.01, 0.05, 0.1])
def test_stride_minimum_one(lag_ns):
    orch = _make_orchestrator(auto_stride=True, frame_dt_ps=1000.0, prep_stride=1)
    assert orch._compute_auto_stride(lag_ns) >= 1


# ---------------------------------------------------------------------------
# 3. per-lag-time differentiation
# ---------------------------------------------------------------------------

def test_stride_per_lag_time():
    orch = _make_orchestrator(auto_stride=True, frame_dt_ps=200.0, prep_stride=1)
    strides = {tau: orch._compute_auto_stride(tau) for tau in (1.0, 5.0, 20.0, 50.0)}
    # Every distinct tau in this set should produce a distinct stride
    # for frame_dt=200ps.
    assert len(set(strides.values())) == len(strides), strides
    # Strictly non-decreasing in tau
    sorted_taus = sorted(strides)
    strides_in_order = [strides[t] for t in sorted_taus]
    assert strides_in_order == sorted(strides_in_order), strides


# ---------------------------------------------------------------------------
# 4. constant-within-lag (retrain reuse)
# ---------------------------------------------------------------------------

def test_stride_constant_within_lag():
    """Calling _compute_auto_stride repeatedly with the same tau must
    always return the same value (retrains at the same lag must reuse stride)."""
    orch = _make_orchestrator(auto_stride=True, frame_dt_ps=250.0, prep_stride=1)
    first = orch._compute_auto_stride(10.0)
    for _ in range(5):
        assert orch._compute_auto_stride(10.0) == first


def test_create_train_args_propagates_runtime_stride_within_lag():
    """The orchestrator's training-args builder must emit the same
    runtime_stride for repeated calls at the same lag (simulates retrain loop)."""
    orch = _make_orchestrator(auto_stride=True, frame_dt_ps=250.0, prep_stride=1)
    lag = 10.0
    ns_values = (10, 9, 8)  # retrain shrinks n_states
    strides = [_make_train_args(orch, lag).runtime_stride for _ in ns_values]
    assert len(set(strides)) == 1, f"runtime_stride drifted across retrain rounds: {strides}"


# ---------------------------------------------------------------------------
# 5. off-by-default
# ---------------------------------------------------------------------------

def test_stride_off_by_default():
    """With auto_stride disabled, _create_train_args must set runtime_stride=1
    and not invoke _compute_auto_stride."""
    orch = _make_orchestrator(auto_stride=False, frame_dt_ps=250.0, prep_stride=1)
    args = _make_train_args(orch, lag_time_ns=10.0)
    assert args.runtime_stride == 1
    assert args.stride == 1  # config default, untouched


def test_baseconfig_default_auto_stride_false():
    assert BaseConfig().auto_stride is False


def test_auto_stride_requires_frame_dt():
    """If auto_stride is on but frame_dt_ps wasn't discovered, we fail loud."""
    orch = _make_orchestrator(auto_stride=True, frame_dt_ps=None, prep_stride=1)
    with pytest.raises(RuntimeError, match="auto_stride"):
        orch._compute_auto_stride(10.0)


# ---------------------------------------------------------------------------
# 6. preprocessing-stride floor
# ---------------------------------------------------------------------------

def test_preprocessing_stride_floor_total_effective_never_below_prep_stride():
    """The effective total stride (prep_stride * runtime_stride) must be >= prep_stride
    for every tau — since runtime_stride is always >= 1."""
    frame_dt_ps = 100.0
    for prep_stride in (1, 5, 10):
        orch = _make_orchestrator(
            auto_stride=True, frame_dt_ps=frame_dt_ps, prep_stride=prep_stride
        )
        for tau_ns in (0.1, 1.0, 5.0, 10.0, 50.0):
            rs = orch._compute_auto_stride(tau_ns)
            assert rs >= 1, (prep_stride, tau_ns, rs)
            effective = prep_stride * rs
            assert effective >= prep_stride, (prep_stride, tau_ns, effective)


def test_preprocessing_stride_floor_when_target_below_cache_density():
    """If the formula's target is < prep_stride (i.e. the cache is already
    coarser than auto wants), runtime_stride clamps to 1 and the effective
    total stays at prep_stride.  This verifies the floor behavior."""
    # prep_stride=10, frame_dt=100 -> cache_dt=1000ps.  For tau=1ns (1000ps):
    # formula says max(1, floor(1000 / 10000)) = 1.  Effective = 10 (prep floor).
    orch = _make_orchestrator(
        auto_stride=True, frame_dt_ps=100.0, prep_stride=10
    )
    runtime_stride = orch._compute_auto_stride(1.0)
    assert runtime_stride == 1
    effective = 10 * runtime_stride
    assert effective == 10


# ---------------------------------------------------------------------------
# 7. analysis-side stride  (job 877 τ=50 ns failure, 2026-08-03)
#
# runtime_stride is a PAIR subsample: VAMPNetDataset applies it to the
# (t0, t1) index lists after the pairs are built (vampnet_dataset.py:184-190).
# It does not change the time grid.  `stride` is the preprocessing frame stride
# and DOES change the time grid.  Folding one into the other for the analysis
# phase changes the analysis's time resolution, which:
#   (a) makes lag times that are fine at the prep grid indivisible at the
#       coarser one — τ=50 ns died with "cannot be achieved with timestep of
#       200.0 ps and stride of 20"; and
#   (b) forces a cache miss, because prep wrote its cache at the prep stride.
# ---------------------------------------------------------------------------

def _make_analysis_args(orch, lag_time_ns, model_path=None):
    """Call _create_analysis_args with minimal filler."""
    import tempfile
    dirs = {"cache": None, "preparation": Path(tempfile.gettempdir()) / "no_such_prep_dir"}
    if model_path is None:
        model_path = f"training/lag{lag_time_ns}ns_10states/20260730_124941/models/best_model.pt"
    orch.config.lag_times = [lag_time_ns]
    return orch._create_analysis_args(dirs, model_path, "/tmp/test_analysis_dummy")


def test_analysis_stride_does_not_fold_in_runtime_stride():
    """Analysis must keep the preprocessing time grid, not the pair-subsampled one.

    GTT τ=50 ns: frame_dt 0.2 ns, prep stride 10 -> cache grid 2 ns.
    auto_stride gives runtime_stride 2, and folding it in made the analysis grid
    4 ns, which does not divide 50 ns.
    """
    orch = _make_orchestrator(
        auto_stride=True, frame_dt_ps=200.0, prep_stride=10, stride=10, timestep_ns=0.2
    )
    args = _make_analysis_args(orch, 50.0)
    assert args.stride == 10, (
        f"analysis stride is {args.stride}; folding runtime_stride into the frame "
        "stride changes the analysis time grid and breaks lag divisibility"
    )


@pytest.mark.parametrize("lag_ns,prep_stride", [
    (5.0, 5), (10.0, 10), (20.0, 10), (50.0, 10), (100.0, 10), (500.0, 10),
])
def test_analysis_lag_is_divisible_by_effective_timestep(lag_ns, prep_stride):
    """Every rung of the GTT ladder must survive VAMPNetDataset's lag validation.

    That check is `lag_ps % (frame_dt_ps * stride) != 0 -> raise`
    (vampnet_dataset.py:668). Reproduced here so a stride change that breaks a rung
    fails in unit tests instead of three hours into a cluster job.
    """
    frame_dt_ps = 200.0
    orch = _make_orchestrator(
        auto_stride=True, frame_dt_ps=frame_dt_ps, prep_stride=prep_stride,
        stride=prep_stride, timestep_ns=0.2,
    )
    args = _make_analysis_args(orch, lag_ns)

    effective_dt_ps = frame_dt_ps * args.stride
    lag_ps = lag_ns * 1000.0
    assert lag_ps % effective_dt_ps == 0, (
        f"τ={lag_ns}ns with analysis stride {args.stride} gives an effective "
        f"timestep of {effective_dt_ps}ps; {lag_ps} % {effective_dt_ps} != 0, so "
        "VAMPNetDataset will reject this lag and Phase 3 dies"
    )


def test_analysis_stride_matches_prep_so_the_cache_is_reused():
    """Analysis must look for the cache prep actually wrote.

    Prep wrote vampnet_data_<hash>_lag50.0_nn10_str10_cont.pkl; the folded stride
    made analysis look for _str20_ and rebuild the whole dataset (job 877 logged
    "No cache file found at ..._str20_cont.pkl" immediately before failing).
    """
    prep_stride = 10
    orch = _make_orchestrator(
        auto_stride=True, frame_dt_ps=200.0, prep_stride=prep_stride,
        stride=prep_stride, timestep_ns=0.2,
    )
    for lag in (5.0, 10.0, 20.0, 50.0, 100.0, 500.0):
        args = _make_analysis_args(orch, lag)
        assert args.stride == prep_stride, (
            f"τ={lag}: analysis stride {args.stride} != prep stride {prep_stride}, "
            "so the prepared cache cannot be reused"
        )


def test_analysis_stride_prefers_the_stride_prep_actually_used():
    """When prep applied a stride different from the CLI value, follow prep.

    dataset_stats.json is authoritative: it records what the cache on disk was
    actually built with, and that is what the cache lookup and the time grid must
    agree with.
    """
    orch = _make_orchestrator(
        auto_stride=True, frame_dt_ps=200.0, prep_stride=4, stride=10, timestep_ns=0.2
    )
    args = _make_analysis_args(orch, 20.0)
    assert args.stride == 4, (
        f"analysis used {args.stride}, but prep built the cache at stride 4"
    )
