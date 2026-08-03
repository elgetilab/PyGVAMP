"""
Tests for requeue-resumable runs (job 877 post-mortem, 2026-08-03).

The GTT lag sweep lost 11 attempts to node crashes because every requeue starts a
fresh timestamped experiment directory and retrains from epoch 0.  Turning on the
existing `--resume` flag is NOT safe until the checks below hold, because
`best_model.pt` is written on every validation improvement from epoch 1
(vampnet.py:994-998) — so a run killed at epoch 61/100 leaves one behind, and
`run_training_phase` would skip it and feed an undertrained model to analysis and
the JSD retrain loop with nothing recording that it is undertrained.

The contract these tests pin:
  * a model is "complete" only if a completion marker says so AND the marker's
    epochs_run matches what was requested
  * a bare best_model.pt with no marker means INTERRUPTED -> retrain
  * the same rule for analysis, which is the ~3.5h phase and must not silently
    re-run (or silently skip) on resume
"""

import argparse
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from pygv.config.base_config import BaseConfig
from pygv.pipe.master_pipeline import PipelineOrchestrator


EPOCHS_REQUESTED = 100


def _make_orchestrator(tmp_path, *, epochs=EPOCHS_REQUESTED, lag=10.0, n_states=5):
    cfg = BaseConfig()
    cfg.output_dir = str(tmp_path)
    cfg.protein_name = "gtt"
    cfg.lag_times = [lag]
    cfg.n_states_list = [n_states]
    cfg.epochs = epochs
    cfg.cache = False
    cfg.auto_stride = False
    cfg.timestep = 0.2
    orch = PipelineOrchestrator(cfg)
    orch._frame_dt_ps = 200.0
    orch._prep_stride = 10
    return orch


def _make_dirs(tmp_path):
    dirs = {
        "root": tmp_path,
        "preparation": tmp_path / "preparation",
        "training": tmp_path / "training",
        "analysis": tmp_path / "analysis",
        "cache": None,
        "logs": tmp_path / "logs",
    }
    for p in dirs.values():
        if p is not None:
            p.mkdir(parents=True, exist_ok=True)
    return dirs


def _plant_model(dirs, exp_name="lag10.0ns_5states", *, marker=None):
    """Create a training dir holding a best_model.pt, optionally with a completion marker.

    Mirrors the on-disk layout the orchestrator produces:
        training/<exp_name>/<timestamp>/models/best_model.pt
    """
    model_dir = dirs["training"] / exp_name / "20260730_124941" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best_model.pt").write_bytes(b"not-a-real-checkpoint")
    if marker is not None:
        (model_dir / "training_complete.json").write_text(json.dumps(marker))
    return model_dir


# ---------------------------------------------------------------------------
# 1. training completeness
# ---------------------------------------------------------------------------

def test_partial_training_is_not_treated_as_complete(tmp_path):
    """A best_model.pt with no completion marker means the run was interrupted.

    This is the job 877 failure mode: crash at epoch 61/100 leaves a best_model.pt
    that currently causes run_training_phase to skip the combo entirely.
    """
    orch = _make_orchestrator(tmp_path)
    dirs = _make_dirs(tmp_path)
    _plant_model(dirs)  # no marker

    with patch("pygv.pipe.master_pipeline.run_training") as mock_train:
        mock_train.return_value = str(tmp_path / "retrained.pt")
        orch.run_training_phase(dirs, dataset_path=str(tmp_path / "preparation"))

    assert mock_train.called, (
        "training was skipped for a model with no completion marker — an "
        "interrupted run would be published as a finished one"
    )


def test_marker_from_a_smaller_epoch_budget_is_not_reused(tmp_path):
    """A finished run does not satisfy a later request for MORE epochs.

    Reachable case: the sweep ran at --epochs 50, then is relaunched at --epochs 100.
    The old marker is valid but describes a smaller budget, so the model must be
    retrained rather than silently reused at 50 epochs.
    """
    orch = _make_orchestrator(tmp_path, epochs=100)
    dirs = _make_dirs(tmp_path)
    _plant_model(dirs, marker={"epochs_run": 50, "epochs_requested": 50,
                               "early_stopped": False})

    with patch("pygv.pipe.master_pipeline.run_training") as mock_train:
        mock_train.return_value = str(tmp_path / "retrained.pt")
        orch.run_training_phase(dirs, dataset_path=str(tmp_path / "preparation"))

    assert mock_train.called, (
        "a 50-epoch model was reused to satisfy a 100-epoch request"
    )


def test_early_stopped_run_counts_as_complete(tmp_path):
    """Early stopping is a legitimate finish, not an interruption.

    epochs_run < epochs_requested here, but the run converged and rerunning it
    would just early-stop again. Distinguishing this from a crash is the whole
    reason the marker records `early_stopped` rather than epoch counts alone.
    """
    orch = _make_orchestrator(tmp_path, epochs=EPOCHS_REQUESTED)
    dirs = _make_dirs(tmp_path)
    _plant_model(dirs, marker={"epochs_run": 61,
                               "epochs_requested": EPOCHS_REQUESTED,
                               "early_stopped": True})

    with patch("pygv.pipe.master_pipeline.run_training") as mock_train:
        mock_train.return_value = str(tmp_path / "should-not-be-called.pt")
        orch.run_training_phase(dirs, dataset_path=str(tmp_path / "preparation"))

    assert not mock_train.called, "an early-stopped (converged) model was retrained"


def test_complete_training_is_skipped(tmp_path):
    """The other direction: a genuinely finished model must NOT be retrained.

    Guards against over-correcting the fix into 'always retrain', which would make
    resume useless.
    """
    orch = _make_orchestrator(tmp_path, epochs=EPOCHS_REQUESTED)
    dirs = _make_dirs(tmp_path)
    _plant_model(
        dirs,
        marker={"epochs_run": EPOCHS_REQUESTED, "epochs_requested": EPOCHS_REQUESTED},
    )

    with patch("pygv.pipe.master_pipeline.run_training") as mock_train:
        mock_train.return_value = str(tmp_path / "should-not-be-called.pt")
        trained = orch.run_training_phase(dirs, dataset_path=str(tmp_path / "preparation"))

    assert not mock_train.called, "a completed model was retrained on resume"
    assert "lag10.0ns_5states" in trained


# ---------------------------------------------------------------------------
# 2. analysis completeness
# ---------------------------------------------------------------------------

def test_completed_analysis_is_skipped_on_resume(tmp_path):
    """Analysis is the ~3.5h phase; a finished one must not be redone on resume."""
    orch = _make_orchestrator(tmp_path)
    dirs = _make_dirs(tmp_path)
    exp_name = "lag10.0ns_5states"
    model_dir = _plant_model(
        dirs,
        exp_name,
        marker={"epochs_run": EPOCHS_REQUESTED, "epochs_requested": EPOCHS_REQUESTED},
    )

    analysis_dir = dirs["analysis"] / exp_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis_complete.json").write_text(json.dumps({"status": "ok"}))

    with patch("pygv.pipe.master_pipeline.run_analysis") as mock_analysis:
        mock_analysis.return_value = {}
        orch.run_analysis_phase(dirs, {exp_name: str(model_dir / "best_model.pt")})

    assert not mock_analysis.called, "a completed analysis was re-run on resume"


def test_incomplete_analysis_is_rerun_on_resume(tmp_path):
    """Output files without a marker mean the analysis was interrupted -> re-run.

    Job 877 task 3 is the real case: exit 0, analysis dir created, but Phase 3 had
    raised and the dir was empty.
    """
    orch = _make_orchestrator(tmp_path)
    dirs = _make_dirs(tmp_path)
    exp_name = "lag10.0ns_5states"
    model_dir = _plant_model(
        dirs,
        exp_name,
        marker={"epochs_run": EPOCHS_REQUESTED, "epochs_requested": EPOCHS_REQUESTED},
    )

    analysis_dir = dirs["analysis"] / exp_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "partial_plot.png").write_bytes(b"stub")  # no marker

    with patch("pygv.pipe.master_pipeline.run_analysis") as mock_analysis:
        mock_analysis.return_value = {}
        orch.run_analysis_phase(dirs, {exp_name: str(model_dir / "best_model.pt")})

    assert mock_analysis.called, "an interrupted analysis was treated as complete"


def test_skipped_analysis_still_carries_its_retrain_verdict(tmp_path):
    """Resume must not silently disable the JSD retrain loop.

    _run_retrain_loop reads results["diagnostic_report"] and skips any experiment
    whose report is None. If a skipped analysis returned a bare stub, a resumed run
    would quietly stop reducing k — and k*(tau) from the retrain loop is exactly
    what the GTT sweep exists to measure.
    """
    orch = _make_orchestrator(tmp_path)
    dirs = _make_dirs(tmp_path)
    exp_name = "lag10.0ns_10states"
    model_dir = _plant_model(
        dirs, exp_name,
        marker={"epochs_run": EPOCHS_REQUESTED, "epochs_requested": EPOCHS_REQUESTED},
    )

    analysis_dir = dirs["analysis"] / exp_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis_complete.json").write_text(json.dumps({
        "exp_name": exp_name,
        "diagnostic": {
            "recommendation": "retrain",
            "effective_n_states": 9,
            "original_n_states": 10,
        },
    }))

    with patch("pygv.pipe.master_pipeline.run_analysis") as mock_analysis:
        results = orch.run_analysis_phase(dirs, {exp_name: str(model_dir / "best_model.pt")})

    assert not mock_analysis.called
    report = results[exp_name].get("diagnostic_report")
    assert report is not None, "skipped analysis lost its diagnostic verdict"
    assert report.recommendation == "retrain"
    assert report.effective_n_states == 9


def test_analysis_marker_persists_the_diagnostic_verdict(tmp_path):
    """The verdict must be written at analysis time, or there is nothing to restore."""
    orch = _make_orchestrator(tmp_path)
    dirs = _make_dirs(tmp_path)
    exp_name = "lag10.0ns_10states"
    model_dir = _plant_model(
        dirs, exp_name,
        marker={"epochs_run": EPOCHS_REQUESTED, "epochs_requested": EPOCHS_REQUESTED},
    )

    fake_report = argparse.Namespace(
        recommendation="retrain", effective_n_states=9, original_n_states=10,
    )
    with patch("pygv.pipe.master_pipeline.run_analysis") as mock_analysis:
        mock_analysis.return_value = {"diagnostic_report": fake_report}
        orch.run_analysis_phase(dirs, {exp_name: str(model_dir / "best_model.pt")})

    marker = json.loads((dirs["analysis"] / exp_name / "analysis_complete.json").read_text())
    assert marker["diagnostic"] == {
        "recommendation": "retrain",
        "effective_n_states": 9,
        "original_n_states": 10,
    }


def test_interrupted_retrain_round_is_not_discovered_as_a_finished_model(tmp_path):
    """The retrain loop's own models need the same completion rule.

    run_training_phase never iterates over retrained experiments
    (lag10ns_9states_retrained); they reach analysis via _discover_trained_models.
    An interrupted retrain round left there would be analysed and reported as a
    converged k — a wrong k*(tau) datapoint, which is the sweep's actual output.
    """
    orch = _make_orchestrator(tmp_path)
    dirs = _make_dirs(tmp_path)
    _plant_model(dirs, "lag10.0ns_10states",
                 marker={"epochs_run": EPOCHS_REQUESTED,
                         "epochs_requested": EPOCHS_REQUESTED})
    _plant_model(dirs, "lag10ns_9states_retrained")  # interrupted: no marker

    discovered = orch._discover_trained_models(dirs)

    assert "lag10.0ns_10states" in discovered
    assert "lag10ns_9states_retrained" not in discovered, (
        "an interrupted retrain round was discovered as a finished model"
    )


# ---------------------------------------------------------------------------
# 3. directory pinning — a requeued job must land in the same place
# ---------------------------------------------------------------------------

def test_exp_name_gives_a_stable_experiment_directory(tmp_path):
    """Two invocations with the same --exp_name share one directory.

    Without this, every requeue mints exp_<protein>_<timestamp> and rebuilds the
    cache and all training from scratch — the job 877 failure.
    """
    first = _make_orchestrator(tmp_path)
    first.config.exp_name = "exp_gtt_s0_lag10"
    dirs_a = first.setup_experiment_directory()

    second = _make_orchestrator(tmp_path)
    second.config.exp_name = "exp_gtt_s0_lag10"
    dirs_b = second.setup_experiment_directory()

    assert dirs_a["root"] == dirs_b["root"]
    assert dirs_a["root"].name == "exp_gtt_s0_lag10"


def test_default_experiment_directory_is_still_timestamped(tmp_path):
    """Backward compat: without --exp_name the old timestamped naming is kept."""
    orch = _make_orchestrator(tmp_path)
    dirs = orch.setup_experiment_directory()
    assert dirs["root"].name.startswith("exp_gtt_")
    assert dirs["root"].name != "exp_gtt_"


def test_resume_auto_resolves_to_the_newest_experiment_dir(tmp_path):
    """A requeued job cannot know its first attempt's timestamp, so 'auto' finds it."""
    orch = _make_orchestrator(tmp_path)
    (tmp_path / "exp_gtt_20260730_090343").mkdir()
    (tmp_path / "exp_gtt_20260803_082405").mkdir()

    dirs = orch.resume_experiment_directory("auto")
    assert dirs["root"].name == "exp_gtt_20260803_082405"


def test_resume_training_reuses_the_interrupted_run_directory(tmp_path):
    """Resuming must train back into the interrupted run's own directory.

    A fresh timestamp would create a sibling directory and never find the previous
    resume_state.pt, silently degrading resume into a full restart.
    """
    orch = _make_orchestrator(tmp_path)
    orch.config.resume_training = True
    dirs = _make_dirs(tmp_path)
    _plant_model(dirs)  # interrupted: no marker, run dir '20260730_124941'

    captured = {}

    def _capture(train_args):
        captured["run_name"] = train_args.run_name
        return str(tmp_path / "model.pt")

    with patch("pygv.pipe.master_pipeline.run_training", side_effect=_capture):
        orch.run_training_phase(dirs, dataset_path=str(tmp_path / "preparation"))

    assert captured["run_name"] == "20260730_124941", (
        "resumed training was pointed at a new run directory, so the previous "
        "resume_state.pt would be invisible"
    )


# ---------------------------------------------------------------------------
# 4. trainer checkpoint round-trip
# ---------------------------------------------------------------------------

def test_resume_state_round_trip_restores_optimizer_and_epoch(tmp_path):
    """resume_state.pt must carry optimizer moments and the epoch, not just weights.

    Without the optimizer state a resumed run restarts Adam's moment estimates,
    which is a different optimization trajectory than the one it claims to continue.
    """
    import torch
    from tests.test_warm_start import _make_vampnet

    model = _make_vampnet(n_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Take a real step so the optimizer has non-trivial state to restore.
    loss = sum(p.sum() for p in model.parameters())
    loss.backward()
    optimizer.step()

    path = str(tmp_path / "resume_state.pt")
    model._save_resume_state(
        path, epoch=60, optimizer=optimizer, scheduler=None,
        best_score=8.61, plateau_ref=8.61, no_improvement_count=2,
        global_batch=1234, history={"train_scores": [1.0]},
    )

    fresh = _make_vampnet(n_classes=5)
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=5e-4)
    (start_epoch, best_score, plateau_ref, no_improve, global_batch, history,
     best_state) = fresh._load_resume_state(path, fresh_opt, None, {},
                                            device="cpu", verbose=False)

    assert start_epoch == 61, "resume must continue at the epoch AFTER the last completed one"
    assert best_score == pytest.approx(8.61)
    assert plateau_ref == pytest.approx(8.61)
    assert no_improve == 2
    assert global_batch == 1234
    assert history == {"train_scores": [1.0]}
    assert fresh_opt.state_dict()["state"], "optimizer moments were not restored"
    assert best_state is None  # none was supplied to _save_resume_state here


def test_resume_carries_the_best_weights_not_just_the_current_ones(tmp_path):
    """best_score and the weights it refers to must stay consistent across a resume.

    The best model can be several epochs behind the checkpoint. Seeding the resumed
    run's best_model_state from the *current* weights would pair epoch-N weights
    with an epoch-M score, and fit() loads best_model_state back into the model
    before returning — so the in-memory model handed to the ITS/CK analysis would
    not be the model the reported score describes.
    """
    import torch
    from tests.test_warm_start import _make_vampnet

    model = _make_vampnet(n_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Move the live weights away from the best ones.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    path = str(tmp_path / "resume_state.pt")
    model._save_resume_state(
        path, epoch=40, optimizer=optimizer, scheduler=None,
        best_score=9.0, plateau_ref=9.0, no_improvement_count=10,
        global_batch=1, history={}, best_model_state=best_weights,
    )

    fresh = _make_vampnet(n_classes=5)
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=5e-4)
    *_, restored_best = fresh._load_resume_state(path, fresh_opt, None, {},
                                                 device="cpu", verbose=False)

    assert restored_best is not None, "best weights were lost across the resume"
    for k, v in best_weights.items():
        assert torch.allclose(restored_best[k], v), (
            f"{k} came back as the checkpoint weights, not the best weights"
        )


def test_resume_state_is_written_atomically(tmp_path):
    """No .tmp file may survive, and the target must be complete after the call."""
    import torch
    from tests.test_warm_start import _make_vampnet

    model = _make_vampnet(n_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    path = tmp_path / "resume_state.pt"
    model._save_resume_state(
        str(path), epoch=9, optimizer=optimizer, scheduler=None,
        best_score=1.0, plateau_ref=1.0, no_improvement_count=0,
        global_batch=1, history={},
    )

    assert path.is_file()
    assert not (tmp_path / "resume_state.pt.tmp").exists()
    assert torch.load(str(path), weights_only=False)["epoch"] == 9


# ---------------------------------------------------------------------------
# 4b. resume must not break training loops that cannot resume
# ---------------------------------------------------------------------------

def test_fit_without_resume_support_is_untouched_when_not_resuming(tmp_path):
    """RevVAMPNet.fit() has its own loop and no resume_state_path parameter.

    Passing the kwarg unconditionally killed every reversible run with
    "unexpected keyword argument 'resume_state_path'" — caught by
    test_phase5_integration. The kwarg may only be passed when resuming.
    """
    import inspect
    from pygv.vampnet.rev_vampnet import RevVAMPNet
    from pygv.pipe import training as training_mod

    assert 'resume_state_path' not in inspect.signature(RevVAMPNet.fit).parameters, (
        "this test guards the case where fit() lacks resume support; if RevVAMPNet "
        "gained it, extend the resume path to cover it instead of deleting this test"
    )
    src = inspect.getsource(training_mod.train_model)
    assert 'resume_state_path=' not in src.split('model.fit(')[1], (
        "resume_state_path is passed positionally/unconditionally into model.fit(); "
        "it must only go in via fit_kwargs when --resume_training is set"
    )


def test_resume_requested_against_unsupported_fit_fails_loudly(tmp_path):
    """Asking for resume where it cannot work must raise, not silently degrade.

    Silently dropping the request would leave a multi-hour reversible run believing
    it is crash-safe when every crash still costs the whole run.
    """
    import argparse as _argparse
    from pygv.pipe import training as training_mod

    class _NoResumeModel:
        def fit(self, **kwargs):  # no resume_state_path parameter
            return {}

    args = _argparse.Namespace(
        resume_training=True, epochs=10, save_every=10, cpu=True,
        clip_grad=None, sample_validate_every=100, lr=1e-3,
    )
    paths = {'model_dir': str(tmp_path), 'scores_plot': str(tmp_path / 'p.png')}

    with pytest.raises(RuntimeError, match="does not support resuming"):
        training_mod.train_model(args=args, model=_NoResumeModel(),
                                 train_loader=None, test_loader=None, paths=paths)


# ---------------------------------------------------------------------------
# 5. end-to-end: an interrupted fit() actually continues where it stopped
# ---------------------------------------------------------------------------

def test_interrupted_fit_continues_from_checkpoint(tmp_path):
    """The whole point, exercised end to end.

    Train 4 epochs (checkpointing every 2), then resume the same save_dir with a
    budget of 8. Only epochs 5-8 may run — a resumed run that silently restarts at
    epoch 0 would report 8 fresh epochs and burn the time the resume was meant to
    save.
    """
    import torch
    from torch_geometric.loader import DataLoader
    from tests.test_training import SyntheticTimeLaggedDataset
    from tests.test_warm_start import _make_vampnet

    def _loaders():
        torch.manual_seed(0)
        ds = SyntheticTimeLaggedDataset(n_samples=32, num_nodes=6, node_dim=8, edge_dim=4)
        return DataLoader(ds, batch_size=16, shuffle=False)

    save_dir = str(tmp_path / "models")

    # First leg: 4 epochs, checkpoint every 2 -> resume_state.pt at epoch 4.
    model = _make_vampnet(n_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.fit(train_loader=_loaders(), test_loader=_loaders(), optimizer=opt,
              n_epochs=4, device="cpu", save_dir=save_dir, save_every=2,
              verbose=False, plot_scores=False)

    state_path = tmp_path / "models" / "resume_state.pt"
    assert state_path.is_file(), "no resume_state.pt was written"
    assert torch.load(str(state_path), weights_only=False)["epoch"] == 3  # 0-based

    # Second leg: fresh model + optimizer, resumed to a budget of 8.
    resumed = _make_vampnet(n_classes=3)
    resumed_opt = torch.optim.Adam(resumed.parameters(), lr=1e-3)
    history = resumed.fit(train_loader=_loaders(), test_loader=_loaders(),
                          optimizer=resumed_opt, n_epochs=8, device="cpu",
                          save_dir=save_dir, save_every=2, verbose=False,
                          plot_scores=False, resume_state_path=str(state_path))

    assert history["epochs_run"] == 8, "resumed run did not reach the new budget"
    # 4 epochs already done, so this leg may only have trained the remaining 4.
    assert len(history["epoch_val_scores"]) == 8, (
        f"expected 8 total epochs of history, got {len(history['epoch_val_scores'])} — "
        "the resumed run restarted from epoch 0 instead of continuing"
    )
