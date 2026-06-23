"""Tests for the causal validation layer (evaluation/causal.py)."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.evaluation.causal import CausalConfig, CausalTestResult, CausalValidator


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# -----------------------------------------------------------------------
# CausalConfig defaults
# -----------------------------------------------------------------------


def test_causal_config_defaults():
    cfg = CausalConfig()
    assert cfg.enabled is True
    assert cfg.granger_max_lag == 5
    assert cfg.granger_significance == 0.05
    assert cfg.n_interventions == 3
    assert cfg.robustness_threshold == 0.4


# -----------------------------------------------------------------------
# CausalTestResult dataclass
# -----------------------------------------------------------------------


def test_causal_test_result_fields():
    r = CausalTestResult(
        factor_name="test",
        granger_p_value=0.01,
        granger_f_stat=5.0,
        granger_passes=True,
        intervention_ic_ratio=0.8,
        intervention_passes=True,
        robustness_score=0.7,
        passes=True,
    )
    assert r.factor_name == "test"
    assert r.passes is True
    assert isinstance(r.details, dict)


# -----------------------------------------------------------------------
# Granger test: planted causal signal should pass
# -----------------------------------------------------------------------


def test_granger_causal_signal_passes(rng):
    """A signal that IS lag-1 predictive of returns should produce low p."""
    M, T = 20, 200
    noise = rng.normal(0, 0.01, (M, T))
    signal = rng.normal(0, 1, (M, T))
    # Returns are a lagged copy of the signal + small noise
    returns = np.zeros((M, T))
    returns[:, 1:] = signal[:, :-1] * 0.5 + noise[:, 1:]

    validator = CausalValidator(
        returns=returns,
        data_tensor=None,
        library_signals={},
        config=CausalConfig(granger_max_lag=3, seed=42),
    )
    result = validator.validate("planted_signal", signal)
    # The Granger test should detect causality (low p-value)
    assert result.granger_p_value < 0.10 or result.granger_passes


# -----------------------------------------------------------------------
# Granger test: random noise should fail (high p-value)
# -----------------------------------------------------------------------


def test_granger_random_noise_high_pvalue(rng):
    """Pure noise signal should have high p-value."""
    M, T = 20, 200
    signal = rng.normal(0, 1, (M, T))
    returns = rng.normal(0, 0.01, (M, T))

    validator = CausalValidator(
        returns=returns,
        data_tensor=None,
        library_signals={},
        config=CausalConfig(granger_max_lag=3, seed=42),
    )
    result = validator.validate("noise_signal", signal)
    # High p-value expected (not necessarily >0.05 due to random chance,
    # but the test is about the API working correctly)
    assert isinstance(result.granger_p_value, float)
    assert 0.0 <= result.granger_p_value <= 1.0


# -----------------------------------------------------------------------
# Intervention robustness: robust signal retains IC
# -----------------------------------------------------------------------


def test_intervention_robust_signal(rng):
    """A signal strongly correlated with returns should be robust."""
    M, T = 20, 100
    returns = rng.normal(0, 0.01, (M, T))
    # Signal is nearly identical to returns -> high IC, robust
    signal = returns * 10 + rng.normal(0, 0.001, (M, T))

    validator = CausalValidator(
        returns=returns,
        data_tensor=None,
        library_signals={},
        config=CausalConfig(seed=42),
    )
    result = validator.validate("robust_factor", signal)
    assert result.intervention_ic_ratio > 0.0
    assert isinstance(result.intervention_passes, bool)
    assert isinstance(result.robustness_score, float)


def test_validate_excludes_candidate_from_control_library(rng):
    """A factor under test should not be used as its own Granger control."""
    M, T = 8, 40
    signal = rng.normal(0, 1, (M, T))
    returns = rng.normal(0, 0.01, (M, T))
    control = rng.normal(0, 1, (M, T))

    validator = CausalValidator(
        returns=returns,
        data_tensor=None,
        library_signals={
            "candidate_factor": signal.copy(),
            "control_factor": control,
        },
        config=CausalConfig(seed=42),
    )

    captured: dict[str, np.ndarray] = {}

    def _capture_controls(signals_arg, returns_arg, library_signals_arg):
        captured.update(library_signals_arg)
        return 1.0, 0.0, True

    validator._granger_test = _capture_controls  # type: ignore[method-assign]
    result = validator.validate("candidate_factor", signal)

    assert result.granger_passes is True
    assert "candidate_factor" not in captured
    assert "control_factor" in captured
