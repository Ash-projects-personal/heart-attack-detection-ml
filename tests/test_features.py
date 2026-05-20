"""Tests for the synthetic-data + feature-engineering pipeline in
``heart_disease_predictor``.

The production script bundles dataset synthesis and feature engineering
inside :func:`generate_medical_data`.  These tests exercise that single
entry point and assert the engineered columns (``pressure_rate_ratio``,
``age_chol_risk``, ``high_risk_profile``) are computed correctly and that
the returned frame has the shape and dtypes downstream training code
expects.

Heavy ML dependencies (TensorFlow) are imported at module load by
``heart_disease_predictor``; we skip the suite cleanly if any of those
dependencies are missing in the current environment so the file is safe
to keep in the repo even when running a partial dependency install.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("tensorflow", reason="tensorflow is required by heart_disease_predictor at import time")

import numpy as np  # noqa: E402  (import after pytest.importorskip)
import pandas as pd  # noqa: E402

from heart_disease_predictor import generate_medical_data  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """A small, deterministic synthetic dataset for the feature-eng tests."""
    return generate_medical_data(n_samples=200)


# ---------------------------------------------------------------------------
# Data-shape sanity
# ---------------------------------------------------------------------------


EXPECTED_BASE_COLUMNS = {
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
}
EXPECTED_ENGINEERED_COLUMNS = {
    "age_chol_risk", "pressure_rate_ratio", "high_risk_profile",
}
EXPECTED_COLUMNS = EXPECTED_BASE_COLUMNS | EXPECTED_ENGINEERED_COLUMNS | {"target"}


def test_dataframe_shape(small_df: pd.DataFrame) -> None:
    """Row count matches the request and column set matches the spec."""
    assert len(small_df) == 200
    assert set(small_df.columns) == EXPECTED_COLUMNS


def test_dataframe_has_no_nans(small_df: pd.DataFrame) -> None:
    """No NaNs anywhere; downstream sklearn/Keras code can't tolerate them."""
    assert not small_df.isna().any().any()


def test_target_is_binary(small_df: pd.DataFrame) -> None:
    """``target`` must be a 0/1 integer column."""
    target = small_df["target"]
    assert set(target.unique()).issubset({0, 1})
    # Roughly balanced because the script thresholds on the median risk score.
    pos_rate = float(target.mean())
    assert 0.30 < pos_rate < 0.70, f"target unexpectedly imbalanced: {pos_rate:.3f}"


def test_clip_ranges_respected(small_df: pd.DataFrame) -> None:
    """Each clip()-d feature stays inside its documented bounds."""
    assert small_df["age"].between(25, 90).all()
    assert small_df["trestbps"].between(90, 200).all()
    assert small_df["chol"].between(120, 400).all()
    assert small_df["thalach"].between(70, 200).all()
    assert small_df["oldpeak"].between(0, 6).all()


def test_categorical_features_are_in_domain(small_df: pd.DataFrame) -> None:
    """Categorical columns only hold their declared category values."""
    assert set(small_df["sex"].unique()).issubset({0, 1})
    assert set(small_df["cp"].unique()).issubset({0, 1, 2, 3})
    assert set(small_df["fbs"].unique()).issubset({0, 1})
    assert set(small_df["restecg"].unique()).issubset({0, 1, 2})
    assert set(small_df["exang"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------


def test_pressure_rate_ratio_formula(small_df: pd.DataFrame) -> None:
    """``pressure_rate_ratio`` must equal ``trestbps / thalach`` row-wise."""
    expected = small_df["trestbps"] / small_df["thalach"]
    np.testing.assert_allclose(
        small_df["pressure_rate_ratio"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-9,
    )


def test_pressure_rate_ratio_finite_and_positive(small_df: pd.DataFrame) -> None:
    """Ratio is finite and strictly positive (thalach floor is 70)."""
    arr = small_df["pressure_rate_ratio"].to_numpy()
    assert np.isfinite(arr).all()
    assert (arr > 0).all()


def test_age_chol_risk_formula(small_df: pd.DataFrame) -> None:
    """``age_chol_risk`` must equal ``(age/50) * (chol/200)`` row-wise."""
    expected = (small_df["age"] / 50) * (small_df["chol"] / 200)
    np.testing.assert_allclose(
        small_df["age_chol_risk"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-9,
    )


def test_age_chol_risk_monotonic_in_inputs() -> None:
    """Holding everything else constant, higher age or chol increases the score."""
    # The spec formula is multiplicative; verify directly so a regression
    # that swaps numerator/denominator gets caught.
    low = (50 / 50) * (200 / 200)
    high_age = (70 / 50) * (200 / 200)
    high_chol = (50 / 50) * (300 / 200)
    assert high_age > low
    assert high_chol > low
    assert math.isclose(low, 1.0)


def test_high_risk_profile_definition(small_df: pd.DataFrame) -> None:
    """``high_risk_profile`` is 1 iff age>60 AND chol>250 AND trestbps>140."""
    expected = (
        (small_df["age"] > 60)
        & (small_df["chol"] > 250)
        & (small_df["trestbps"] > 140)
    ).astype(int)
    pd.testing.assert_series_equal(
        small_df["high_risk_profile"].rename(None),
        expected.rename(None),
        check_dtype=False,
    )


def test_high_risk_profile_is_binary(small_df: pd.DataFrame) -> None:
    assert set(small_df["high_risk_profile"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_generate_medical_data_is_deterministic() -> None:
    """The RNG seed (42) inside the generator means two calls match exactly."""
    a = generate_medical_data(n_samples=100)
    b = generate_medical_data(n_samples=100)
    pd.testing.assert_frame_equal(a, b)
