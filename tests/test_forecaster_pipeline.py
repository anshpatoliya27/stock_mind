"""Unit and Integration Tests for StockMind Predictor Pipeline.

Validates:
1. Exact target calculation formulas on hand-built numerical series:
   - target_1d = (close.shift(-1) / close - 1) * 100
   - target_5d = (close.shift(-5) / close - 1) * 100
2. Multi-symbol rolling feature isolation (no cross-symbol leakage).
3. Chronological train/validation/test date isolation.
4. Preprocessing parameter isolation (fitted strictly on training set only).
5. Standalone inference schema validation and artifact serialization.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.dataset_builder import (
    DatasetBuilder,
    FEATURE_NAMES,
    PreprocessingPipeline,
    SplitConfig,
)
from src.models.forecaster import (
    ForecastResult,
    SchemaValidationError,
    StockForecaster,
)


def test_exact_target_calculation_hand_built():
    """Verify target_1d and target_5d match hand-computed values to 6 decimal places."""
    # Hand-built sequence of 7 observations for a single symbol
    # Observation 0: 100.0 -> next: 105.0 (+5.0%), 5-ahead: 115.0 (+15.0%)
    # Observation 1: 105.0 -> next: 110.25 (+5.0%), 5-ahead: 121.275 (+15.5%)
    # Observation 2: 110.25 -> next: 99.225 (-10.0%)
    dates = pd.date_range("2024-01-01", periods=7, freq="B")
    close_prices = [100.0, 105.0, 110.25, 99.225, 105.0, 115.0, 121.275]
    df = pd.DataFrame({
        "date": dates,
        "symbol": ["TEST"] * 7,
        "close": close_prices,
    })

    df_targets = DatasetBuilder.construct_targets(df)

    # 1. Verify Observation 0
    row_0 = df_targets.iloc[0]
    expected_target_1d_0 = (105.0 / 100.0 - 1.0) * 100.0  # +5.0%
    expected_target_5d_0 = (115.0 / 100.0 - 1.0) * 100.0  # +15.0%
    assert np.isclose(row_0["target_1d"], expected_target_1d_0, atol=1e-5)
    assert np.isclose(row_0["target_5d"], expected_target_5d_0, atol=1e-5)

    # 2. Verify Observation 2 (-10% next day)
    row_2 = df_targets.iloc[2]
    expected_target_1d_2 = (99.225 / 110.25 - 1.0) * 100.0  # -10.0%
    assert np.isclose(row_2["target_1d"], expected_target_1d_2, atol=1e-5)

    # 3. Verify terminal boundary behavior (last rows must be NaN, no forward leakage)
    assert not np.isnan(df_targets.iloc[0]["target_5d"])
    assert not np.isnan(df_targets.iloc[1]["target_5d"])
    assert np.isnan(df_targets.iloc[2]["target_5d"])
    assert np.isnan(df_targets.iloc[-1]["target_1d"])
    assert not np.isnan(df_targets.iloc[-2]["target_1d"])


def test_symbols_never_share_rolling_history():
    """Verify that symbol A's calculations are 100% isolated from symbol B."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")

    # Stock A has constant low price 10.0
    df_a = pd.DataFrame({
        "date": dates,
        "symbol": ["A_LOW"] * 30,
        "open": [10.0] * 30,
        "high": [10.5] * 30,
        "low": [9.5] * 30,
        "close": [10.0] * 30,
        "volume": [1000] * 30,
    })

    # Stock B has constant high price 10000.0
    df_b = pd.DataFrame({
        "date": dates,
        "symbol": ["B_HIGH"] * 30,
        "open": [10000.0] * 30,
        "high": [10050.0] * 30,
        "low": [9950.0] * 30,
        "close": [10000.0] * 30,
        "volume": [50000] * 30,
    })

    # Interleave rows randomly to test groupby isolation
    combined = pd.concat([df_a, df_b]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    features = DatasetBuilder.compute_symbol_features(combined)

    # Filter back to Stock A
    feat_a = features[features["symbol"] == "A_LOW"].sort_values("date")
    # All returns for constant price stock must be 0.0
    assert np.allclose(feat_a["ret_1d"].dropna(), 0.0)
    # EMA ratio for constant price stock must be 0.0 (close == ema)
    assert np.allclose(feat_a["ema_ratio_50"].dropna(), 0.0, atol=1e-4)
    # High price of B (10000) must NEVER leak into A's volatility or ATR
    assert (feat_a["natr_14"].dropna() < 15.0).all()


def test_chronological_splits_isolation():
    """Verify train, val, and test date ranges have zero overlap."""
    dates = pd.date_range("2019-01-01", "2026-06-30", freq="ME")
    df = pd.DataFrame({
        "date": dates,
        "symbol": ["TEST"] * len(dates),
        "close": np.linspace(100, 200, len(dates)),
    })

    config = SplitConfig(
        train_start="2019-01-01",
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2026-12-31",
    )

    splits = DatasetBuilder.split_chronological(df, config)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()


def test_preprocessing_isolation_train_only():
    """Verify that preprocessor statistics are fit ONLY on train and transform test cleanly."""
    np.random.seed(42)
    # Training data: mean 10.0, std 2.0
    X_train = pd.DataFrame({
        col: np.random.normal(loc=10.0, scale=2.0, size=100) for col in FEATURE_NAMES
    })
    # Add a NaN to check median imputation
    X_train.iloc[0, 0] = np.nan

    # Test data: distribution shifted to mean 50.0 (e.g. regime change)
    X_test = pd.DataFrame({
        col: np.random.normal(loc=50.0, scale=5.0, size=50) for col in FEATURE_NAMES
    })
    X_test.iloc[0, 0] = np.nan

    pipeline = PreprocessingPipeline(feature_names=FEATURE_NAMES)
    pipeline.fit(X_train)

    # Imputed median for column 0 must match train median (~10.0), NOT test median (~50.0)
    train_med_0 = pipeline.medians_[FEATURE_NAMES[0]]
    assert 9.0 <= train_med_0 <= 11.0

    # Transform test without refitting
    X_test_scaled = pipeline.transform(X_test)
    assert X_test_scaled.shape == (50, len(FEATURE_NAMES))
    assert not np.isnan(X_test_scaled).any()


def test_inference_schema_validation():
    """Verify that StockForecaster rejects invalid or missing feature schemas."""
    forecaster = StockForecaster(
        model_1d=Ridge(),
        model_5d=Ridge(),
        preprocessor=PreprocessingPipeline(feature_names=FEATURE_NAMES),
        feature_schema=FEATURE_NAMES,
    )

    # 1. Missing columns should raise SchemaValidationError
    incomplete_df = pd.DataFrame({"ret_1d": [1.2], "ret_2d": [0.5]})
    with pytest.raises(SchemaValidationError, match="missing required features"):
        forecaster.validate_features(incomplete_df)

    # 2. Non-numeric columns should raise SchemaValidationError
    invalid_type_df = pd.DataFrame({col: [1.0] for col in FEATURE_NAMES})
    invalid_type_df["ret_1d"] = "NON_NUMERIC_STRING"
    with pytest.raises(SchemaValidationError, match="cannot be cast to numeric"):
        forecaster.validate_features(invalid_type_df)

    # 3. Correct dataframe should pass and align columns
    valid_df = pd.DataFrame({col: [np.random.randn()] for col in FEATURE_NAMES})
    aligned = forecaster.validate_features(valid_df)
    assert list(aligned.columns) == FEATURE_NAMES


def test_forecaster_save_and_load(tmp_path: Path):
    """Verify full serialization, reloading, and prediction consistency."""
    np.random.seed(42)
    X = pd.DataFrame({col: np.random.randn(50) for col in FEATURE_NAMES})
    y1 = np.random.randn(50)
    y5 = np.random.randn(50)

    preprocessor = PreprocessingPipeline(feature_names=FEATURE_NAMES).fit(X)
    X_scaled = preprocessor.transform(X)

    m1 = Ridge().fit(X_scaled, y1)
    m5 = Ridge().fit(X_scaled, y5)

    forecaster = StockForecaster(
        model_1d=m1,
        model_5d=m5,
        preprocessor=preprocessor,
        feature_schema=FEATURE_NAMES,
        metadata={"version": "1.0-test"},
    )

    save_dir = tmp_path / "test_model_export"
    forecaster.save(save_dir)

    # Reload from disk
    loaded = StockForecaster.load(save_dir)
    assert loaded.feature_schema == FEATURE_NAMES
    assert loaded.metadata.get("version") == "1.0-test"

    # Test single-row prediction
    test_features = {col: 0.5 for col in FEATURE_NAMES}
    res: ForecastResult = loaded.predict(test_features, symbol="RELIANCE")
    assert isinstance(res, ForecastResult)
    assert res.symbol == "RELIANCE"
    assert isinstance(res.target_1d_pred_pct, float)
    assert isinstance(res.target_5d_pred_pct, float)
    assert res.direction_1d in ["UP", "DOWN"]
    assert res.direction_5d in ["UP", "DOWN"]


def test_prepare_features_from_history():
    """Verify single-stock rolling feature extraction from raw OHLCV buffer."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    history_df = pd.DataFrame({
        "date": dates,
        "open": np.linspace(100, 150, 60),
        "high": np.linspace(102, 153, 60),
        "low": np.linspace(98, 147, 60),
        "close": np.linspace(101, 151, 60),
        "volume": [100000] * 60,
    })

    feat_row = StockForecaster.prepare_features_from_history(history_df)
    assert isinstance(feat_row, pd.DataFrame)
    assert len(feat_row) == 1

    # Verify all expected feature columns are present and non-empty
    for col in FEATURE_NAMES:
        assert col in feat_row.columns
        assert not np.isnan(feat_row[col].iloc[0])
