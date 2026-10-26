"""Unit tests for src.data.validation_reporter module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation_reporter import DataValidationReporter


def test_validation_reporter_clean_dataframe():
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "symbol": ["INFY", "INFY", "INFY"],
        "open": [1500.0, 1510.0, 1520.0],
        "high": [1530.0, 1540.0, 1550.0],
        "low": [1490.0, 1500.0, 1510.0],
        "close": [1515.0, 1525.0, 1535.0],
        "volume": [100000, 120000, 110000],
    })

    rep = DataValidationReporter.audit_dataframe(
        df,
        dataset_name="clean_test_stock",
        date_col="date",
        price_cols=["open", "high", "low", "close"],
        volume_col="volume",
        key_cols=["date", "symbol"],
    )

    assert rep["validation_grade"] == "PASSED"
    assert rep["duplicate_records"]["key_collisions"] == 0
    assert rep["suspicious_prices"]["high_less_than_low"] == 0
    assert rep["suspicious_prices"]["non_positive_prices"] == 0


def test_validation_reporter_anomalies():
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01", "1980-05-01"],  # Collision + out-of-epoch
        "symbol": ["TCS", "TCS", "TCS"],
        "open": [3200.0, 3210.0, 3220.0],
        "high": [3100.0, 3300.0, 3250.0],  # Row 0: High (3100) < Low (3150)! Microstructure violation
        "low": [3150.0, 3180.0, 3190.0],
        "close": [-10.0, 3250.0, 3210.0],   # Row 0: Negative price!
        "volume": [50000, -500, 60000],     # Row 1: Negative volume!
    })

    rep = DataValidationReporter.audit_dataframe(
        df,
        dataset_name="corrupted_test_stock",
        date_col="date",
        price_cols=["open", "high", "low", "close"],
        volume_col="volume",
        key_cols=["date", "symbol"],
    )

    assert rep["validation_grade"] == "CRITICAL_ISSUES_FOUND"
    assert rep["duplicate_records"]["key_collisions"] == 1
    assert rep["invalid_dates"]["out_of_epoch_dates"] == 1
    assert rep["suspicious_prices"]["high_less_than_low"] == 1
    assert rep["suspicious_prices"]["non_positive_prices"] >= 1
    assert len(rep["critical_issues"]) >= 3

    md = DataValidationReporter.render_markdown(rep)
    assert "CRITICAL" in md
    assert "High < Low" in md
