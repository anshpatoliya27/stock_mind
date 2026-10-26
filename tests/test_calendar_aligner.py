"""Unit tests for src.data.calendar_aligner module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.calendar_aligner import (
    MarketCalendarAligner,
    estimate_india_cpi_release_date,
)


def test_estimate_india_cpi_release_date():
    # Jan 2024 observation -> released around Feb 12, 2024
    rel_jan = estimate_india_cpi_release_date("2024-01-01")
    assert rel_jan == pd.Timestamp("2024-02-12")

    # Dec 2023 observation -> released around Jan 12, 2024
    rel_dec = estimate_india_cpi_release_date("2023-12-01")
    assert rel_dec == pd.Timestamp("2024-01-12")


def test_align_macro_anti_lookahead():
    aligner = MarketCalendarAligner()

    # Trading sessions in Jan and Feb 2024
    trading_df = pd.DataFrame({
        "date": [
            "2024-01-15",
            "2024-01-31",
            "2024-02-10",
            "2024-02-12",  # Release day
            "2024-02-13",  # Subsequent trading day
            "2024-02-28",
        ],
        "close": [21500, 21700, 21800, 21750, 21850, 22000],
    })

    # CPI macro record for Jan 2024: observation Jan 1, official release Feb 12
    macro_df = pd.DataFrame({
        "observation_date": ["2024-01-01"],
        "release_date": ["2024-02-12"],
        "cpi_index": [185.2],
    })

    aligned, report = align_macro_with_release_dates = aligner.align_macro_with_release_dates(
        trading_df=trading_df,
        macro_df=macro_df,
        trading_date_col="date",
        macro_release_col="release_date",
        macro_obs_col="observation_date",
        value_cols=["cpi_index"],
    )

    # 1. Lookahead check: Before Feb 12, CPI must NOT be visible!
    assert pd.isna(aligned.loc[aligned["date"] == "2024-01-15", "cpi_index"].iloc[0])
    assert pd.isna(aligned.loc[aligned["date"] == "2024-01-31", "cpi_index"].iloc[0])
    assert pd.isna(aligned.loc[aligned["date"] == "2024-02-10", "cpi_index"].iloc[0])

    # 2. On and after Feb 12, CPI becomes visible
    assert aligned.loc[aligned["date"] == "2024-02-12", "cpi_index"].iloc[0] == 185.2
    assert aligned.loc[aligned["date"] == "2024-02-13", "cpi_index"].iloc[0] == 185.2

    # 3. Provenance flags
    feb12_row = aligned[aligned["date"] == "2024-02-12"].iloc[0]
    feb13_row = aligned[aligned["date"] == "2024-02-13"].iloc[0]

    assert bool(feb12_row["is_forward_filled"]) is False  # Exact release date
    assert bool(feb13_row["is_forward_filled"]) is True   # Forward filled

    # Lag days relative to observation date (Jan 1 to Feb 12 = 42 days)
    assert feb12_row["lag_days"] == 42
    assert feb13_row["lag_days"] == 43

    assert report["lookahead_violations_detected"] == 0
    assert report["validation_status"] == "PASSED"


def test_align_forex_rates():
    aligner = MarketCalendarAligner()
    trading_df = pd.DataFrame({
        "date": ["2024-01-05", "2024-01-08"],  # Friday and Monday
        "close": [21500, 21600],
    })
    forex_df = pd.DataFrame({
        "date": ["2024-01-05"],  # Published Friday
        "USD": [83.15],
    })

    merged, report = aligner.align_forex_rates(trading_df, forex_df)
    assert len(merged) == 2
    # Monday receives Friday's rate backward asof
    assert merged.loc[merged["date"] == "2024-01-08", "USD"].iloc[0] == 83.15
    assert report["unaligned_sessions"] == 0
