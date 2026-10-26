"""Unit tests for src.data.cleaners module."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.data.cleaners import (
    clean_bulk_deals_df,
    clean_bulk_deals_file,
    clean_world_bank_df,
    clean_world_bank_file,
    sanitize_column_names,
)


def test_sanitize_column_names():
    df = pd.DataFrame(columns=["Date ", "Symbol ", "Trade Price / Wght. Avg. Price ", "Remarks "])
    rename_map = sanitize_column_names(df)
    assert rename_map["Date "] == "date"
    assert rename_map["Symbol "] == "symbol"
    assert "trade_price" in rename_map["Trade Price / Wght. Avg. Price "]
    assert rename_map["Remarks "] == "remarks"


def test_clean_bulk_deals_df():
    data = {
        "Date ": ["27-OCT-2025", "28-OCT-2025", "INVALID-DATE", "27-OCT-2025"],
        "Symbol ": [" 360ONE ", "TCS", "INFY", " 360ONE "],
        "Client Name ": ["Client A", "Client B", "Client C", "Client A"],
        "Buy / Sell ": ["buy", "SELL", "buy", "buy"],
        "Quantity Traded ": ["43,85,856", "10,000", "invalid_qty", "43,85,856"],
        "Trade Price / Wght. Avg. Price ": ["1,179.70", "3,450.00", "-5.00", "1,179.70"],
        "Remarks ": ["-", "Normal trade", "-", "-"],
    }
    df = pd.DataFrame(data)
    cleaned, report = clean_bulk_deals_df(df)

    # Whitespace headers stripped
    assert "date" in cleaned.columns
    assert "symbol" in cleaned.columns

    # Check numeric conversions
    assert cleaned["quantity_traded"].iloc[0] == 4385856
    assert cleaned["trade_price_wght_avg_price"].iloc[0] == 1179.70
    assert cleaned["buy_sell"].iloc[0] == "BUY"

    # Remarks stripped
    assert cleaned["remarks"].iloc[0] is None
    assert cleaned["remarks"].iloc[1] == "Normal trade"

    # Validation findings
    assert report["invalid_dates"] == 1
    assert report["invalid_numeric_quantities"] == 1
    assert report["suspicious_prices"] == 1  # -5.00 is negative
    assert report["duplicate_rows"] == 1  # Row 0 and Row 3 are duplicates


def test_clean_bulk_deals_file(tmp_path: Path):
    test_csv = tmp_path / "test_bulk_deals.csv"
    data = {
        "Date ": ["27-OCT-2025"],
        "Symbol ": ["RELIANCE"],
        "Buy / Sell ": ["BUY"],
        "Quantity Traded ": ["1,00,000"],
        "Trade Price / Wght. Avg. Price ": ["2,500.50"],
    }
    pd.DataFrame(data).to_csv(test_csv, index=False)

    out_csv = tmp_path / "cleaned_bulk.csv"
    cleaned, report = clean_bulk_deals_file(test_csv, output_filepath=out_csv)

    assert len(cleaned) == 1
    assert out_csv.exists()
    assert report["file_source"] == str(test_csv)
    assert cleaned["quantity_traded"].iloc[0] == 100000


def test_clean_world_bank_df():
    data = {
        "Country Name": ["India", "United States", "India"],
        "Country Code": ["IND", "USA", "IND"],
        "Indicator Name": ["Inflation, consumer prices", "Inflation, consumer prices", "CPI 2010=100"],
        "Indicator Code": ["FP.CPI.TOTL.ZG", "FP.CPI.TOTL.ZG", "FP.CPI.TOTL"],
        "2020": ["6.62", "1.23", "150.2"],
        "2021": ["5.13", "4.70", "158.4"],
        "Unnamed: 6": [None, None, None],
    }
    df = pd.DataFrame(data)
    cleaned_long, report = clean_world_bank_df(df, country_codes=["IND"], melt_to_long=True)

    # Only India retained
    assert set(cleaned_long["country_code"].unique()) == {"IND"}
    # Unnamed dropped
    assert "Unnamed: 6" not in cleaned_long.columns
    # Reshaped to long
    assert "year" in cleaned_long.columns
    assert "indicator_value" in cleaned_long.columns
    assert len(cleaned_long) == 4  # 2 indicators * 2 years
    assert cleaned_long["year"].dtype in [int, "int64", "int32"]
    assert cleaned_long["indicator_value"].dtype in [float, "float64"]


def test_clean_world_bank_file(tmp_path: Path):
    test_csv = tmp_path / "test_wb.csv"
    # Write 4 dummy commentary lines then the header
    content = (
        "Data Source,World Development Indicators\n"
        "Last Updated Date,2024-03-28\n"
        "\n"
        "\n"
        "Country Name,Country Code,Indicator Name,Indicator Code,2022,2023\n"
        "India,IND,Inflation,FP.CPI.TOTL.ZG,6.7,5.4\n"
    )
    test_csv.write_text(content, encoding="utf-8")

    cleaned, report = clean_world_bank_file(test_csv, country_codes=["IND"], skiprows=4)
    assert len(cleaned) == 2  # 2022 and 2023
    assert cleaned["indicator_value"].iloc[0] == 6.7
    assert cleaned["indicator_value"].iloc[1] == 5.4
