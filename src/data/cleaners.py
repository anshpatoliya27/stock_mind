"""Data cleaning and sanitization engines for StockMind AI.

Provides reusable, memory-conscious cleaners for:
1. Corporate Events: Bulk Deals CSV files (handling Indian numbering formats,
   trailing whitespace headers, price comma formatting, and anomaly detection).
2. Macroeconomic: World Bank WDI CSV files (handling metadata header rows,
   indicator reshaping, country filtering, and numeric conversion).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def sanitize_column_names(df: pd.DataFrame) -> Dict[str, str]:
    """Generate mapping of sanitized, stripped, lowercase snake_case column names."""
    rename_map = {}
    for col in df.columns:
        clean_name = str(col).strip()
        # Common domain renames
        clean_name = clean_name.replace(" / ", "_").replace(" /", "_").replace("/ ", "_").replace("/", "_")
        clean_name = clean_name.replace(". ", "_").replace(".", "_").replace(" ", "_")
        clean_name = clean_name.lower()
        while "__" in clean_name:
            clean_name = clean_name.replace("__", "_")
        rename_map[col] = clean_name
    return rename_map


def clean_bulk_deals_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate a Bulk Deals dataframe in-memory.
    
    Returns:
        (cleaned_df, validation_report)
    """
    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "missing_values_before": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "whitespace_headers_detected": [c for c in df.columns if c != str(c).strip()],
        "invalid_numeric_prices": 0,
        "invalid_numeric_quantities": 0,
        "suspicious_prices": 0,
        "invalid_dates": 0,
        "duplicate_rows": 0,
        "buy_sell_distribution": {},
        "date_bounds": {},
    }

    # 1. Strip column names and rename to canonical schema
    clean_map = sanitize_column_names(df)
    df_clean = df.rename(columns=clean_map).copy()

    # Canonical expected names: date, symbol, security_name, client_name, buy_sell, quantity_traded, trade_price_wght_avg_price, remarks
    price_col = None
    qty_col = None
    date_col = None
    action_col = None

    for c in df_clean.columns:
        if "trade_price" in c or "price" in c:
            price_col = c
        elif "quantity" in c or "qty" in c:
            qty_col = c
        elif "date" in c:
            date_col = c
        elif "buy" in c or "sell" in c:
            action_col = c

    # 2. Clean Date
    if date_col and date_col in df_clean.columns:
        raw_dates = df_clean[date_col].astype(str).str.strip()
        parsed_dates = pd.to_datetime(raw_dates, format="%d-%b-%Y", errors="coerce")
        # Fallback for alternative date formats
        null_mask = parsed_dates.isna() & (raw_dates != "nan") & (raw_dates != "")
        if null_mask.any():
            fallback_parsed = pd.to_datetime(raw_dates[null_mask], errors="coerce")
            parsed_dates.loc[null_mask] = fallback_parsed

        report["invalid_dates"] = int(parsed_dates.isna().sum())
        df_clean[date_col] = parsed_dates
        if not parsed_dates.dropna().empty:
            report["date_bounds"] = {
                "min": str(parsed_dates.min()),
                "max": str(parsed_dates.max()),
            }

    # 3. Clean Quantity (remove commas from Indian format e.g. 43,85,856)
    if qty_col and qty_col in df_clean.columns:
        raw_qty = df_clean[qty_col].astype(str).str.replace(",", "").str.strip()
        parsed_qty = pd.to_numeric(raw_qty, errors="coerce")
        report["invalid_numeric_quantities"] = int(parsed_qty.isna().sum())
        # Replace non-positive or NaN with 0 or NaN
        df_clean[qty_col] = parsed_qty

    # 4. Clean Price (remove commas from format e.g. 1,179.70)
    if price_col and price_col in df_clean.columns:
        raw_price = df_clean[price_col].astype(str).str.replace(",", "").str.strip()
        parsed_price = pd.to_numeric(raw_price, errors="coerce")
        report["invalid_numeric_prices"] = int(parsed_price.isna().sum())

        # Check for suspicious price records: <= 0 or extreme outliers
        suspicious_mask = (parsed_price <= 0) | (parsed_price > 500000)
        report["suspicious_prices"] = int(suspicious_mask.sum())
        df_clean[price_col] = parsed_price

    # 5. Clean Action (BUY / SELL)
    if action_col and action_col in df_clean.columns:
        df_clean[action_col] = df_clean[action_col].astype(str).str.strip().str.upper()
        report["buy_sell_distribution"] = df_clean[action_col].value_counts().to_dict()

    # 6. Clean Remarks
    for c in df_clean.columns:
        if "remark" in c:
            df_clean[c] = df_clean[c].replace({"-": None, "NA": None, "": None, "nan": None})

    # 7. Check duplicates
    report["duplicate_rows"] = int(df_clean.duplicated().sum())
    report["final_rows"] = len(df_clean)
    report["missing_values_after"] = {str(c): int(df_clean[c].isna().sum()) for c in df_clean.columns}

    return df_clean, report


def clean_bulk_deals_file(
    filepath: Union[str, Path],
    output_filepath: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean a Bulk Deals CSV from disk, generate validation report, and optionally save."""
    filepath = Path(filepath)
    df = pd.read_csv(filepath, encoding="utf-8")
    cleaned_df, report = clean_bulk_deals_df(df)
    report["file_source"] = str(filepath)

    if output_filepath:
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        report["saved_output_path"] = str(output_path)

    return cleaned_df, report


def clean_world_bank_df(
    df: pd.DataFrame,
    country_codes: Optional[List[str]] = None,
    melt_to_long: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and reshape a World Bank WDI dataframe.
    
    Returns:
        (cleaned_df, validation_report)
    """
    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "initial_columns": len(df.columns),
        "melted_to_long": melt_to_long,
        "country_filter_applied": country_codes,
    }

    # Drop trailing empty Unnamed columns
    drop_cols = [c for c in df.columns if "unnamed" in str(c).lower()]
    df_clean = df.drop(columns=drop_cols).copy()

    # Sanitize metadata headers
    rename_dict = {}
    year_cols = []
    for c in df_clean.columns:
        col_str = str(c).strip()
        if col_str.isdigit():
            year_cols.append(col_str)
            rename_dict[c] = col_str
        elif "Country Name" in col_str:
            rename_dict[c] = "country_name"
        elif "Country Code" in col_str:
            rename_dict[c] = "country_code"
        elif "Indicator Name" in col_str:
            rename_dict[c] = "indicator_name"
        elif "Indicator Code" in col_str:
            rename_dict[c] = "indicator_code"
        else:
            rename_dict[c] = col_str.lower().replace(" ", "_")

    df_clean = df_clean.rename(columns=rename_dict)
    report["identified_year_columns_count"] = len(year_cols)

    # Optional country filter
    if country_codes:
        country_codes_upper = [c.upper() for c in country_codes]
        df_clean = df_clean[df_clean["country_code"].str.upper().isin(country_codes_upper)].copy()
        report["filtered_rows"] = len(df_clean)

    if melt_to_long:
        id_vars = [c for c in ["country_name", "country_code", "indicator_name", "indicator_code"] if c in df_clean.columns]
        val_vars = [y for y in year_cols if y in df_clean.columns]
        df_long = pd.melt(
            df_clean,
            id_vars=id_vars,
            value_vars=val_vars,
            var_name="year",
            value_name="indicator_value",
        )
        df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype(int)
        df_long["indicator_value"] = pd.to_numeric(df_long["indicator_value"], errors="coerce")

        report["final_rows"] = len(df_long)
        report["missing_values"] = int(df_long["indicator_value"].isna().sum())
        report["non_null_records"] = int(df_long["indicator_value"].notna().sum())
        if df_long["indicator_value"].notna().any():
            report["value_summary"] = {
                "min": float(df_long["indicator_value"].min()),
                "max": float(df_long["indicator_value"].max()),
                "mean": float(df_long["indicator_value"].mean()),
            }
        return df_long, report
    else:
        report["final_rows"] = len(df_clean)
        return df_clean, report


def clean_world_bank_file(
    filepath: Union[str, Path],
    output_filepath: Optional[Union[str, Path]] = None,
    country_codes: Optional[List[str]] = None,
    melt_to_long: bool = True,
    skiprows: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean a World Bank CSV file from disk with safe skiprows handling."""
    filepath = Path(filepath)
    df = pd.read_csv(filepath, skiprows=skiprows, encoding="utf-8")
    cleaned_df, report = clean_world_bank_df(df, country_codes=country_codes, melt_to_long=melt_to_long)
    report["file_source"] = str(filepath)
    report["skiprows_applied"] = skiprows

    if output_filepath:
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        report["saved_output_path"] = str(output_path)

    return cleaned_df, report
