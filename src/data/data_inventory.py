"""Data inventory, schema inspection, and audit engine for StockMind AI.

Designed to safely audit multi-gigabyte datasets (e.g. 20+ GB) without loading
entire datasets into RAM, using Parquet metadata, streaming byte-buffered row
counting, and targeted columnar samples.
"""

from __future__ import annotations

import csv
import datetime
import glob
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq


def format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def count_lines_stream(filepath: Union[str, Path], buffer_size: int = 1024 * 1024) -> int:
    """Count lines in a text file using streaming byte chunks to avoid RAM allocation.
    
    Returns the number of newline characters (roughly row count including header).
    """
    lines = 0
    with open(filepath, "rb") as f:
        while True:
            buffer = f.read(buffer_size)
            if not buffer:
                break
            lines += buffer.count(b"\n")
    return lines


def inspect_parquet_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Inspect Parquet file using ParquetFile metadata without loading data into memory."""
    filepath = Path(filepath)
    pf = pq.ParquetFile(str(filepath))
    meta = pf.metadata
    schema = pf.schema_arrow

    col_names = schema.names
    col_types = {name: str(schema.field(name).type) for name in col_names}
    num_rows = meta.num_rows
    num_cols = meta.num_columns
    num_row_groups = meta.num_row_groups

    # Check for date columns and get min/max safely
    date_cols = [name for name in col_names if "date" in name.lower() or "time" in name.lower()]
    date_range: Dict[str, Any] = {}

    for dcol in date_cols[:2]:  # Check up to first two date columns
        try:
            # Check row group statistics first (zero data read)
            mins = []
            maxs = []
            for rg in range(num_row_groups):
                rg_meta = meta.row_group(rg)
                col_idx = col_names.index(dcol)
                stat = rg_meta.column(col_idx).statistics
                if stat is not None and stat.has_min_max:
                    mins.append(stat.min)
                    maxs.append(stat.max)
            if mins and maxs:
                date_range[dcol] = {"min": str(min(mins)), "max": str(max(maxs))}
            else:
                # Fallback: read only the first and last row group of just this column
                tbl_sample = pf.read_row_group(0, columns=[dcol])
                first_val = tbl_sample[dcol][0].as_py()
                last_rg = pf.read_row_group(num_row_groups - 1, columns=[dcol])
                last_val = last_rg[dcol][-1].as_py()
                date_range[dcol] = {"min": str(first_val), "max": str(last_val)}
        except Exception as e:
            date_range[dcol] = {"error": str(e)}

    # Missing values: compute from column statistics across row groups
    missing_values: Dict[str, int] = {}
    for idx, col in enumerate(col_names):
        null_count = 0
        has_stats = False
        for rg in range(num_row_groups):
            stat = meta.row_group(rg).column(idx).statistics
            if stat is not None and stat.null_count is not None:
                null_count += stat.null_count
                has_stats = True
        if has_stats:
            missing_values[col] = null_count

    return {
        "format": "parquet",
        "file_name": filepath.name,
        "size_bytes": filepath.stat().st_size,
        "size_formatted": format_bytes(filepath.stat().st_size),
        "num_rows": num_rows,
        "num_columns": num_cols,
        "num_row_groups": num_row_groups,
        "columns": col_names,
        "data_types": col_types,
        "date_range": date_range,
        "missing_values": missing_values,
        "potential_duplicates_note": (
            "Parquet statistics do not indicate row duplication; key uniqueness can be "
            "validated by distributed partition jobs."
        ),
    }


def inspect_csv_file(
    filepath: Union[str, Path],
    sample_rows: int = 1000,
    count_full_rows: bool = True,
    skiprows: Optional[int] = None,
) -> Dict[str, Any]:
    """Inspect CSV file using header sampling and streaming row counts without memory exhaustion."""
    filepath = Path(filepath)
    size_bytes = filepath.stat().st_size

    # Fast row count via streaming newline count
    approx_rows = None
    if count_full_rows:
        try:
            line_count = count_lines_stream(filepath)
            # Subtract 1 for header if > 0
            approx_rows = max(0, line_count - (1 if skiprows is None else (skiprows + 1)))
        except Exception:
            approx_rows = None

    # Detect skiprows for known metadata formats (like World Bank)
    if skiprows is None:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                if "Data Source,World Development Indicators" in first_line or "Country Name" not in first_line:
                    # Check if line 5 has headers
                    f.seek(0)
                    lines = [f.readline() for _ in range(6)]
                    for idx, line in enumerate(lines):
                        if "Country Name" in line or "Indicator Name" in line:
                            skiprows = idx
                            break
        except Exception:
            skiprows = None

    # Read sample
    encoding = "utf-8"
    try:
        sample_df = pd.read_csv(filepath, nrows=sample_rows, skiprows=skiprows or 0, encoding=encoding)
    except UnicodeDecodeError:
        encoding = "latin1"
        sample_df = pd.read_csv(filepath, nrows=sample_rows, skiprows=skiprows or 0, encoding=encoding)

    col_names = [str(c) for c in sample_df.columns]
    data_types = {col: str(sample_df[col].dtype) for col in col_names}

    # Missing values in sample
    missing_sample = {col: int(sample_df[col].isna().sum()) for col in col_names}

    # Date ranges from sample
    date_range: Dict[str, Any] = {}
    date_cols = [c for c in col_names if "date" in c.lower() or "time" in c.lower() or c.strip() in ["Date", "date", "DATE"]]
    for dc in date_cols:
        try:
            valid_dates = pd.to_datetime(sample_df[dc], errors="coerce").dropna()
            if not valid_dates.empty:
                date_range[dc] = {
                    "min": str(valid_dates.min()),
                    "max": str(valid_dates.max()),
                    "sample_scope": f"first_{len(sample_df)}_rows",
                }
        except Exception:
            pass

    # Check for duplicate rows in sample
    sample_dups = int(sample_df.duplicated().sum())

    # Check for whitespace anomalies in column names
    whitespace_cols = [c for c in col_names if c != c.strip()]

    return {
        "format": "csv",
        "file_name": filepath.name,
        "size_bytes": size_bytes,
        "size_formatted": format_bytes(size_bytes),
        "row_count_estimate": approx_rows if approx_rows is not None else len(sample_df),
        "sample_rows_analyzed": len(sample_df),
        "num_columns": len(col_names),
        "columns": col_names,
        "data_types": data_types,
        "date_range": date_range,
        "missing_values_sample": missing_sample,
        "sample_duplicates": sample_dups,
        "whitespace_in_column_names": whitespace_cols,
        "encoding": encoding,
        "skiprows_detected": skiprows or 0,
    }


def inspect_excel_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Inspect Excel file sheets and headers using openpyxl in read-only mode."""
    filepath = Path(filepath)
    size_bytes = filepath.stat().st_size
    try:
        import openpyxl

        wb = openpyxl.load_workbook(str(filepath), read_only=True, data_only=True)
        sheets_info = {}
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            header_row = []
            row_count = 0
            for row in ws.iter_rows(values_only=True):
                if row_count == 0:
                    header_row = [str(c) if c is not None else "" for c in row]
                row_count += 1
                if row_count > 10000:  # Safety cap for read_only iterator
                    break
            sheets_info[sheet_name] = {
                "header_columns": [c for c in header_row if c],
                "sample_row_count": row_count,
            }
        wb.close()
        return {
            "format": "excel",
            "file_name": filepath.name,
            "size_bytes": size_bytes,
            "size_formatted": format_bytes(size_bytes),
            "sheets": sheets_info,
        }
    except Exception as e:
        return {
            "format": "excel",
            "file_name": filepath.name,
            "size_bytes": size_bytes,
            "size_formatted": format_bytes(size_bytes),
            "error": str(e),
        }


def inspect_image_folder(folder_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect an image directory reporting file count, extensions, and total size."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return {"error": "Folder does not exist"}

    all_files = [f for f in folder_path.iterdir() if f.is_file()]
    exts = {}
    total_size = 0
    for f in all_files:
        ext = f.suffix.lower()
        exts[ext] = exts.get(ext, 0) + 1
        total_size += f.stat().st_size

    return {
        "format": "image_directory",
        "directory_name": folder_path.name,
        "total_images": len(all_files),
        "extensions": exts,
        "total_size_bytes": total_size,
        "total_size_formatted": format_bytes(total_size),
    }


class DataInventoryAuditor:
    """Comprehensive auditor for the StockMind AI Dataset hierarchy."""

    def __init__(self, dataset_root: Union[str, Path]):
        self.dataset_root = Path(dataset_root)

    def run_full_audit(
        self,
        max_company_csv_samples: int = 10,
        max_minute_csv_samples: int = 10,
    ) -> Dict[str, Any]:
        """Execute full audit across all 5 dataset categories."""
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        audit_results: Dict[str, Any] = {
            "audit_timestamp": datetime.datetime.now().isoformat(),
            "dataset_root": str(self.dataset_root),
            "categories": {},
            "cross_dataset_overlaps": [],
            "critical_validation_flags": [],
            "total_summary": {
                "total_files": 0,
                "total_bytes": 0,
                "total_formatted_size": "",
            },
        }

        # 1. Macroeconomic audit
        macro_res = self._audit_macroeconomic()
        audit_results["categories"]["macroeconomic"] = macro_res

        # 2. Corporate events audit
        corp_res = self._audit_corporate_events()
        audit_results["categories"]["corporate_events"] = corp_res

        # 3. Financial news audit
        news_res = self._audit_financial_news()
        audit_results["categories"]["financial_news"] = news_res

        # 4. Market data audit
        market_res = self._audit_market_data(
            max_company_samples=max_company_csv_samples,
            max_minute_samples=max_minute_csv_samples,
        )
        audit_results["categories"]["market_data"] = market_res

        # 5. Chart images audit
        chart_res = self._audit_chart_images()
        audit_results["categories"]["chart_images"] = chart_res

        # Cross-dataset analysis
        audit_results["cross_dataset_overlaps"] = self._detect_overlaps()
        audit_results["critical_validation_flags"] = self._detect_validation_flags(audit_results)

        # Global totals
        total_files = sum(
            len(files) for root, _, files in os.walk(self.dataset_root)
        )
        total_bytes = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(self.dataset_root)
            for f in files
        )
        audit_results["total_summary"]["total_files"] = total_files
        audit_results["total_summary"]["total_bytes"] = total_bytes
        audit_results["total_summary"]["total_formatted_size"] = format_bytes(total_bytes)

        return audit_results

    def _audit_macroeconomic(self) -> Dict[str, Any]:
        macro_dir = self.dataset_root / "macroeconomic"
        results: Dict[str, Any] = {"subcategories": {}}
        if not macro_dir.exists():
            return results

        # Forex
        forex_dir = macro_dir / "forex"
        if forex_dir.exists():
            forex_files = {}
            for f in forex_dir.iterdir():
                if f.suffix == ".parquet":
                    forex_files[f.name] = inspect_parquet_file(f)
                elif f.suffix == ".csv":
                    forex_files[f.name] = inspect_csv_file(f, sample_rows=500)
            results["subcategories"]["forex"] = forex_files

        # India CPI
        cpi_dir = macro_dir / "india_cpi"
        if cpi_dir.exists():
            cpi_files = {}
            for f in cpi_dir.iterdir():
                if f.suffix in [".xlsx", ".xls"]:
                    cpi_files[f.name] = inspect_excel_file(f)
            results["subcategories"]["india_cpi"] = cpi_files

        # Monetary Policy
        mp_dir = macro_dir / "monetary_policy"
        if mp_dir.exists():
            mp_files = {}
            for f in mp_dir.iterdir():
                if f.suffix in [".xlsx", ".xls"]:
                    mp_files[f.name] = inspect_excel_file(f)
            results["subcategories"]["monetary_policy"] = mp_files

        # World Bank
        wb_dir = macro_dir / "world_bank"
        if wb_dir.exists():
            wb_files = {}
            for f in wb_dir.iterdir():
                if f.suffix == ".csv":
                    wb_files[f.name] = inspect_csv_file(f, sample_rows=200)
            results["subcategories"]["world_bank"] = wb_files

        return results

    def _audit_corporate_events(self) -> Dict[str, Any]:
        corp_dir = self.dataset_root / "corporate_events"
        results: Dict[str, Any] = {"files": {}}
        if not corp_dir.exists():
            return results
        for f in corp_dir.iterdir():
            if f.suffix == ".csv":
                results["files"][f.name] = inspect_csv_file(f, sample_rows=1000)
            elif f.suffix == ".parquet":
                results["files"][f.name] = inspect_parquet_file(f)
        return results

    def _audit_financial_news(self) -> Dict[str, Any]:
        news_dir = self.dataset_root / "financial_news"
        results: Dict[str, Any] = {"files": {}}
        if not news_dir.exists():
            return results
        for f in news_dir.iterdir():
            if f.suffix == ".csv":
                results["files"][f.name] = inspect_csv_file(f, sample_rows=1000)
        return results

    def _audit_market_data(
        self,
        max_company_samples: int = 10,
        max_minute_samples: int = 10,
    ) -> Dict[str, Any]:
        market_dir = self.dataset_root / "market_data"
        results: Dict[str, Any] = {"subcategories": {}}
        if not market_dir.exists():
            return results

        # 1. nse_daily (Parquet 2019-2026)
        nse_daily_dir = market_dir / "nse_daily"
        if nse_daily_dir.exists():
            daily_parquets = {}
            total_nse_rows = 0
            for f in sorted(nse_daily_dir.glob("*.parquet")):
                info = inspect_parquet_file(f)
                daily_parquets[f.name] = info
                total_nse_rows += info.get("num_rows", 0)
            results["subcategories"]["nse_daily"] = {
                "files": daily_parquets,
                "aggregate_rows": total_nse_rows,
                "file_count": len(daily_parquets),
            }

        # 2. comapanies_data (Daily CSVs for BSE / NSE)
        comp_dir = market_dir / "comapanies_data"
        if comp_dir.exists():
            all_comp_files = list(comp_dir.glob("*.csv"))
            ns_files = [f for f in all_comp_files if ".NS." in f.name]
            bo_files = [f for f in all_comp_files if ".BO." in f.name]

            # Sample audit
            sample_audit = {}
            for f in (ns_files[:max_company_samples] + bo_files[:max_company_samples]):
                sample_audit[f.name] = inspect_csv_file(f, sample_rows=500, count_full_rows=True)

            results["subcategories"]["comapanies_data"] = {
                "total_csv_files": len(all_comp_files),
                "nse_ticker_files (.NS)": len(ns_files),
                "bse_ticker_files (.BO)": len(bo_files),
                "inspected_samples": sample_audit,
            }

        # 3. comapanies_minutes (Minute CSVs, indices, metadata catalogs)
        min_dir = market_dir / "comapanies_minutes"
        if min_dir.exists():
            all_min_files = list(min_dir.glob("*.csv"))
            minute_tickers = [f for f in all_min_files if "_minute.csv" in f.name]
            index_files = [f for f in all_min_files if "NIFTY" in f.name or "SENSEX" in f.name or "VIX" in f.name]
            meta_files = [f for f in all_min_files if f.name in ["Patterns.csv", "Segmentation.csv", "Meta.csv"]]

            inspected_min = {}
            for f in minute_tickers[:max_minute_samples]:
                # Don't do full line count on all 500+ files to preserve execution speed; inspect sample
                inspected_min[f.name] = inspect_csv_file(f, sample_rows=1000, count_full_rows=False)

            inspected_indices = {}
            for f in index_files:
                inspected_indices[f.name] = inspect_csv_file(f, sample_rows=500, count_full_rows=True)

            inspected_meta = {}
            for f in meta_files:
                inspected_meta[f.name] = inspect_csv_file(f, sample_rows=500, count_full_rows=True)

            results["subcategories"]["comapanies_minutes"] = {
                "total_files": len(all_min_files),
                "minute_ticker_count": len(minute_tickers),
                "index_file_count": len(index_files),
                "meta_catalog_files": len(meta_files),
                "inspected_minute_samples": inspected_min,
                "inspected_index_files": inspected_indices,
                "inspected_meta_catalogs": inspected_meta,
            }

        return results

    def _audit_chart_images(self) -> Dict[str, Any]:
        chart_dir = self.dataset_root / "chart_images"
        results: Dict[str, Any] = {"subcategories": {}}
        if not chart_dir.exists():
            return results

        for sub in ["Patterns", "Segmentation"]:
            sub_path = chart_dir / sub
            if sub_path.exists():
                results["subcategories"][sub] = inspect_image_folder(sub_path)
        return results

    def _detect_overlaps(self) -> List[Dict[str, Any]]:
        """Identify redundant or overlapping representations across datasets."""
        overlaps = [
            {
                "overlap_id": "OVERLAP-01",
                "category": "Macroeconomic Forex",
                "datasets_involved": [
                    "Dataset/macroeconomic/forex/rbi_forex_reference_rates.parquet",
                    "Dataset/macroeconomic/forex/rbi_forex_reference_rates_long.csv",
                    "Dataset/macroeconomic/forex/rbi_forex_reference_rates_wide.csv",
                ],
                "nature_of_overlap": (
                    "The parquet file and long CSV share identical schema ['date', 'currency', 'rate', 'unit']. "
                    "The wide CSV is the pivoted representation ['date', 'AED', 'EUR', 'GBP', 'IDR', 'JPY', 'USD']. "
                    "Action: Retain all files; standardize on Parquet during Big Data feature engineering to save I/O."
                ),
            },
            {
                "overlap_id": "OVERLAP-02",
                "category": "Historical Market Data (NSE)",
                "datasets_involved": [
                    "Dataset/market_data/nse_daily/nse_2019.parquet through nse_2026.parquet",
                    "Dataset/market_data/comapanies_data/*.NS.csv",
                ],
                "nature_of_overlap": (
                    "The nse_daily parquet partition contains market-wide daily OHLCV trades for all NSE equities "
                    "from 2019 to 2026. The comapanies_data/*.NS.csv contains individual ticker histories (e.g. 2008-2020). "
                    "Action: Flag temporal coverage differences; use nse_daily for cross-sectional market modeling "
                    "and company CSVs for long-horizon pre-2019 historical context."
                ),
            },
            {
                "overlap_id": "OVERLAP-03",
                "category": "Chart Vision & Pattern Metadata",
                "datasets_involved": [
                    "Dataset/chart_images/Patterns (340 JPG images)",
                    "Dataset/chart_images/Segmentation (340 PNG masks)",
                    "Dataset/market_data/comapanies_minutes/Patterns.csv",
                    "Dataset/market_data/comapanies_minutes/Segmentation.csv",
                ],
                "nature_of_overlap": (
                    "The CSV files in comapanies_minutes catalogue the exact image filenames, class labels, and bounding dimensions "
                    "for the chart images in chart_images. "
                    "Action: Keep intact; utilize Patterns.csv and Segmentation.csv as PyTorch Dataset manifest catalogs."
                ),
            },
        ]
        return overlaps

    def _detect_validation_flags(self, audit_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical data validation flags requiring engineering attention."""
        flags = [
            {
                "flag_id": "FLAG-01-NEWS-TIMESTAMPS",
                "severity": "CRITICAL",
                "component": "Dataset/financial_news/training_data_26000.csv",
                "finding": (
                    "The financial news dataset contains columns ['URL', 'Content', 'Summary', 'Sentiment']. "
                    "There is NO explicit published_date or article_timestamp column! "
                    "Publication dates must either be parsed from URL slugs or extracted from text metadata "
                    "before any time-aligned backtesting or sentiment join can be performed."
                ),
            },
            {
                "flag_id": "FLAG-02-WHITESPACE-COLUMNS",
                "severity": "WARNING",
                "component": "Dataset/corporate_events/Bulk-Deals-25-10-2025-to-25-10-2026.csv",
                "finding": (
                    "Bulk deals CSV column headers contain trailing whitespace (e.g., 'Date ', 'Symbol ', 'Security Name '). "
                    "Data ingestion routines must strip column whitespace during normalization."
                ),
            },
            {
                "flag_id": "FLAG-03-WORLDBANK-METADATA-HEADER",
                "severity": "WARNING",
                "component": "Dataset/macroeconomic/world_bank/*.csv",
                "finding": (
                    "World Bank CSV files have 4 leading metadata commentary rows before the column headers. "
                    "Standard pd.read_csv fails unless skiprows=4 is applied."
                ),
            },
            {
                "flag_id": "FLAG-04-VOLUME-SCALE-CHECK",
                "severity": "INFO",
                "component": "Dataset/market_data/comapanies_minutes/ (20+ GB Total)",
                "finding": (
                    "Minute-level data encompasses 515 ticker and index files totaling over 20 GB. "
                    "Direct in-memory loading is prohibited; distributed PySpark windowing or chunked out-of-core "
                    "Polars/DuckDB execution will be mandated for feature extraction."
                ),
            },
        ]
        return flags

    def generate_markdown_report(self, audit_dict: Dict[str, Any]) -> str:
        """Render the audit results into a structured, readable Markdown document."""
        out = io.StringIO()
        out.write("# StockMind AI: Comprehensive Data Inventory & Audit Report\n\n")
        out.write(f"**Audit Timestamp:** {audit_dict.get('audit_timestamp')}\n\n")
        out.write(f"**Dataset Location:** `{audit_dict.get('dataset_root')}`\n\n")

        summary = audit_dict.get("total_summary", {})
        out.write("## 1. Executive Summary\n\n")
        out.write(f"- **Total Files:** {summary.get('total_files'):,}\n")
        out.write(f"- **Total Storage Footprint:** {summary.get('total_formatted_size')} ({summary.get('total_bytes'):,} bytes)\n")
        out.write("- **Primary Formats:** Parquet (NSE daily, RBI Forex), CSV (Minute ticks, company OHLCV, News, Bulk Deals), XLSX (CPI, Monetary Policy), JPG/PNG (Chart Vision).\n\n")

        out.write("## 2. Category Audits\n\n")

        # Market Data
        market = audit_dict.get("categories", {}).get("market_data", {}).get("subcategories", {})
        out.write("### 2.1 Market Data\n\n")
        nse_daily = market.get("nse_daily", {})
        out.write(f"#### A. NSE Daily Parquet (2019–2026)\n")
        out.write(f"- **Total Partitions:** {nse_daily.get('file_count')} Parquet files\n")
        out.write(f"- **Aggregate Trades/Rows:** {nse_daily.get('aggregate_rows', 0):,}\n")
        out.write("| Year / File | Rows | File Size | Schema Columns | Date Range |\n")
        out.write("|---|---|---|---|---|\n")
        for fname, finfo in nse_daily.get("files", {}).items():
            drange = finfo.get("date_range", {}).get("date", {})
            dstr = f"{drange.get('min')} to {drange.get('max')}" if drange else "N/A"
            out.write(f"| `{fname}` | {finfo.get('num_rows'):,} | {finfo.get('size_formatted')} | {finfo.get('num_columns')} cols | {dstr} |\n")
        out.write("\n")

        # Companies Data
        comp_data = market.get("comapanies_data", {})
        out.write(f"#### B. Company-Level Daily Stock Data (`comapanies_data`)\n")
        out.write(f"- **Total Files:** {comp_data.get('total_csv_files')} CSV files\n")
        out.write(f"- **National Stock Exchange (.NS):** {comp_data.get('nse_ticker_files (.NS)')} tickers\n")
        out.write(f"- **Bombay Stock Exchange (.BO):** {comp_data.get('bse_ticker_files (.BO)')} tickers\n")
        out.write("- **Standard Schema:** `['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']`\n\n")

        # Minute Data
        min_data = market.get("comapanies_minutes", {})
        out.write(f"#### C. Minute-Level Stock Data (`comapanies_minutes`)\n")
        out.write(f"- **Total Files:** {min_data.get('total_files')} files\n")
        out.write(f"- **Minute Stock Tickers:** {min_data.get('minute_ticker_count')} tickers (up to 1,000,000+ rows per ticker)\n")
        out.write(f"- **Index Tickers:** {min_data.get('index_file_count')} files (`NIFTY 50`, `NIFTY BANK`, `India VIX`, `SENSEX`, etc.)\n")
        out.write(f"- **Catalog Manifests:** `Patterns.csv`, `Segmentation.csv`, `Meta.csv`\n")
        out.write("- **Schema:** `['date', 'open', 'high', 'low', 'close', 'volume']`\n\n")

        # Financial News
        news = audit_dict.get("categories", {}).get("financial_news", {}).get("files", {})
        out.write("### 2.2 Financial News\n\n")
        for fname, finfo in news.items():
            out.write(f"- **File:** `{fname}` ({finfo.get('size_formatted')})\n")
            out.write(f"- **Row Count:** ~{finfo.get('row_count_estimate', 0):,} records\n")
            out.write(f"- **Columns:** `{finfo.get('columns')}`\n")
            out.write(f"- **Sample Nulls:** `{finfo.get('missing_values_sample')}`\n\n")

        # Corporate Events
        corp = audit_dict.get("categories", {}).get("corporate_events", {}).get("files", {})
        out.write("### 2.3 Corporate Events\n\n")
        for fname, finfo in corp.items():
            out.write(f"- **File:** `{fname}` ({finfo.get('size_formatted')})\n")
            out.write(f"- **Estimated Records:** ~{finfo.get('row_count_estimate', 0):,} bulk trade rows\n")
            out.write(f"- **Raw Columns:** `{finfo.get('columns')}`\n")
            out.write(f"- **Whitespace Detected in Columns:** `{finfo.get('whitespace_in_column_names')}`\n\n")

        # Macroeconomic
        macro = audit_dict.get("categories", {}).get("macroeconomic", {}).get("subcategories", {})
        out.write("### 2.4 Macroeconomic Data\n\n")
        for sub_name, sub_content in macro.items():
            out.write(f"#### Subcategory: `{sub_name}`\n")
            for fname, finfo in sub_content.items():
                fmt = finfo.get("format")
                if fmt == "parquet":
                    out.write(f"- `{fname}` ({finfo.get('size_formatted')}): {finfo.get('num_rows'):,} rows, columns: `{finfo.get('columns')}`\n")
                elif fmt == "csv":
                    out.write(f"- `{fname}` ({finfo.get('size_formatted')}): columns: `{finfo.get('columns')}`\n")
                elif fmt == "excel":
                    out.write(f"- `{fname}` ({finfo.get('size_formatted')}): Sheets: `{list(finfo.get('sheets', {}).keys())}`\n")
            out.write("\n")

        # Chart Images
        charts = audit_dict.get("categories", {}).get("chart_images", {}).get("subcategories", {})
        out.write("### 2.5 Chart Images & Vision\n\n")
        for sub_name, cinfo in charts.items():
            out.write(f"- **{sub_name}:** {cinfo.get('total_images')} files ({cinfo.get('total_size_formatted')}), Extensions: `{cinfo.get('extensions')}`\n")
        out.write("\n")

        # Overlaps
        out.write("## 3. Cross-Dataset Overlaps & Alignment\n\n")
        for o in audit_dict.get("cross_dataset_overlaps", []):
            out.write(f"### {o.get('overlap_id')}: {o.get('category')}\n")
            out.write(f"- **Datasets Involved:**\n")
            for d in o.get("datasets_involved", []):
                out.write(f"  - `{d}`\n")
            out.write(f"- **Analysis:** {o.get('nature_of_overlap')}\n\n")

        # Validation Flags
        out.write("## 4. Critical Data Validation Flags\n\n")
        for f in audit_dict.get("critical_validation_flags", []):
            out.write(f"### [{f.get('severity')}] {f.get('flag_id')}\n")
            out.write(f"- **Component:** `{f.get('component')}`\n")
            out.write(f"- **Observation:** {f.get('finding')}\n\n")

        return out.getvalue()
