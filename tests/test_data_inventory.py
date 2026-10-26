"""Unit tests for the StockMind AI data inventory engine."""

from __future__ import annotations

import os
from pathlib import Path

import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.data_inventory import (
    DataInventoryAuditor,
    count_lines_stream,
    format_bytes,
    inspect_csv_file,
    inspect_excel_file,
    inspect_image_folder,
    inspect_parquet_file,
)


def test_format_bytes():
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.00 KB"
    assert format_bytes(1024 * 1024 * 5) == "5.00 MB"
    assert format_bytes(1024 * 1024 * 1024 * 2) == "2.00 GB"


def test_count_lines_stream(tmp_path: Path):
    test_file = tmp_path / "sample.csv"
    test_file.write_text("col1,col2\n1,a\n2,b\n3,c\n", encoding="utf-8")
    line_count = count_lines_stream(test_file)
    assert line_count == 4


def test_inspect_csv_file(tmp_path: Path):
    test_csv = tmp_path / "test_trades.csv"
    data = {
        "Date ": ["2023-01-01", "2023-01-02", "2023-01-02", None],
        "Price": [100.5, 102.0, 102.0, 105.0],
        "Volume": [1000, 2000, 2000, 1500],
    }
    df = pd.DataFrame(data)
    df.to_csv(test_csv, index=False)

    report = inspect_csv_file(test_csv, sample_rows=10, count_full_rows=True)
    assert report["format"] == "csv"
    assert report["file_name"] == "test_trades.csv"
    assert report["num_columns"] == 3
    assert "Date " in report["whitespace_in_column_names"]
    assert report["sample_duplicates"] == 1  # Row 2 and Row 3 are duplicates
    assert report["missing_values_sample"]["Date "] == 1
    assert "Date " in report["date_range"]
    assert report["date_range"]["Date "]["min"].startswith("2023-01-01")


def test_inspect_parquet_file(tmp_path: Path):
    test_parquet = tmp_path / "nse_test.parquet"
    data = {
        "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")],
        "symbol": ["INFY", "TCS", "RELIANCE"],
        "close": [1500.0, 3200.0, 2400.0],
        "volume": [100000, 200000, 300000],
    }
    table = pa.Table.from_pydict(data)
    pq.write_table(table, test_parquet)

    report = inspect_parquet_file(test_parquet)
    assert report["format"] == "parquet"
    assert report["num_rows"] == 3
    assert report["num_columns"] == 4
    assert "date" in report["columns"]
    assert report["data_types"]["symbol"].startswith("string")
    assert report["date_range"]["date"]["min"].startswith("2023-01-01")
    assert report["date_range"]["date"]["max"].startswith("2023-01-03")


def test_inspect_excel_file(tmp_path: Path):
    test_excel = tmp_path / "cpi_test.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "CPI_Data"
    ws.append(["Year", "Month", "Combined_Index"])
    ws.append([2024, "Jan", 185.2])
    wb.save(test_excel)
    wb.close()

    report = inspect_excel_file(test_excel)
    assert report["format"] == "excel"
    assert "CPI_Data" in report["sheets"]
    assert report["sheets"]["CPI_Data"]["header_columns"] == ["Year", "Month", "Combined_Index"]


def test_inspect_image_folder(tmp_path: Path):
    img_dir = tmp_path / "patterns"
    img_dir.mkdir()
    (img_dir / "img1.jpg").write_bytes(b"\xff\xd8\xff dummy jpeg data")
    (img_dir / "img2.jpg").write_bytes(b"\xff\xd8\xff dummy jpeg data 2")
    (img_dir / "mask1.png").write_bytes(b"\x89PNG dummy png")

    report = inspect_image_folder(img_dir)
    assert report["format"] == "image_directory"
    assert report["total_images"] == 3
    assert report["extensions"][".jpg"] == 2
    assert report["extensions"][".png"] == 1


def test_auditor_overlap_and_flags(tmp_path: Path):
    dummy_root = tmp_path / "Dataset"
    dummy_root.mkdir()
    auditor = DataInventoryAuditor(dummy_root)
    overlaps = auditor._detect_overlaps()
    assert len(overlaps) >= 3
    flags = auditor._detect_validation_flags({})
    assert len(flags) >= 4
    flag_ids = [f["flag_id"] for f in flags]
    assert "FLAG-01-NEWS-TIMESTAMPS" in flag_ids
    assert "FLAG-02-WHITESPACE-COLUMNS" in flag_ids
