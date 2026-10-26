#!/usr/bin/env python
"""Execution script to run the StockMind AI Data Inventory and generate reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_inventory import DataInventoryAuditor


def main():
    parser = argparse.ArgumentParser(description="Audit and inventory StockMind AI datasets.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(PROJECT_ROOT / "Dataset"),
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "DATA_INVENTORY.md"),
        help="Path to write the Markdown inventory report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "data_inventory_report.json"),
        help="Path to write the JSON audit report.",
    )
    parser.add_argument(
        "--company-samples",
        type=int,
        default=10,
        help="Number of company CSV files to sample per exchange.",
    )
    parser.add_argument(
        "--minute-samples",
        type=int,
        default=10,
        help="Number of minute ticker files to sample.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    print(f"[*] Starting StockMind AI data audit on: {dataset_path}")
    if not dataset_path.exists():
        print(f"[!] Error: Dataset directory {dataset_path} does not exist.")
        sys.exit(1)

    auditor = DataInventoryAuditor(dataset_path)
    results = auditor.run_full_audit(
        max_company_csv_samples=args.company_samples,
        max_minute_csv_samples=args.minute_samples,
    )

    # Save JSON report
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved structured JSON audit report to: {json_path}")

    # Save Markdown report
    md_content = auditor.generate_markdown_report(results)
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Saved Markdown audit report to: {md_path}")

    # Print summary highlights
    total = results.get("total_summary", {})
    print("\n" + "=" * 60)
    print(" STOCKMIND AI - DATA AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Files Audited : {total.get('total_files', 0):,}")
    print(f"Total Disk Footprint: {total.get('total_formatted_size')} ({total.get('total_bytes', 0):,} bytes)")
    print(f"Parquet Daily Rows  : {results['categories']['market_data']['subcategories']['nse_daily']['aggregate_rows']:,}")
    print(f"Company CSV Files   : {results['categories']['market_data']['subcategories']['comapanies_data']['total_csv_files']}")
    print(f"Minute Data Files   : {results['categories']['market_data']['subcategories']['comapanies_minutes']['total_files']}")
    print(f"Cross-Dataset Flags : {len(results.get('critical_validation_flags', []))} critical items flagged")
    print("=" * 60)


if __name__ == "__main__":
    main()
