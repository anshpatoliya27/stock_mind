"""Validation Reporting Framework for StockMind AI.

Generates rigorous statistical quality audits:
- Missing value counts & percentages
- Duplicate key collisions
- Invalid numeric metrics (NaNs, infs, negative quantities)
- Invalid dates and out-of-epoch timestamps
- Suspicious price records (High < Low, Close out of bounds, price <= 0, abnormal returns)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class DataValidationReporter:
    """Validator for tabular financial data series."""

    @staticmethod
    def audit_dataframe(
        df: pd.DataFrame,
        dataset_name: str,
        date_col: Optional[str] = None,
        price_cols: Optional[List[str]] = None,
        volume_col: Optional[str] = None,
        key_cols: Optional[List[str]] = None,
        max_acceptable_price_jump: float = 2.0,  # 200% single bar move
    ) -> Dict[str, Any]:
        """Perform full validation audit on a dataframe.
        
        Returns:
            Dictionary containing categorized audit findings.
        """
        total_rows = len(df)
        report: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values": {},
            "duplicate_records": {
                "total_duplicates": int(df.duplicated().sum()),
                "key_collisions": 0,
            },
            "invalid_dates": {
                "unparseable_dates": 0,
                "out_of_epoch_dates": 0,
            },
            "invalid_numerics": {
                "nan_counts": {},
                "infinite_counts": {},
                "negative_counts": {},
            },
            "suspicious_prices": {
                "high_less_than_low": 0,
                "close_out_of_bounds": 0,
                "open_out_of_bounds": 0,
                "non_positive_prices": 0,
                "extreme_price_jumps": 0,
            },
            "validation_grade": "PASSED",
            "critical_issues": [],
            "warnings": [],
        }

        # 1. Missing Values
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                report["missing_values"][col] = {
                    "count": null_count,
                    "percentage": round((null_count / total_rows * 100) if total_rows else 0, 2),
                }

        # 2. Key Collisions
        if key_cols:
            valid_keys = [k for k in key_cols if k in df.columns]
            if valid_keys:
                key_dups = int(df.duplicated(subset=valid_keys).sum())
                report["duplicate_records"]["key_collisions"] = key_dups
                if key_dups > 0:
                    report["warnings"].append(f"Found {key_dups} key collisions on subset {valid_keys}")

        # 3. Date Validation
        if date_col and date_col in df.columns:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            unparseable = int(parsed.isna().sum()) - int(df[date_col].isna().sum())
            report["invalid_dates"]["unparseable_dates"] = max(0, unparseable)

            valid_dates = parsed.dropna()
            if not valid_dates.empty:
                # Stock market epoch: 1990 to end of 2026
                out_of_epoch = int(((valid_dates.dt.year < 1990) | (valid_dates.dt.year > 2026)).sum())
                report["invalid_dates"]["out_of_epoch_dates"] = out_of_epoch
                report["invalid_dates"]["date_min"] = str(valid_dates.min())
                report["invalid_dates"]["date_max"] = str(valid_dates.max())

                if out_of_epoch > 0:
                    report["critical_issues"].append(f"{out_of_epoch} records outside 1990-2026 epoch in '{date_col}'")

        # 4. Numeric & Price Microstructure Checks
        check_prices = price_cols or [c for c in ["open", "high", "low", "close"] if c in df.columns]

        for p_col in check_prices:
            if p_col in df.columns:
                series = pd.to_numeric(df[p_col], errors="coerce")
                nans = int(series.isna().sum())
                infs = int(np.isinf(series).sum()) if not series.empty else 0
                negs = int((series <= 0).sum())

                if nans > 0:
                    report["invalid_numerics"]["nan_counts"][p_col] = nans
                if infs > 0:
                    report["invalid_numerics"]["infinite_counts"][p_col] = infs
                if negs > 0:
                    report["invalid_numerics"]["negative_counts"][p_col] = negs
                    report["suspicious_prices"]["non_positive_prices"] += negs
                    report["critical_issues"].append(f"Discovered {negs} non-positive prices in '{p_col}'")

        # Microstructure consistency (High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close)
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            o = pd.to_numeric(df["open"], errors="coerce")
            h = pd.to_numeric(df["high"], errors="coerce")
            l = pd.to_numeric(df["low"], errors="coerce")
            c = pd.to_numeric(df["close"], errors="coerce")

            hl_viol = int((h < l).sum())
            report["suspicious_prices"]["high_less_than_low"] = hl_viol
            if hl_viol > 0:
                report["critical_issues"].append(f"Severe microstructure violation: {hl_viol} bars where High < Low")

            c_oob = int(((c > h) | (c < l)).sum())
            report["suspicious_prices"]["close_out_of_bounds"] = c_oob
            if c_oob > 0:
                report["critical_issues"].append(f"{c_oob} bars where Close is outside [Low, High]")

            o_oob = int(((o > h) | (o < l)).sum())
            report["suspicious_prices"]["open_out_of_bounds"] = o_oob

            # Check abnormal single-bar return jumps
            if "close" in df.columns and len(c) > 1:
                ret = c.pct_change().abs()
                extreme_jumps = int((ret > max_acceptable_price_jump).sum())
                report["suspicious_prices"]["extreme_price_jumps"] = extreme_jumps
                if extreme_jumps > 0:
                    report["warnings"].append(f"Observed {extreme_jumps} abnormal price jumps (> {max_acceptable_price_jump*100:.0f}%)")

        # 5. Volume checks
        if volume_col and volume_col in df.columns:
            v_series = pd.to_numeric(df[volume_col], errors="coerce")
            neg_v = int((v_series < 0).sum())
            if neg_v > 0:
                report["invalid_numerics"]["negative_counts"][volume_col] = neg_v
                report["critical_issues"].append(f"Negative volumes detected: {neg_v} in '{volume_col}'")

        # Final grade assignment
        if report["critical_issues"]:
            report["validation_grade"] = "CRITICAL_ISSUES_FOUND"
        elif report["warnings"]:
            report["validation_grade"] = "WARNINGS_FOUND"
        else:
            report["validation_grade"] = "PASSED"

        return report

    @staticmethod
    def render_markdown(report: Dict[str, Any]) -> str:
        """Render validation report dictionary as GitHub Markdown."""
        lines = [
            f"# Validation Report: {report.get('dataset_name')}",
            f"\n**Overall Grade:** `{report.get('validation_grade')}`",
            f"**Total Records Inspected:** {report.get('total_rows'):,}",
            "\n## 1. Critical Issues & Warnings",
        ]

        if report.get("critical_issues"):
            for crit in report["critical_issues"]:
                lines.append(f"- :x: **CRITICAL:** {crit}")
        else:
            lines.append("- :white_check_mark: No critical microstructure or corruption issues detected.")

        if report.get("warnings"):
            for warn in report["warnings"]:
                lines.append(f"- :warning: **WARNING:** {warn}")

        lines.extend([
            "\n## 2. Suspicious Price & Microstructure Metrics",
            f"- High < Low Violations: `{report['suspicious_prices'].get('high_less_than_low', 0)}`",
            f"- Close Outside [Low, High]: `{report['suspicious_prices'].get('close_out_of_bounds', 0)}`",
            f"- Non-Positive Price Values: `{report['suspicious_prices'].get('non_positive_prices', 0)}`",
            f"- Extreme Price Jumps: `{report['suspicious_prices'].get('extreme_price_jumps', 0)}`",
            "\n## 3. Duplication & Integrity",
            f"- Identical Duplicates: `{report['duplicate_records'].get('total_duplicates', 0)}`",
            f"- Key Collisions: `{report['duplicate_records'].get('key_collisions', 0)}`",
            f"- Unparseable Dates: `{report['invalid_dates'].get('unparseable_dates', 0)}`",
        ])

        return "\n".join(lines)
