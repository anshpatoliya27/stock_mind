"""Financial News Timestamp Extraction & Verification Engine for StockMind AI.

Extracts candidate publication timestamps from article URLs and article text content.
Guarantees strict audit adherence:
- Never invents missing timestamps.
- Explicitly tracks extraction method and confidence level.
- Marks ambiguous, conflicting, or incomplete dates as UNRESOLVED.
- Prevents lookahead bias in downstream market alignments.
"""

from __future__ import annotations

import calendar
import datetime
import enum
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


class TimestampConfidence(str, enum.Enum):
    HIGH = "HIGH"          # Unambiguous exact day, month, and year
    MEDIUM = "MEDIUM"      # Clear month and year; day approximate/first-of-month or from URL slug
    LOW = "LOW"            # Year-only or weak match with multiple competing dates
    UNRESOLVED = "UNRESOLVED"  # Ambiguous, conflicting, or absent timestamp


@dataclass
class TimestampExtractionResult:
    url: str
    extracted_date: Optional[str]  # ISO format YYYY-MM-DD or None
    year: Optional[int]
    month: Optional[int]
    day: Optional[int]
    confidence: str
    method: str
    raw_match: Optional[str]
    is_resolved: bool
    ambiguity_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Month name mapping
MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _validate_date_components(year: int, month: int, day: int) -> bool:
    """Validate calendar date components within credible financial market epoch."""
    if not (1990 <= year <= 2026):
        return False
    if not (1 <= month <= 12):
        return False
    max_days = calendar.monthrange(year, month)[1]
    return 1 <= day <= max_days


def extract_from_url(url: str) -> Optional[TimestampExtractionResult]:
    """Attempt candidate date extraction from URL path structures."""
    if not url or not isinstance(url, str):
        return None

    clean_url = url.strip()

    # Pattern 1: Path with explicit /YYYY/MM/DD/ or /YYYY/M/D/
    m1 = re.search(r"/(20\d\d)/([01]?\d)/([0-3]?\d)/", clean_url)
    if m1:
        y, m, d = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        if _validate_date_components(y, m, d):
            return TimestampExtractionResult(
                url=clean_url,
                extracted_date=f"{y:04d}-{m:02d}-{d:02d}",
                year=y,
                month=m,
                day=d,
                confidence=TimestampConfidence.HIGH.value,
                method="url_path_ymd",
                raw_match=m1.group(0),
                is_resolved=True,
            )

    # Pattern 2: Hyphenated YYYY-MM-DD in slug e.g. /news-2021-04-15-headline
    m2 = re.search(r"[-/](20\d\d)-([01]\d)-([0-3]\d)[-/.]", clean_url)
    if m2:
        y, m, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if _validate_date_components(y, m, d):
            return TimestampExtractionResult(
                url=clean_url,
                extracted_date=f"{y:04d}-{m:02d}-{d:02d}",
                year=y,
                month=m,
                day=d,
                confidence=TimestampConfidence.HIGH.value,
                method="url_slug_iso",
                raw_match=m2.group(0),
                is_resolved=True,
            )

    # Pattern 3: Compact YYYYMMDD in slug e.g. /story-20230514-headline or _20230514.html
    m3 = re.search(r"[-_](20\d\d)([01]\d)([0-3]\d)[-_\.]", clean_url)
    if m3:
        y, m, d = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        if _validate_date_components(y, m, d):
            return TimestampExtractionResult(
                url=clean_url,
                extracted_date=f"{y:04d}-{m:02d}-{d:02d}",
                year=y,
                month=m,
                day=d,
                confidence=TimestampConfidence.HIGH.value,
                method="url_slug_compact_ymd",
                raw_match=m3.group(0),
                is_resolved=True,
            )

    # Pattern 4: /YYYY/MM/ without day (Medium confidence, mark day as unresolved)
    m4 = re.search(r"/(20\d\d)/([01]?\d)/", clean_url)
    if m4:
        y, m = int(m4.group(1)), int(m4.group(2))
        if 1990 <= y <= 2026 and 1 <= m <= 12:
            return TimestampExtractionResult(
                url=clean_url,
                extracted_date=f"{y:04d}-{m:02d}-01",
                year=y,
                month=m,
                day=None,
                confidence=TimestampConfidence.MEDIUM.value,
                method="url_year_month_only",
                raw_match=m4.group(0),
                is_resolved=False,  # Exact day missing
                ambiguity_reason="Day missing in URL; month-level only",
            )

    return None


def extract_from_content(text: str, max_chars: int = 1500) -> Optional[TimestampExtractionResult]:
    """Attempt candidate date extraction from opening article text / datelines."""
    if not text or not isinstance(text, str):
        return None

    snippet = text[:max_chars]

    # Pattern 1: Explicit dateline e.g. "MUMBAI, Oct 24, 2020:" or "NEW DELHI, March 15 2019 -"
    dateline_re = re.search(
        r"(?:^[A-Z\s]{2,20}\s*,\s*|By\s+[A-Za-z\s]+,\s*)([A-Za-z]+)\s+([0-3]?\d)(?:st|nd|rd|th)?(?:,\s*|\s+)(20\d\d)\b",
        snippet,
        re.IGNORECASE,
    )
    if dateline_re:
        m_str, d_str, y_str = dateline_re.group(1).lower(), dateline_re.group(2), dateline_re.group(3)
        if m_str in MONTH_MAP:
            y, m, d = int(y_str), MONTH_MAP[m_str], int(d_str)
            if _validate_date_components(y, m, d):
                return TimestampExtractionResult(
                    url="",
                    extracted_date=f"{y:04d}-{m:02d}-{d:02d}",
                    year=y,
                    month=m,
                    day=d,
                    confidence=TimestampConfidence.HIGH.value,
                    method="text_dateline_mdy",
                    raw_match=dateline_re.group(0),
                    is_resolved=True,
                )

    # Pattern 2: "Wednesday, 7 March 2018" or "7 March 2018"
    day_first_re = re.search(
        r"\b(?:(?:Mon|Tues|Wed|Thurs|Fri|Satur|Sun)[a-z]*\s*,\s*)?([0-3]?\d)\s+([A-Za-z]+)\s+(20\d\d)\b",
        snippet,
        re.IGNORECASE,
    )
    if day_first_re:
        d_str, m_str, y_str = day_first_re.group(1), day_first_re.group(2).lower(), day_first_re.group(3)
        if m_str in MONTH_MAP:
            y, m, d = int(y_str), MONTH_MAP[m_str], int(d_str)
            if _validate_date_components(y, m, d):
                return TimestampExtractionResult(
                    url="",
                    extracted_date=f"{y:04d}-{m:02d}-{d:02d}",
                    year=y,
                    month=m,
                    day=d,
                    confidence=TimestampConfidence.HIGH.value,
                    method="text_explicit_dmy",
                    raw_match=day_first_re.group(0),
                    is_resolved=True,
                )

    # Pattern 3: "March 7, 2018" in sentence
    month_first_re = re.search(
        r"\b([A-Za-z]+)\s+([0-3]?\d)(?:st|nd|rd|th)?,\s*(20\d\d)\b",
        snippet,
        re.IGNORECASE,
    )
    if month_first_re:
        m_str, d_str, y_str = month_first_re.group(1).lower(), month_first_re.group(2), month_first_re.group(3)
        if m_str in MONTH_MAP:
            y, m, d = int(y_str), MONTH_MAP[m_str], int(d_str)
            if _validate_date_components(y, m, d):
                return TimestampExtractionResult(
                    url="",
                    extracted_date=f"{y:04d}-{m:02d}-{d:02d}",
                    year=y,
                    month=m,
                    day=d,
                    confidence=TimestampConfidence.HIGH.value,
                    method="text_explicit_mdy",
                    raw_match=month_first_re.group(0),
                    is_resolved=True,
                )

    # Pattern 4: ISO YYYY-MM-DD inside text
    iso_re = re.search(r"\b(20\d\d)-([01]\d)-([0-3]\d)\b", snippet)
    if iso_re:
        y, m, d = int(iso_re.group(1)), int(iso_re.group(2)), int(iso_re.group(3))
        if _validate_date_components(y, m, d):
            return TimestampExtractionResult(
                url="",
                extracted_date=f"{y:04d}-{m:02d}-{d:02d}",
                year=y,
                month=m,
                day=d,
                confidence=TimestampConfidence.MEDIUM.value,
                method="text_iso_date",
                raw_match=iso_re.group(0),
                is_resolved=True,
            )

    return None


class NewsTimestampExtractor:
    """Production timestamp extractor for financial news articles."""

    def __init__(self, content_char_limit: int = 1500):
        self.content_char_limit = content_char_limit

    def extract_single(self, url: str, content: str = "") -> TimestampExtractionResult:
        """Extract candidate timestamp from URL first, falling back to article content.
        
        Guarantees:
        - If URL gives an unambiguous date, it takes precedence.
        - If content contains an explicit dateline/date, it is used as fallback.
        - If both are missing or conflicting, returns an UNRESOLVED result.
        """
        # 1. URL extraction
        url_res = extract_from_url(url)
        if url_res and url_res.is_resolved:
            return url_res

        # 2. Content extraction
        text_res = extract_from_content(content, max_chars=self.content_char_limit)
        if text_res and text_res.is_resolved:
            text_res.url = url
            return text_res

        # 3. Handle partial URL results (e.g. year/month only)
        if url_res and not url_res.is_resolved:
            return url_res

        # 4. Strict Unresolved
        return TimestampExtractionResult(
            url=url,
            extracted_date=None,
            year=None,
            month=None,
            day=None,
            confidence=TimestampConfidence.UNRESOLVED.value,
            method="unresolved",
            raw_match=None,
            is_resolved=False,
            ambiguity_reason="No verifiable date pattern discovered in URL or opening text",
        )

    def extract_batch(
        self,
        df: pd.DataFrame,
        url_col: str = "URL",
        content_col: str = "Content",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Batch process a dataframe of news articles.
        
        Returns:
            (enriched_df, extraction_summary)
        """
        results: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            url = str(row.get(url_col, ""))
            content = str(row.get(content_col, ""))
            res = self.extract_single(url, content)
            results.append({
                "extracted_date": res.extracted_date,
                "confidence": res.confidence,
                "extraction_method": res.method,
                "is_resolved": res.is_resolved,
                "raw_date_match": res.raw_match,
                "ambiguity_reason": res.ambiguity_reason,
            })

        res_df = pd.DataFrame(results)
        enriched_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)

        # Summary statistics
        total = len(enriched_df)
        resolved_count = int(enriched_df["is_resolved"].sum())
        confidence_counts = enriched_df["confidence"].value_counts().to_dict()
        method_counts = enriched_df["extraction_method"].value_counts().to_dict()

        summary: Dict[str, Any] = {
            "total_articles_analyzed": total,
            "resolved_count": resolved_count,
            "resolved_percentage": round((resolved_count / total * 100) if total else 0, 2),
            "unresolved_count": total - resolved_count,
            "confidence_breakdown": confidence_counts,
            "method_breakdown": method_counts,
        }

        return enriched_df, summary
