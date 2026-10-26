"""Unit tests for src.data.news_timestamp_extractor module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.news_timestamp_extractor import (
    NewsTimestampExtractor,
    TimestampConfidence,
    extract_from_content,
    extract_from_url,
)


def test_extract_from_url_iso_path():
    url = "https://www.financialexpress.com/market/2021/04/15/sensex-today-live-updates/"
    res = extract_from_url(url)
    assert res is not None
    assert res.is_resolved is True
    assert res.extracted_date == "2021-04-15"
    assert res.confidence == TimestampConfidence.HIGH.value
    assert res.method == "url_path_ymd"


def test_extract_from_url_slug_compact():
    url = "https://www.business-standard.com/article/markets/markets-live-20220930_1.html"
    res = extract_from_url(url)
    assert res is not None
    assert res.is_resolved is True
    assert res.extracted_date == "2022-09-30"
    assert res.confidence == TimestampConfidence.HIGH.value


def test_extract_from_url_no_date():
    url = "https://www.moneycontrol.com/news/business/markets/fiis-turn-net-sellers-4885251.html"
    res = extract_from_url(url)
    assert res is None


def test_extract_from_content_dateline():
    content = "MUMBAI, OCT 24, 2020: The Reserve Bank of India announced a new liquidity framework."
    res = extract_from_content(content)
    assert res is not None
    assert res.is_resolved is True
    assert res.extracted_date == "2020-10-24"
    assert res.confidence == TimestampConfidence.HIGH.value


def test_extract_from_content_explicit_dmy():
    content = "Indian stock markets are likely to open lower on Wednesday, 7 March 2018 following the declines in US stock futures."
    res = extract_from_content(content)
    assert res is not None
    assert res.is_resolved is True
    assert res.extracted_date == "2018-03-07"
    assert res.confidence == TimestampConfidence.HIGH.value


def test_extract_from_content_no_date():
    content = "Markets traded higher today as corporate earnings showed broad-based recovery."
    res = extract_from_content(content)
    assert res is None


def test_extractor_single_fallback_and_unresolved():
    extractor = NewsTimestampExtractor()

    # Case 1: URL has date
    res1 = extractor.extract_single(
        url="https://domain.com/news/2020/05/12/stock-update.html",
        content="General news content without date.",
    )
    assert res1.is_resolved is True
    assert res1.extracted_date == "2020-05-12"

    # Case 2: URL has no date, content has dateline
    res2 = extractor.extract_single(
        url="https://domain.com/news/article-12345.html",
        content="NEW DELHI, March 15 2019 - Direct tax collections exceeded budget targets.",
    )
    assert res2.is_resolved is True
    assert res2.extracted_date == "2019-03-15"

    # Case 3: Neither has date -> strictly UNRESOLVED (never invent timestamps)
    res3 = extractor.extract_single(
        url="https://domain.com/news/generic-article.html",
        content="Oil prices surged after production cuts were announced.",
    )
    assert res3.is_resolved is False
    assert res3.extracted_date is None
    assert res3.confidence == TimestampConfidence.UNRESOLVED.value


def test_extract_batch():
    extractor = NewsTimestampExtractor()
    df = pd.DataFrame({
        "URL": [
            "https://domain.com/2020/01/10/first.html",
            "https://domain.com/no-date-here.html",
        ],
        "Content": [
            "Content 1",
            "No dateline here either.",
        ],
    })
    enriched_df, summary = extractor.extract_batch(df)

    assert len(enriched_df) == 2
    assert "extracted_date" in enriched_df.columns
    assert "is_resolved" in enriched_df.columns
    assert summary["total_articles_analyzed"] == 2
    assert summary["resolved_count"] == 1
    assert summary["unresolved_count"] == 1
    assert summary["resolved_percentage"] == 50.0
