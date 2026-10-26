# StockMind AI: Data Sources, Lineage & Validation Registry

This document establishes the official data provenance, licensing parameters, schema characteristics, known overlaps, and fields requiring strict validation for all datasets within the **StockMind AI** framework.

---

## 1. Primary Dataset Registry

| Category | Storage Path | Format & File Count | Size | Observed Temporal Span | Primary Origin / Source |
|---|---|---|---|---|---|
| **Market Data (Daily Parquet)** | `Dataset/market_data/nse_daily/` | 8 Parquet files | 103 MB (compressed) | 2019-01-01 to 2026-10-23 | National Stock Exchange of India (NSE) historical trade archives |
| **Market Data (Daily Company CSV)** | `Dataset/market_data/comapanies_data/` | 5,028 CSV files (1,549 `.NS`, 3,479 `.BO`) | ~1.3 GB | ~2008 to ~2020 (ticker-dependent) | Historical quotes (NSE & BSE equities) |
| **Market Data (Minute Tickers & Indices)** | `Dataset/market_data/comapanies_minutes/` | 515 CSV files (499 stocks, 13 indices, 3 manifests) | ~19.2 GB | 2015-02-02 to 2026-01-23 | High-frequency intraday market data feeds (1-minute resolution) |
| **Financial News** | `Dataset/financial_news/` | 1 CSV (`training_data_26000.csv`) | 109.7 MB | Unverified (URL-dependent) | Indian financial media (Moneycontrol, Business Today, Mint) |
| **Corporate Events (Bulk Deals)** | `Dataset/corporate_events/` | 1 CSV (`Bulk-Deals-25-10-2025-to-25-10-2026.csv`) | 3.08 MB | 2025-10-27 to 2026-10-23 | Official NSE/BSE Bulk Deals statutory disclosures |
| **Forex Rates** | `Dataset/macroeconomic/forex/` | 1 Parquet, 2 CSVs | 1.1 MB | ~1998 to ~2026 | Reserve Bank of India (RBI) Reference Rates |
| **India CPI** | `Dataset/macroeconomic/india_cpi/` | 1 XLSX (`CPI - Rural, Urban, Combined.xlsx`) | 472 KB | 2011 to 2026 (Monthly) | Ministry of Statistics and Programme Implementation (MoSPI) / RBI |
| **Monetary Policy Rates** | `Dataset/macroeconomic/monetary_policy/` | 1 XLSX (`Major Monetary Policy Rates...xlsx`) | 16 KB | Multi-year policy rate history | Reserve Bank of India (RBI) Monetary Policy Committee |
| **World Bank Macro Indicators** | `Dataset/macroeconomic/world_bank/` | 6 CSVs (Data + Country/Indicator Metadata) | ~660 KB | 1960 to 2025 (Annual) | World Bank Open Data (World Development Indicators) |
| **Chart Vision & Segmentation** | `Dataset/chart_images/` | 340 JPGs (`Patterns/`), 340 PNGs (`Segmentation/`) | ~1.4 MB | Aligned with `comapanies_minutes` | Computer vision technical pattern detection benchmark |

---

## 2. Dataset Overlap & Redundancy Analysis

Per engineering guidelines, duplicate-looking or overlapping files **must not be deleted**. Instead, their functional roles in the architecture are classified below:

### OVERLAP-01: Macroeconomic Forex Representations
- **Files Involved:**
  - `Dataset/macroeconomic/forex/rbi_forex_reference_rates.parquet` (26,977 rows)
  - `Dataset/macroeconomic/forex/rbi_forex_reference_rates_long.csv` (26,977 rows)
  - `Dataset/macroeconomic/forex/rbi_forex_reference_rates_wide.csv` (Pivoted columns: `AED`, `EUR`, `GBP`, `IDR`, `JPY`, `USD`)
- **Analysis:** The Parquet and long CSV files contain identical normalized triples `[date, currency, rate, unit]`. The wide CSV represents the same data in denormalized form.
- **Architectural Directive:** Standardize downstream Big Data jobs on `rbi_forex_reference_rates.parquet` for optimal zero-copy columnar scanning. Keep CSV files intact for cross-verification.

### OVERLAP-02: Market Data (Cross-Sectional vs. Single-Ticker Longitudinal)
- **Files Involved:**
  - `Dataset/market_data/nse_daily/nse_2019.parquet` through `nse_2026.parquet`
  - `Dataset/market_data/comapanies_data/*.NS.csv`
- **Analysis:** `nse_daily` contains comprehensive market-wide cross-sections (every traded stock per trading day from 2019 to 2026). `comapanies_data` contains isolated historical trajectories (many ending around 2020).
- **Architectural Directive:** Use `nse_daily` as the authoritative source for modern multi-asset modeling (2019–2026). Use `comapanies_data` for historical pre-2019 training or survivorship benchmarking.

### OVERLAP-03: Visual Images vs. Metadata Catalogs
- **Files Involved:**
  - `Dataset/chart_images/Patterns/` (340 JPG images)
  - `Dataset/chart_images/Segmentation/` (340 PNG segmentation masks)
  - `Dataset/market_data/comapanies_minutes/Patterns.csv`
  - `Dataset/market_data/comapanies_minutes/Segmentation.csv`
- **Analysis:** `Patterns.csv` and `Segmentation.csv` located in `comapanies_minutes/` are the exact catalog manifests mapping image filenames to class labels (`Double top`, etc.), width/height, and bounding dimensions.
- **Architectural Directive:** Treat these pairs as a unified multimodal computer vision dataset. Do not move or rename.

---

## 3. Critical Validation Flags & Fields Needing Verification

### FLAG-01: Financial News Publication Timestamps (CRITICAL)
- **Component:** `Dataset/financial_news/training_data_26000.csv`
- **Finding:** Columns are strictly `['URL', 'Content', 'Summary', 'Sentiment']`. **There is no explicit `published_at` or `article_date` column.**
- **Action for Validation Phase:**
  1. Inspect whether URLs contain date patterns (e.g. `/news/business/economy/article-name-20240315.html`).
  2. If URLs lack complete timestamps, apply an extraction pipeline using regex/spaCy on the article `Content` opening text (e.g., datelines like `"MUMBAI, OCT 24:"`).
  3. Prohibit any live temporal market merge until publication timestamps are verified to avoid lookahead bias.

### FLAG-02: Column Name Whitespace Normalization (WARNING)
- **Component:** `Dataset/corporate_events/Bulk-Deals-25-10-2025-to-25-10-2026.csv`
- **Finding:** Raw CSV headers contain trailing whitespace: `'Date '`, `'Symbol '`, `'Security Name '`, `'Client Name '`, `'Buy / Sell '`, `'Quantity Traded '`, `'Trade Price / Wght. Avg. Price '`, `'Remarks '`.
- **Action for Validation Phase:** Ingestion schemas must execute `.str.strip()` on column names before dictionary lookups.

### FLAG-03: Numeric Formatting with Thousands Separators
- **Component:** `Dataset/corporate_events/Bulk-Deals-25-10-2025-to-25-10-2026.csv`
- **Finding:** The price column `'Trade Price / Wght. Avg. Price '` contains strings formatted with commas (e.g. `"1,179.70"`).
- **Action for Validation Phase:** Coercion pipeline must sanitize comma characters prior to float conversion.

### FLAG-04: Leading Metadata Rows in Macroeconomic CSVs
- **Component:** `Dataset/macroeconomic/world_bank/*.csv`
- **Finding:** World Bank CSV files contain 4 commentary lines before row 5 header names.
- **Action for Validation Phase:** Parsing routines must enforce `skiprows=4`.

### FLAG-05: Missing Value Handling across Minute-Level Data
- **Component:** `Dataset/market_data/comapanies_minutes/*_minute.csv`
- **Finding:** Minute files range up to 1,000,000+ lines per ticker. Non-trading minutes (holidays, trading pauses, circuit breakers) result in irregular timestamps.
- **Action for Validation Phase:** Determine if downsampling to 5-minute / 15-minute bars or forward-filling price during market hours is required.

---

## 4. Licensing and Academic Use Evaluation

1. **NSE & BSE Trade Records:** Official market disclosures. Under fair academic research use for university coursework, these records can be processed locally. Redistribution of raw tick data is restricted.
2. **RBI Forex & Monetary Policy Data:** Published by the Reserve Bank of India under Government of India open data initiatives. Publicly accessible statistical reports.
3. **MoSPI CPI Data:** Public statistical index published by the Indian Government.
4. **World Bank WDI:** Open Data under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**.
5. **Financial News Data:** Sourced for research in financial NLP and sentiment classification. Strictly for offline academic experimentation; no commercial publication.
