# StockMind AI: Comprehensive Data Inventory & Audit Report

**Audit Timestamp:** 2026-10-25T02:34:09.998565

**Dataset Location:** `C:\Users\khush\OneDrive\Desktop\stockmind\Dataset`

## 1. Executive Summary

- **Total Files:** 6,244
- **Total Storage Footprint:** 20.62 GB (22,141,872,923 bytes)
- **Primary Formats:** Parquet (NSE daily, RBI Forex), CSV (Minute ticks, company OHLCV, News, Bulk Deals), XLSX (CPI, Monetary Policy), JPG/PNG (Chart Vision).

## 2. Category Audits

### 2.1 Market Data

#### A. NSE Daily Parquet (2019–2026)
- **Total Partitions:** 8 Parquet files
- **Aggregate Trades/Rows:** 3,919,007
| Year / File | Rows | File Size | Schema Columns | Date Range |
|---|---|---|---|---|
| `nse_2019.parquet` | 406,878 | 10.06 MB | 16 cols | 2019-01-01 to 2019-12-31 |
| `nse_2020.parquet` | 424,952 | 10.74 MB | 16 cols | 2020-01-01 to 2020-12-31 |
| `nse_2021.parquet` | 443,555 | 11.23 MB | 16 cols | 2021-01-01 to 2021-12-31 |
| `nse_2022.parquet` | 479,856 | 12.21 MB | 16 cols | 2022-01-03 to 2022-12-30 |
| `nse_2023.parquet` | 503,785 | 12.88 MB | 16 cols | 2023-01-02 to 2023-12-29 |
| `nse_2024.parquet` | 540,760 | 14.95 MB | 16 cols | 2024-01-01 to 2024-12-31 |
| `nse_2025.parquet` | 592,230 | 16.36 MB | 16 cols | 2025-01-01 to 2025-12-31 |
| `nse_2026.parquet` | 526,991 | 14.51 MB | 16 cols | 2026-01-01 to 2026-10-23 |

#### B. Company-Level Daily Stock Data (`comapanies_data`)
- **Total Files:** 5028 CSV files
- **National Stock Exchange (.NS):** 1549 tickers
- **Bombay Stock Exchange (.BO):** 3479 tickers
- **Standard Schema:** `['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']`

#### C. Minute-Level Stock Data (`comapanies_minutes`)
- **Total Files:** 515 files
- **Minute Stock Tickers:** 499 tickers (up to 1,000,000+ rows per ticker)
- **Index Tickers:** 13 files (`NIFTY 50`, `NIFTY BANK`, `India VIX`, `SENSEX`, etc.)
- **Catalog Manifests:** `Patterns.csv`, `Segmentation.csv`, `Meta.csv`
- **Schema:** `['date', 'open', 'high', 'low', 'close', 'volume']`

### 2.2 Financial News

- **File:** `training_data_26000.csv` (109.66 MB)
- **Row Count:** ~610,624 records
- **Columns:** `['URL', 'Content', 'Summary', 'Sentiment']`
- **Sample Nulls:** `{'URL': 0, 'Content': 0, 'Summary': 0, 'Sentiment': 0}`

### 2.3 Corporate Events

- **File:** `Bulk-Deals-25-10-2025-to-25-10-2026.csv` (3.08 MB)
- **Estimated Records:** ~29,087 bulk trade rows
- **Raw Columns:** `['Date ', 'Symbol ', 'Security Name ', 'Client Name ', 'Buy / Sell ', 'Quantity Traded ', 'Trade Price / Wght. Avg. Price ', 'Remarks ']`
- **Whitespace Detected in Columns:** `['Date ', 'Symbol ', 'Security Name ', 'Client Name ', 'Buy / Sell ', 'Quantity Traded ', 'Trade Price / Wght. Avg. Price ', 'Remarks ']`

### 2.4 Macroeconomic Data

#### Subcategory: `forex`
- `rbi_forex_reference_rates.parquet` (181.50 KB): 26,977 rows, columns: `['date', 'currency', 'rate', 'unit']`
- `rbi_forex_reference_rates_long.csv` (647.27 KB): columns: `['date', 'currency', 'rate', 'unit']`
- `rbi_forex_reference_rates_wide.csv` (270.17 KB): columns: `['date', 'AED', 'EUR', 'GBP', 'IDR', 'JPY', 'USD']`

#### Subcategory: `india_cpi`
- `CPI - Rural, Urban, Combined.xlsx` (471.99 KB): Sheets: `['CPI - 2010=100 (All India)', 'CPI - 2012=100 (All India) (1)']`

#### Subcategory: `monetary_policy`
- `Major Monetary Policy Rates and Reserve Requirements - Bank Rate, LAF (Repo, Reverse Repo, SDF and MSF) Rates, CRR & SLR.xlsx` (15.80 KB): Sheets: `['Report 1']`

#### Subcategory: `world_bank`
- `API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_1323.csv` (253.61 KB): columns: `['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', 'Unnamed: 70']`
- `API_FP.CPI.TOTL_DS2_en_csv_v2_253.csv` (214.91 KB): columns: `['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', 'Unnamed: 70']`
- `Metadata_Country_API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_1323.csv` (63.70 KB): columns: `['Country Code', 'Region', 'IncomeGroup', 'SpecialNotes', 'TableName', 'Unnamed: 5']`
- `Metadata_Country_API_FP.CPI.TOTL_DS2_en_csv_v2_253.csv` (63.70 KB): columns: `['Country Code', 'Region', 'IncomeGroup', 'SpecialNotes', 'TableName', 'Unnamed: 5']`
- `Metadata_Indicator_API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_1323.csv` (454 B): columns: `['INDICATOR_CODE', 'INDICATOR_NAME', 'SOURCE_NOTE', 'SOURCE_ORGANIZATION', 'Unnamed: 4']`
- `Metadata_Indicator_API_FP.CPI.TOTL_DS2_en_csv_v2_253.csv` (319 B): columns: `['INDICATOR_CODE', 'INDICATOR_NAME', 'SOURCE_NOTE', 'SOURCE_ORGANIZATION', 'Unnamed: 4']`

### 2.5 Chart Images & Vision

- **Patterns:** 340 files (677.90 KB), Extensions: `{'.jpg': 340}`
- **Segmentation:** 340 files (686.66 KB), Extensions: `{'.png': 340}`

## 3. Cross-Dataset Overlaps & Alignment

### OVERLAP-01: Macroeconomic Forex
- **Datasets Involved:**
  - `Dataset/macroeconomic/forex/rbi_forex_reference_rates.parquet`
  - `Dataset/macroeconomic/forex/rbi_forex_reference_rates_long.csv`
  - `Dataset/macroeconomic/forex/rbi_forex_reference_rates_wide.csv`
- **Analysis:** The parquet file and long CSV share identical schema ['date', 'currency', 'rate', 'unit']. The wide CSV is the pivoted representation ['date', 'AED', 'EUR', 'GBP', 'IDR', 'JPY', 'USD']. Action: Retain all files; standardize on Parquet during Big Data feature engineering to save I/O.

### OVERLAP-02: Historical Market Data (NSE)
- **Datasets Involved:**
  - `Dataset/market_data/nse_daily/nse_2019.parquet through nse_2026.parquet`
  - `Dataset/market_data/comapanies_data/*.NS.csv`
- **Analysis:** The nse_daily parquet partition contains market-wide daily OHLCV trades for all NSE equities from 2019 to 2026. The comapanies_data/*.NS.csv contains individual ticker histories (e.g. 2008-2020). Action: Flag temporal coverage differences; use nse_daily for cross-sectional market modeling and company CSVs for long-horizon pre-2019 historical context.

### OVERLAP-03: Chart Vision & Pattern Metadata
- **Datasets Involved:**
  - `Dataset/chart_images/Patterns (340 JPG images)`
  - `Dataset/chart_images/Segmentation (340 PNG masks)`
  - `Dataset/market_data/comapanies_minutes/Patterns.csv`
  - `Dataset/market_data/comapanies_minutes/Segmentation.csv`
- **Analysis:** The CSV files in comapanies_minutes catalogue the exact image filenames, class labels, and bounding dimensions for the chart images in chart_images. Action: Keep intact; utilize Patterns.csv and Segmentation.csv as PyTorch Dataset manifest catalogs.

## 4. Critical Data Validation Flags

### [CRITICAL] FLAG-01-NEWS-TIMESTAMPS
- **Component:** `Dataset/financial_news/training_data_26000.csv`
- **Observation:** The financial news dataset contains columns ['URL', 'Content', 'Summary', 'Sentiment']. There is NO explicit published_date or article_timestamp column! Publication dates must either be parsed from URL slugs or extracted from text metadata before any time-aligned backtesting or sentiment join can be performed.

### [WARNING] FLAG-02-WHITESPACE-COLUMNS
- **Component:** `Dataset/corporate_events/Bulk-Deals-25-10-2025-to-25-10-2026.csv`
- **Observation:** Bulk deals CSV column headers contain trailing whitespace (e.g., 'Date ', 'Symbol ', 'Security Name '). Data ingestion routines must strip column whitespace during normalization.

### [WARNING] FLAG-03-WORLDBANK-METADATA-HEADER
- **Component:** `Dataset/macroeconomic/world_bank/*.csv`
- **Observation:** World Bank CSV files have 4 leading metadata commentary rows before the column headers. Standard pd.read_csv fails unless skiprows=4 is applied.

### [INFO] FLAG-04-VOLUME-SCALE-CHECK
- **Component:** `Dataset/market_data/comapanies_minutes/ (20+ GB Total)`
- **Observation:** Minute-level data encompasses 515 ticker and index files totaling over 20 GB. Direct in-memory loading is prohibited; distributed PySpark windowing or chunked out-of-core Polars/DuckDB execution will be mandated for feature extraction.

