# Apache Spark & Big Data Jobs

This directory contains batch and streaming PySpark jobs for large-scale Indian stock market analytics.

## Structure & Modules (Planned)
- `ingest_nse_daily.py`: Distributed partition-aware transformation of raw historical NSE parquet files.
- `minute_aggregations.py`: Spark streaming / distributed windowing across 500+ minute-level ticker datasets.
- `macro_alignment_job.py`: Large-scale temporal joins between market ticks, macroeconomic indicators (RBI, CPI, World Bank), and bulk deals.
- `news_nlp_pipeline.py`: Scalable text cleaning, tokenization, and distributed feature extraction for financial news articles.
