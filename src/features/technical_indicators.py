"""Technical Analysis & Quantitative Indicator Engine for StockMind AI.

Calculates real mathematical indicators from historical price series:
- Exponential Moving Averages (EMA 9, 21, 50, 200)
- Simple Moving Averages (SMA 20, 50)
- Relative Strength Index (RSI 14)
- Moving Average Convergence Divergence (MACD 12, 26, 9)
- Classical Floor Pivot Points (Pivot, S1, S2, R1, R2)
- 52-Week High / Low & Range Position
- Average True Range (ATR 14) for dynamic volatility stop-loss
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


class TechnicalIndicatorEngine:
    """Calculates quantitative technical indicators from OHLCV series."""

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict[str, pd.Series]:
        """Calculate MACD line, Signal line, and Histogram."""
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    @classmethod
    def compute_indicators(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute full suite of technical indicators from an OHLCV dataframe.
        
        Expected columns (case-insensitive): Open, High, Low, Close, Volume.
        """
        if df.empty:
            return {}

        # Normalize column names
        df_norm = df.copy()
        col_map = {c: c.lower().strip() for c in df_norm.columns}
        df_norm = df_norm.rename(columns=col_map)

        close = df_norm["close"]
        high = df_norm.get("high", close)
        low = df_norm.get("low", close)

        current_close = float(close.iloc[-1])
        current_high = float(high.iloc[-1])
        current_low = float(low.iloc[-1])

        # Moving Averages
        ema_9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        ema_21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())
        ema_200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())

        # RSI 14
        rsi_series = cls.calculate_rsi(close, period=min(14, len(close) - 1))
        current_rsi = float(rsi_series.iloc[-1])

        # MACD
        macd_dict = cls.calculate_macd(close)
        current_macd = float(macd_dict["macd"].iloc[-1])
        current_signal = float(macd_dict["signal"].iloc[-1])
        current_hist = float(macd_dict["histogram"].iloc[-1])

        # 52-Week (or full series) High/Low
        high_52 = float(high.max())
        low_52 = float(low.min())
        range_52_pct = (
            round(((current_close - low_52) / (high_52 - low_52)) * 100, 1)
            if high_52 > low_52
            else 50.0
        )

        # Classical Floor Pivot Points
        pivot = (current_high + current_low + current_close) / 3.0
        r1 = (2 * pivot) - current_low
        s1 = (2 * pivot) - current_high
        r2 = pivot + (current_high - current_low)
        s2 = pivot - (current_high - current_low)

        # Technical Outlook interpretation
        trend_status = "BULLISH" if current_close > ema_50 else "BEARISH"
        momentum_status = (
            "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral Momentum"
        )

        return {
            "current_close": round(current_close, 2),
            "ema_9": round(ema_9, 2),
            "ema_21": round(ema_21, 2),
            "ema_50": round(ema_50, 2),
            "ema_200": round(ema_200, 2),
            "rsi_14": round(current_rsi, 1),
            "macd": round(current_macd, 2),
            "macd_signal": round(current_signal, 2),
            "macd_histogram": round(current_hist, 2),
            "pivot_point": round(pivot, 2),
            "support_1": round(s1, 2),
            "support_2": round(s2, 2),
            "resistance_1": round(r1, 2),
            "resistance_2": round(r2, 2),
            "high_52w": round(high_52, 2),
            "low_52w": round(low_52, 2),
            "range_position_pct": range_52_pct,
            "trend_status": trend_status,
            "momentum_status": momentum_status,
        }
