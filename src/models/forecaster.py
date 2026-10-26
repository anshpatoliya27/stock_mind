"""Standalone Stock Forecasting and Inference Engine for StockMind AI.

Designed for complete decoupling from model training so PaperTradex or other
live market systems can load the model artifacts, validate incoming features,
and generate real-time 1-day and 5-day return forecasts without needing training dependencies.

DATA SUPPLY PROTOCOL FOR PAPERTRADEX:
A single live price tick does NOT contain enough information to calculate technical
indicators (such as 14-day RSI, 200-day EMA, 20-day volatility). PaperTradex must
maintain an in-memory rolling buffer of historical daily OHLCV bars (minimum 50 bars,
recommended 250 bars) for each subscribed ticker.

PaperTradex passes this rolling window to `StockForecaster.prepare_features_from_history()`,
which extracts the zero-leakage feature vector for the current bar t, validates the schema,
and produces the forward return predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger("stockmind.forecaster")


class SchemaValidationError(ValueError):
    """Raised when incoming features violate model schema expectations."""
    pass


@dataclass
class ForecastResult:
    """Standardized prediction output for live execution systems."""
    symbol: str
    timestamp: str
    target_1d_pred_pct: float
    target_5d_pred_pct: float
    direction_1d: str  # "UP" or "DOWN"
    direction_5d: str  # "UP" or "DOWN"
    features_used: Dict[str, float]


class StockForecaster:
    """Production inference engine for Model 1: StockMind Return Predictor."""

    def __init__(
        self,
        model_1d: Optional[Any] = None,
        model_5d: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        feature_schema: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model_1d = model_1d
        self.model_5d = model_5d
        self.preprocessor = preprocessor
        self.feature_schema = feature_schema or []
        self.metadata = metadata or {}

    @classmethod
    def load(cls, model_dir: Path | str) -> StockForecaster:
        """Load exported model artifacts, preprocessor, and feature schema from disk.
        
        Expected directory structure:
            model_dir/
                ├── model_1d.joblib
                ├── model_5d.joblib
                ├── preprocessor.joblib
                └── feature_schema.json
        """
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        schema_file = path / "feature_schema.json"
        if not schema_file.exists():
            raise FileNotFoundError(f"Missing feature_schema.json in {model_dir}")

        with open(schema_file, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
            feature_names = schema_data.get("feature_names", [])
            metadata = schema_data.get("metadata", {})

        model_1d_path = path / "model_1d.joblib"
        model_5d_path = path / "model_5d.joblib"
        preprocessor_path = path / "preprocessor.joblib"

        if not model_1d_path.exists() or not model_5d_path.exists() or not preprocessor_path.exists():
            raise FileNotFoundError(f"Missing required .joblib artifacts in {model_dir}")

        model_1d = joblib.load(model_1d_path)
        model_5d = joblib.load(model_5d_path)
        preprocessor = joblib.load(preprocessor_path)

        logger.info(f"Loaded StockForecaster from {model_dir} ({len(feature_names)} features)")
        return cls(
            model_1d=model_1d,
            model_5d=model_5d,
            preprocessor=preprocessor,
            feature_schema=feature_names,
            metadata=metadata,
        )

    def save(self, output_dir: Path | str) -> None:
        """Save forecaster artifacts into output directory."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.model_1d is not None:
            joblib.dump(self.model_1d, path / "model_1d.joblib")
        if self.model_5d is not None:
            joblib.dump(self.model_5d, path / "model_5d.joblib")
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, path / "preprocessor.joblib")

        schema_payload = {
            "feature_names": self.feature_schema,
            "feature_count": len(self.feature_schema),
            "metadata": self.metadata,
        }
        with open(path / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema_payload, f, indent=2)

    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strictly validate feature names, ordering, and missing values.
        
        Ensures incoming features match the exact training schema.
        """
        if not isinstance(df, pd.DataFrame):
            raise SchemaValidationError(f"Expected pandas DataFrame, got {type(df)}")

        missing_cols = [c for c in self.feature_schema if c not in df.columns]
        if missing_cols:
            raise SchemaValidationError(
                f"Feature validation failed: missing required features: {missing_cols}"
            )

        # Enforce exact column order from schema
        aligned_df = df[self.feature_schema].copy()

        # Check for non-numeric types
        for col in self.feature_schema:
            if not np.issubdtype(aligned_df[col].dtype, np.number):
                try:
                    aligned_df[col] = aligned_df[col].astype(np.float64)
                except Exception as e:
                    raise SchemaValidationError(
                        f"Feature '{col}' cannot be cast to numeric float64: {e}"
                    )

        return aligned_df

    def predict(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        symbol: str = "TICKER",
        timestamp: Optional[str] = None,
    ) -> ForecastResult:
        """Generate 1-day and 5-day return predictions from a validated feature vector."""
        if self.model_1d is None or self.model_5d is None or self.preprocessor is None:
            raise RuntimeError("StockForecaster has not been loaded with valid models.")

        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()

        # Validate schema and order
        validated_df = self.validate_features(features_df)

        # Transform using train-fitted preprocessor
        X_scaled = self.preprocessor.transform(validated_df)

        # Predict
        pred_1d = float(self.model_1d.predict(X_scaled)[0])
        pred_5d = float(self.model_5d.predict(X_scaled)[0])

        features_dict = validated_df.iloc[0].to_dict()

        return ForecastResult(
            symbol=symbol,
            timestamp=timestamp or str(pd.Timestamp.now()),
            target_1d_pred_pct=round(pred_1d, 4),
            target_5d_pred_pct=round(pred_5d, 4),
            direction_1d="UP" if pred_1d >= 0 else "DOWN",
            direction_5d="UP" if pred_5d >= 0 else "DOWN",
            features_used=features_dict,
        )

    @classmethod
    def prepare_features_from_history(
        cls,
        history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute the exact feature vector for the latest observation t from a single stock's history.
        
        Args:
            history_df: DataFrame containing at least ['date', 'open', 'high', 'low', 'close', 'volume'].
                        Must have at least 25 bars (recommended 200+ bars for accurate EMA 200).
        
        Returns:
            Single-row DataFrame with all FEATURE_NAMES computed strictly up to t.
        """
        if len(history_df) < 20:
            raise ValueError(
                f"Insufficient historical bars: got {len(history_df)}, need at least 20 bars to compute technical indicators."
            )

        df = history_df.copy()
        col_map = {c: c.lower().strip() for c in df.columns}
        df = df.rename(columns=col_map)

        for req in ["open", "high", "low", "close", "volume"]:
            if req not in df.columns:
                raise ValueError(f"Missing required OHLCV column: '{req}'")

        # Ensure sorted by date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(by="date").reset_index(drop=True)

        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 1. Backward returns
        ret_1d = float((close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0)
        ret_2d = float((close.iloc[-1] / close.iloc[-3] - 1.0) * 100.0) if len(df) >= 3 else ret_1d
        ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0) if len(df) >= 6 else ret_1d
        ret_10d = float((close.iloc[-1] / close.iloc[-11] - 1.0) * 100.0) if len(df) >= 11 else ret_1d
        ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0) if len(df) >= 21 else ret_1d
        open_to_close_ret = float((close.iloc[-1] / open_.iloc[-1] - 1.0) * 100.0)

        # 2. Moving averages
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()

        current_close = float(close.iloc[-1])
        ema_ratio_9 = float((current_close / ema_9.iloc[-1] - 1.0) * 100.0)
        ema_ratio_21 = float((current_close / ema_21.iloc[-1] - 1.0) * 100.0)
        ema_ratio_50 = float((current_close / ema_50.iloc[-1] - 1.0) * 100.0)
        ema_ratio_200 = float((current_close / ema_200.iloc[-1] - 1.0) * 100.0)
        trend_ema_50_200 = float((ema_50.iloc[-1] / ema_200.iloc[-1] - 1.0) * 100.0)

        # 3. RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain.iloc[-1] / (avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else np.nan)
        rsi_14 = float(100.0 - (100.0 / (1.0 + rs))) if not np.isnan(rs) else 50.0

        # 4. MACD
        fast_ema = close.ewm(span=12, adjust=False).mean()
        slow_ema = close.ewm(span=26, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_sig
        macd_line_pct = float((macd_line.iloc[-1] / current_close) * 100.0)
        macd_signal_pct = float((macd_sig.iloc[-1] / current_close) * 100.0)
        macd_hist_pct = float((macd_hist.iloc[-1] / current_close) * 100.0)

        # 5. Volatility (20-day annualized)
        log_ret = np.log(close / close.shift(1))
        volatility_20d = float(log_ret.iloc[-20:].std() * np.sqrt(252) * 100.0) if len(df) >= 21 else 25.0

        # 6. Normalized ATR 14
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=14).mean().iloc[-1]
        natr_14 = float((atr_14 / current_close) * 100.0) if not np.isnan(atr_14) else 1.5

        # 7. Volume
        vol_5 = volume.iloc[-5:].mean() if len(df) >= 5 else float(volume.iloc[-1])
        vol_20 = volume.iloc[-20:].mean() if len(df) >= 20 else vol_5
        volume_ratio_5_20 = float(vol_5 / vol_20) if vol_20 > 0 else 1.0
        log_volume = float(np.log1p(volume.iloc[-1]))

        # 8. Range Position
        high_20 = high.iloc[-20:].max()
        low_20 = low.iloc[-20:].min()
        range_pos_20d = float(((current_close - low_20) / (high_20 - low_20))) if high_20 > low_20 else 0.5
        range_pos_20d = max(0.0, min(1.0, range_pos_20d))

        high_52w = high.iloc[-252:].max()
        low_52w = low.iloc[-252:].min()
        range_pos_52w = float(((current_close - low_52w) / (high_52w - low_52w))) if high_52w > low_52w else 0.5
        range_pos_52w = max(0.0, min(1.0, range_pos_52w))

        feature_row = {
            "ret_1d": ret_1d,
            "ret_2d": ret_2d,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "ret_20d": ret_20d,
            "open_to_close_ret": open_to_close_ret,
            "ema_ratio_9": ema_ratio_9,
            "ema_ratio_21": ema_ratio_21,
            "ema_ratio_50": ema_ratio_50,
            "ema_ratio_200": ema_ratio_200,
            "trend_ema_50_200": trend_ema_50_200,
            "rsi_14": rsi_14,
            "macd_line_pct": macd_line_pct,
            "macd_signal_pct": macd_signal_pct,
            "macd_hist_pct": macd_hist_pct,
            "volatility_20d": volatility_20d,
            "natr_14": natr_14,
            "volume_ratio_5_20": volume_ratio_5_20,
            "log_volume": log_volume,
            "range_pos_20d": range_pos_20d,
            "range_pos_52w": range_pos_52w,
        }
        return pd.DataFrame([feature_row])
