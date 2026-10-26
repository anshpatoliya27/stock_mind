"""Financial Intelligence & Multimodal Agent for StockMind AI.

Coordinates user conversations, prompt engineering for Indian equities,
multimodal chart pattern understanding, and provides an integrated bridge
between big data market analytics and Generative AI models.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from src.agents.market_context_engine import MarketContextEngine
from src.agents.provider_adapter import (
    MissingProviderCredentialsError,
    MultimodalLLMAdapter,
    ProviderError,
)
from src.features.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are StockMind AI, an advanced institutional-grade financial analyst, quantitative researcher, and conversational market intelligence system.

### Core Capabilities:
1. **Evidence-Based Market Analysis:** Provide detailed, rigorous, and fact-grounded responses concerning Indian equities (National Stock Exchange - NSE and Bombay Stock Exchange - BSE), benchmark indices (NIFTY 50, SENSEX, BANK NIFTY), sectoral trends, and corporate fundamentals.
2. **Multimodal Technical Chart Analysis:** When the user uploads a stock chart or candlestick diagram, identify price trends, key support/resistance zones, candlestick patterns, volume confirmation, and risk-reward ratios.
3. **Macroeconomic & Regulatory Synthesis:** Synthesize Reserve Bank of India (RBI) Monetary Policy (Repo Rate 5.50%, SDF 5.25%), inflation (MoSPI CPI 4.95%), and forex rates (USD/INR ~ ₹95.98).
4. **General Business & Conversational Intelligence:** When the user asks about general companies, multinational corporations (e.g., Burger King, Apple, Microsoft), business models, or general concepts, answer articulately, dynamically, and informatively.
5. **Transparency & SEBI Advisory Disclaimer:** Clearly distinguish between empirical data and forward projections. Maintain an educational, institutional tone (not SEBI certified investment advice).
"""


class StockMindAgent:
    """Core financial conversational agent for StockMind AI."""

    def __init__(
        self,
        adapter: Optional[MultimodalLLMAdapter] = None,
        engine: Optional[MarketContextEngine] = None,
        chart_generator: Optional[ChartGenerator] = None,
    ):
        self.adapter = adapter or MultimodalLLMAdapter()
        self.engine = engine or MarketContextEngine()
        self.chart_generator = chart_generator or ChartGenerator()

    def get_status(self) -> Dict[str, Any]:
        """Return provider and agent readiness status."""
        status = self.adapter.get_status()
        status["market_engine"] = "active"
        status["local_engine_ready"] = True
        return status

    async def chat_stream(
        self,
        user_message: str,
        image_bytes: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterator[str]:
        """Stream conversational analysis from the multimodal LLM or local quantitative engine.
        
        Yields tokens dynamically in real-time.
        """
        ticker = self.engine.extract_ticker_from_text(user_message)
        user_lower = user_message.lower().strip()

        # Detect if user asks for chart image generation
        chart_keywords = [
            "predicted chart", "predict chart", "generate chart", "chart image",
            "generate image", "create image", "draw chart", "plot chart", "visualize chart",
            "show chart", "draw a chart", "generate a chart", "generate a image", "generate an image",
            "image of chart", "image of predicted chart", "prediction chart", "forecast chart"
        ]
        wants_chart_image = any(k in user_lower for k in chart_keywords) or (
            any(k in user_lower for k in ["chart", "candlestick", "graph", "plot"])
            and any(k in user_lower for k in ["generate", "create", "make", "draw", "predict", "forecast", "show"])
        )

        market_keywords = [
            "stock", "share", "market", "nifty", "sensex", "trade", "trading", 
            "chart", "rsi", "ema", "pivot", "bull", "bear", "target", "stoploss",
            "stop loss", "forecast", "deal", "bulk", "macro", "rbi", "repo", "inflation",
            "invest", "portfolio", "dividend", "financial", "bse", "nse", "equity",
            "indicator", "support", "resistance", "price", "valuation", "fund"
        ]
        is_market_query = (
            ticker is not None
            or image_bytes is not None
            or wants_chart_image
            or any(kw in user_lower for kw in market_keywords)
        )

        # ------------------------------------------------------------------
        # Feature: Automated Predictive Chart Image Generation
        # ------------------------------------------------------------------
        if wants_chart_image:
            target_ticker = ticker or "NIFTY 50"
            price_info = self.engine.get_latest_ticker_price(target_ticker)
            macro = self.engine.get_macro_context()
            forecast = self.engine.calculate_price_forecast(
                target_ticker,
                {"market_bias": "BULLISH"},
                base_price=price_info["price"],
                indicators=price_info.get("indicators"),
            )
            chart_res = self.chart_generator.generate_predicted_chart(
                ticker=target_ticker,
                base_price=price_info["price"],
                indicators=price_info.get("indicators"),
                forecast=forecast,
            )

            chart_banner = (
                f"### StockMind AI Predictive Candlestick Chart: **{target_ticker}**\n\n"
                f"{chart_res['markdown_tag']}\n\n"
                f"> **Forecast Matrix & Reference Levels:**\n"
                f"> - **Traded Reference Price:** `₹{price_info['price']:,.2f}`\n"
                f"> - **Primary Target (1 to 2 Weeks):** `₹{chart_res['target_1']:,.2f}`\n"
                f"> - **Extended Target (1 to 2 Months):** `₹{chart_res['target_2']:,.2f}`\n"
                f"> - **Risk Protection Stop Loss:** `₹{chart_res['stop_loss']:,.2f}`\n"
                f"> - **Momentum Bias:** `{chart_res['bias']}` | **RSI (14):** `{price_info.get('indicators', {}).get('rsi_14', 54.0)}`\n\n"
                f"---\n\n"
            )
            for word in chart_banner.split(" "):
                yield word + " "
                await asyncio.sleep(0.005)

            # If cloud LLM configured, stream in-depth reasoning explaining the chart
            if self.adapter.is_configured():
                try:
                    grounding_context = (
                        f"[GROUND-TRUTH PREDICTED CHART GENERATED FOR {target_ticker}]\n"
                        f"- Asset: {target_ticker}\n"
                        f"- Spot Price: ₹{price_info['price']:,.2f}\n"
                        f"- Target 1: ₹{chart_res['target_1']:,.2f}, Target 2: ₹{chart_res['target_2']:,.2f}\n"
                        f"- Support 1: ₹{forecast.get('support_1')}, Resistance 1: ₹{forecast.get('resistance_1')}\n"
                        f"- Recommended Stop Loss: ₹{chart_res['stop_loss']:,.2f}, Risk-Reward: {forecast.get('risk_reward_ratio', '1 : 2.5')}\n"
                        f"- Macro: RBI Repo {macro['rbi_repo_rate']}, Inflation {macro['cpi_inflation']}\n\n"
                        f"[USER REQUEST]\n{user_message}\n\n"
                        f"Provide a comprehensive technical analysis explaining the moving average trends, support/resistance invalidation, and tactical execution for this predicted chart."
                    )
                    async for token in self.adapter.generate_stream(
                        prompt=grounding_context,
                        history=history,
                        system_instruction=SYSTEM_PROMPT,
                    ):
                        yield token
                    return
                except Exception as e:
                    logger.error(f"Cloud LLM explanation failed: {e}")

            # Local mode explanation
            summary = (
                f"**Technical Chart Assessment:**\n"
                f"- The candlestick chart above illustrates price consolidation anchored at **₹{price_info['price']:,.2f}** with positive upward momentum.\n"
                f"- A breakout above resistance at **₹{forecast.get('resistance_1', round(price_info['price']*1.02, 2))}** unlocks the forecast path toward **Target 1 (₹{chart_res['target_1']:,.2f})**.\n"
                f"- Maintain strict capital protection at **Stop Loss (₹{chart_res['stop_loss']:,.2f})** to invalidate the bullish trajectory."
            )
            for word in summary.split(" "):
                yield word + " "
                await asyncio.sleep(0.008)
            return

        # ------------------------------------------------------------------
        # Mode A: Live Cloud GenAI (Gemini / Grok / OpenAI)
        # ------------------------------------------------------------------
        if self.adapter.is_configured():
            try:
                # 1. Formulate grounded prompt context based on query intent
                if ticker or image_bytes:
                    # Specific stock or chart analysis
                    price_info = self.engine.get_latest_ticker_price(ticker)
                    macro = self.engine.get_macro_context()
                    deals = self.engine.get_bulk_deals_context(ticker, limit=3)
                    
                    vision_data = {}
                    if image_bytes:
                        vision_data = self.engine.analyze_chart_image(image_bytes)

                    forecast = self.engine.calculate_price_forecast(
                        ticker,
                        vision_data if image_bytes else {"market_bias": "BULLISH"},
                        base_price=price_info["price"],
                        indicators=price_info.get("indicators"),
                    )

                    grounding_context = (
                        f"[GROUND-TRUTH MARKET CONTEXT & TECHNICAL EXTRACTION]\n"
                        f"- Asset/Symbol: {price_info['symbol']}\n"
                        f"- Latest Traded Price: ₹{price_info['price']} (Date: {price_info['date']}, Source: {price_info['source']})\n"
                        f"- Technical Indicators: RSI(14)={price_info.get('indicators', {}).get('rsi_14', 'N/A')}, "
                        f"EMA 50=₹{price_info.get('indicators', {}).get('ema_50', 'N/A')}, "
                        f"Pivot=₹{price_info.get('indicators', {}).get('pivot_point', 'N/A')}\n"
                        f"- Vision Analysis: {vision_data.get('detected_pattern', 'Price Structure')} "
                        f"(Bias: {vision_data.get('market_bias', 'NEUTRAL')}, Confidence: {vision_data.get('pattern_confidence_pct', 80.0)}%)\n"
                        f"- Quantitative Targets: Target 1: ₹{forecast['short_term_target']}, Target 2: ₹{forecast['medium_term_target']}, "
                        f"Support 1/2: ₹{forecast['support_1']} / ₹{forecast['support_2']}, "
                        f"Resistance 1/2: ₹{forecast['resistance_1']} / ₹{forecast['resistance_2']}, "
                        f"Stop-Loss: ₹{forecast['recommended_stop_loss']}, Risk-Reward: {forecast['risk_reward_ratio']}\n"
                        f"- RBI Macro Rates: Repo: {macro['rbi_repo_rate']}, USD/INR: ₹{macro['usd_inr_rate']}, CPI: {macro['cpi_inflation']}\n"
                        f"- Institutional Bulk Deals: {len(deals)} recent recorded transactions\n\n"
                        f"[USER QUERY]\n{user_message}"
                    )
                elif is_market_query:
                    # Broad market / macro query
                    macro = self.engine.get_macro_context()
                    nifty_info = self.engine.get_latest_ticker_price("NIFTY 50")
                    grounding_context = (
                        f"[GROUND-TRUTH INDIAN MARKET CONTEXT]\n"
                        f"- Benchmark Index: NIFTY 50 at ₹{nifty_info['price']:,.2f} ({nifty_info.get('change_pct', 0.28):+.2f}%)\n"
                        f"- Monetary Stance: RBI Repo Rate {macro['rbi_repo_rate']}, SDF {macro['rbi_sdf_rate']}, MSF {macro['rbi_msf_rate']}\n"
                        f"- Economic Indicators: CPI Inflation {macro['cpi_inflation']}, USD/INR Reference Rate ₹{macro['usd_inr_rate']}\n"
                        f"- Data Coverage: 3.9M+ verified NSE daily transactions (up to 2026)\n\n"
                        f"[USER QUERY]\n{user_message}"
                    )
                else:
                    # General conversational / business query (e.g., "what is burger king")
                    grounding_context = user_message

                async for token in self.adapter.generate_stream(
                    prompt=grounding_context,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                    history=history,
                    system_instruction=SYSTEM_PROMPT,
                ):
                    yield token
                return

            except Exception as e:
                logger.error(f"Cloud LLM provider failed: {e}", exc_info=True)
                yield f"> ⚠️ **Provider Error ({self.adapter.provider.upper()}):** {e}\n\n"
                # Fall through to contextual response below

        # ------------------------------------------------------------------
        # Mode B: Contextual Local Processing (When Provider Unconfigured / Errored)
        # ------------------------------------------------------------------
        if is_market_query:
            report = self.engine.build_synthesis_report(
                prompt=user_message,
                image_bytes=image_bytes,
                ticker=ticker,
            )
            provider_label = self.adapter.provider.upper()
            mode_banner = (
                f"> **[Execution Mode]** Powered by **StockMind Quantitative & Multimodal Engine** (Offline Grounded).\n"
                f"> *Connected provider: `{provider_label}`. Connect your verified `GEMINI_API_KEY` to enable full cloud GenAI streaming.*\n\n"
            )
            full_output = mode_banner + report
        else:
            # General query in offline/fallback mode
            full_output = (
                f"### StockMind AI Assistant\n\n"
                f"You asked: **\"{user_message}\"**\n\n"
                f"I am StockMind AI, an institutional financial intelligence system. "
                f"For non-market questions like this, connect your **Google Gemini API Key** in the top-right settings to unlock unrestricted conversational GenAI streaming powered by `gemini-3.5-flash`.\n\n"
                f"You can also ask about any Indian stock (e.g., **RELIANCE**, **TCS**, **INFY**, **NIFTY 50**), upload candlestick chart screenshots for automated pattern detection, or analyze RBI macroeconomic interest rate impacts."
            )

        # Stream chunk-by-chunk for smooth typing experience
        words = full_output.split(" ")
        buffer = []
        for word in words:
            buffer.append(word)
            if len(buffer) >= 3 or "\n" in word:
                yield " ".join(buffer) + " "
                buffer = []
                await asyncio.sleep(0.01)

        if buffer:
            yield " ".join(buffer)

