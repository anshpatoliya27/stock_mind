"""FastAPI Web Application and API Backend for StockMind AI."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.agents.financial_agent import StockMindAgent
from src.agents.provider_adapter import (
    MissingProviderCredentialsError,
    MultimodalLLMAdapter,
    ProviderError,
)
from src.api.conversation_store import ConversationStore
from src.features.market_screener import MarketScreener

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stockmind.api")

app = FastAPI(
    title="StockMind AI API",
    description="Big Data-Driven Generative AI Framework for Indian Stock Market Analysis",
    version="2.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
store = ConversationStore()
agent = StockMindAgent()
screener = MarketScreener()

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "src" / "frontend"
UPLOADS_DIR = PROJECT_ROOT / "artifacts" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_CHARTS_DIR = PROJECT_ROOT / "artifacts" / "generated_charts"
GENERATED_CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API Routes: Health, Configuration Status & Market Screener
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Basic health and liveness probe."""
    return {"status": "ok", "app": "StockMind AI"}


@app.get("/api/status")
async def get_system_status():
    """Return LLM provider status, dataset readiness, and platform metrics."""
    llm_status = agent.get_status()

    # Dataset status
    dataset_dir = PROJECT_ROOT / "Dataset"
    nse_daily_exists = (dataset_dir / "market_data" / "nse_daily").exists()
    minute_exists = (dataset_dir / "market_data" / "comapanies_minutes").exists()

    return {
        "status": "ready",
        "llm": llm_status,
        "dataset": {
            "root_exists": dataset_dir.exists(),
            "nse_daily_available": nse_daily_exists,
            "minute_ticks_available": minute_exists,
            "path": str(dataset_dir),
            "stats": {
                "daily_trades": "3,919,007 rows (2019–2026)",
                "companies": "5,028 ticker archives",
                "minute_ticks": "515 tick files",
                "chart_patterns": "340 labeled pattern images",
                "financial_news": "26,000 sentiment-classified articles",
            },
        },
    }


@app.get("/api/market/ticker_tape")
async def get_market_ticker_tape():
    """Return live benchmark indices and stock quotes for top header ticker bar."""
    return screener.get_ticker_tape()


@app.post("/api/settings")
async def update_settings(
    provider: str = Form("gemini"),
    api_key: str = Form(""),
    model: Optional[str] = Form(None),
):
    """Save API key and active provider in memory and persist into .env."""
    provider_clean = provider.lower().strip()
    key_clean = api_key.strip()

    # Update active adapter in agent
    agent.adapter.provider = provider_clean
    if key_clean:
        agent.adapter.custom_api_key = key_clean

    if model and model.strip():
        agent.adapter.model = model.strip()
    elif provider_clean in ["grok", "xai"]:
        agent.adapter.model = "grok-2-vision-1212"
    elif provider_clean == "gemini":
        agent.adapter.model = "gemini-3.5-flash"
    elif provider_clean == "openai":
        agent.adapter.model = "gpt-4o-mini"

    # Persist to .env
    env_path = PROJECT_ROOT / ".env"
    env_content = ""
    if env_path.exists():
        try:
            env_content = env_path.read_text(encoding="utf-8")
        except Exception:
            pass

    def set_env_var(content: str, key: str, val: str) -> str:
        pattern = rf"^{key}=.*$"
        if re.search(pattern, content, flags=re.MULTILINE):
            return re.sub(pattern, f"{key}={val}", content, flags=re.MULTILINE)
        return content.rstrip() + f"\n{key}={val}\n"

    env_content = set_env_var(env_content, "LLM_PROVIDER", provider_clean)
    env_content = set_env_var(env_content, "LLM_MODEL", agent.adapter.model)

    if key_clean:
        if provider_clean in ["grok", "xai"]:
            env_content = set_env_var(env_content, "GROK_API_KEY", key_clean)
            os.environ["GROK_API_KEY"] = key_clean
        elif provider_clean == "gemini":
            env_content = set_env_var(env_content, "GEMINI_API_KEY", key_clean)
            os.environ["GEMINI_API_KEY"] = key_clean
        elif provider_clean == "openai":
            env_content = set_env_var(env_content, "OPENAI_API_KEY", key_clean)
            os.environ["OPENAI_API_KEY"] = key_clean

    try:
        env_path.write_text(env_content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to persist settings to .env: {e}")

    return {
        "status": "success",
        "provider": agent.adapter.provider,
        "model": agent.adapter.model,
        "is_configured": agent.adapter.is_configured(),
    }


# ---------------------------------------------------------------------------
# API Routes: Conversation Management
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
async def list_conversations():
    """List all conversations ordered by recent activity."""
    return store.list_conversations()


@app.post("/api/conversations")
async def create_conversation(title: Optional[str] = Form("New Analysis")):
    """Create a new conversation session."""
    return store.create_conversation(title=title or "New Analysis")


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Retrieve full conversation history."""
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation."""
    success = store.delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conv_id}


# ---------------------------------------------------------------------------
# API Routes: Uploads Serving
# ---------------------------------------------------------------------------

@app.get("/api/uploads/{filename}")
async def get_uploaded_image(filename: str):
    """Serve an uploaded chart image."""
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@app.get("/api/generated/{filename}")
async def get_generated_chart(filename: str):
    """Serve a generated predictive stock chart image."""
    file_path = GENERATED_CHARTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Generated chart image not found")
    return FileResponse(file_path, media_type="image/png")


# ---------------------------------------------------------------------------
# API Routes: Streaming Multimodal Chat (SSE)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """Process user financial query and stream back response via Server-Sent Events (SSE)."""
    # 1. Resolve or create conversation
    if not conversation_id or not store.get_conversation(conversation_id):
        conv = store.create_conversation(title="Stock Intelligence Session")
        conversation_id = conv["id"]

    # 2. Process optional chart image attachment
    image_bytes = None
    image_mime = None
    image_url = None

    if image and image.filename:
        image_bytes = await image.read()
        image_mime = image.content_type or "image/png"
        image_url, _ = store.save_upload_image(image_bytes, image.filename)

    # 3. Record user message in history
    store.add_message(
        conv_id=conversation_id,
        role="user",
        content=message,
        image_url=image_url,
    )

    # 4. Fetch conversation history for multi-turn context
    conv_data = store.get_conversation(conversation_id)
    raw_history = conv_data.get("messages", [])[:-1] if conv_data else []
    llm_history = [
        {"role": m["role"], "content": m["content"]}
        for m in raw_history[-6:]
    ]

    # 5. Build streaming response generator
    async def sse_generator() -> AsyncIterator[str]:
        init_event = {
            "type": "init",
            "conversation_id": conversation_id,
            "image_url": image_url,
        }
        yield f"data: {json.dumps(init_event)}\n\n"

        full_response_text = []

        try:
            async for token in agent.chat_stream(
                user_message=message,
                image_bytes=image_bytes,
                image_mime_type=image_mime,
                history=llm_history,
            ):
                full_response_text.append(token)
                event_data = {"type": "token", "token": token}
                yield f"data: {json.dumps(event_data)}\n\n"

            # Save completed response to conversation store
            complete_text = "".join(full_response_text)
            store.add_message(
                conv_id=conversation_id,
                role="assistant",
                content=complete_text,
            )

            done_event = {"type": "done", "conversation_id": conversation_id}
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            logger.error(f"Chat generation error: {e}")
            error_event = {
                "type": "error",
                "error_code": "SYSTEM_ERROR",
                "message": f"Stock analysis service encountered an error: {str(e)}",
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Mount Static Frontend
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        """Serve the StockMind AI web user interface."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>StockMind AI Frontend Initializing...</h1>")
