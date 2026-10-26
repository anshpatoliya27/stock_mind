#!/usr/bin/env python
"""StockMind AI Application Launcher.

Starts the FastAPI server hosting the StockMind AI web terminal and API backend.
Access the web application at: http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Ensure .env exists if .env.example is present
env_file = PROJECT_ROOT / ".env"
env_example = PROJECT_ROOT / ".env.example"
if not env_file.exists() and env_example.exists():
    shutil.copy(env_example, env_file)
    print(f"[*] Created initial .env configuration file from .env.example")

load_dotenv()

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    provider = os.environ.get("LLM_PROVIDER", "grok").upper()
    has_grok = bool(os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    print("\n" + "=" * 65)
    print("           STOCKMIND AI - MULTIMODAL FINANCIAL ADVISOR")
    print("=" * 65)
    print(f" Web Application URL : http://{host}:{port}")
    print(f" API Documentation   : http://{host}:{port}/docs")
    print(f" Configured Provider : {provider}")
    print(f" Grok (xAI) Key      : {'Yes' if has_grok else 'No (Add GROK_API_KEY in .env)'}")
    print(f" Gemini Key Present  : {'Yes' if has_gemini else 'No (Add GEMINI_API_KEY in .env)'}")
    print(f" OpenAI Key Present  : {'Yes' if has_openai else 'No (Add OPENAI_API_KEY in .env)'}")
    print("=" * 65)
    print("[*] Press Ctrl+C in terminal to stop server.\n")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=False,
    )
