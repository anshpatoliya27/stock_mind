"""Multimodal LLM Provider Adapter for StockMind AI.

Provides a unified interface for Gemini, Grok (xAI), and OpenAI providers, supporting:
- Text-only conversation and financial reasoning
- Stock-chart image uploads and multimodal visual pattern analysis
- Server-Sent Events (SSE) streaming token delivery
- Strict missing-credential and connection error handling (no fake responses)
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from dotenv import load_dotenv

# Automatically load .env from project root if present
load_dotenv(override=True)


class ProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class MissingProviderCredentialsError(ProviderError):
    """Raised when the selected LLM provider lacks an API key."""
    pass


class MultimodalLLMAdapter:
    """Configurable adapter connecting StockMind AI to real multimodal LLMs."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower().strip()
        
        default_model = "gemini-3.5-flash"
        if self.provider in ["grok", "xai"]:
            default_model = "grok-2-vision-1212"
        elif self.provider == "openai":
            default_model = "gpt-4o-mini"

        if model:
            self.model = model
        else:
            env_provider = os.environ.get("LLM_PROVIDER", "gemini").lower().strip()
            env_model = os.environ.get("LLM_MODEL")
            if env_model and (provider is None or provider.lower().strip() == env_provider):
                self.model = env_model
            else:
                self.model = default_model

        self.custom_api_key = api_key

    def get_api_key(self) -> Optional[str]:
        """Retrieve active API key for the selected provider."""
        if self.custom_api_key is not None:
            return self.custom_api_key

        if self.provider == "gemini":
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif self.provider in ["grok", "xai"]:
            return os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        elif self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None

    def is_configured(self) -> bool:
        """Check if active provider credentials are valid and present."""
        key = self.get_api_key()
        return bool(key and len(key.strip()) > 5)

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status of the LLM provider adapter."""
        key = self.get_api_key()
        key_name = (
            "GROK_API_KEY"
            if self.provider in ["grok", "xai"]
            else "GEMINI_API_KEY"
            if self.provider == "gemini"
            else "OPENAI_API_KEY"
        )
        return {
            "provider": self.provider,
            "model": self.model,
            "is_configured": bool(key and len(key.strip()) > 5),
            "key_preview": f"{key[:4]}...{key[-4:]}" if (key and len(key) > 8) else None,
            "supported_providers": ["gemini", "grok", "openai"],
            "setup_hint": (
                None
                if self.is_configured()
                else f"Add {key_name} to your .env file or environment variables."
            ),
        }

    async def generate_stream(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens from the multimodal LLM provider.
        
        Yields:
            Token chunks as strings.
            
        Raises:
            MissingProviderCredentialsError: If API key is not configured.
            ProviderError: If remote LLM call fails.
        """
        if not self.is_configured():
            key_name = (
                "GROK_API_KEY"
                if self.provider in ["grok", "xai"]
                else "GEMINI_API_KEY"
                if self.provider == "gemini"
                else "OPENAI_API_KEY"
            )
            raise MissingProviderCredentialsError(
                f"StockMind AI requires a valid {key_name} to generate responses with {self.provider.upper()}. "
                f"Please add your API key to your `.env` file or export `{key_name}` in your environment."
            )

        if self.provider == "gemini":
            async for token in self._stream_gemini(
                prompt=prompt,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                history=history,
                system_instruction=system_instruction,
            ):
                yield token

        elif self.provider in ["grok", "xai"]:
            # xAI Grok API uses standard OpenAI compatibility with base_url
            base_url = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")
            async for token in self._stream_openai_compatible(
                prompt=prompt,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                history=history,
                system_instruction=system_instruction,
                base_url=base_url,
                provider_label="Grok",
            ):
                yield token

        elif self.provider == "openai":
            async for token in self._stream_openai_compatible(
                prompt=prompt,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                history=history,
                system_instruction=system_instruction,
                base_url=None,
                provider_label="OpenAI",
            ):
                yield token

        else:
            raise ProviderError(f"Unsupported LLM provider: {self.provider}")

    async def _stream_gemini(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response using Google Gemini."""
        try:
            import google.generativeai as genai
            from PIL import Image
        except ImportError as e:
            raise ProviderError(f"Missing Google Generative AI dependency: {e}")

        api_key = self.get_api_key()
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "max_output_tokens": 4096,
        }

        # Build contents list
        contents: List[Any] = []

        # Convert historical messages
        if history:
            for msg in history:
                role = "user" if msg.get("role") in ["user", "human"] else "model"
                contents.append({"role": role, "parts": [msg.get("content", "")]})

        # Assemble current turn parts
        current_parts: List[Any] = []
        if image_bytes:
            try:
                img = Image.open(io.BytesIO(image_bytes))
                current_parts.append(img)
            except Exception as e:
                raise ProviderError(f"Invalid image uploaded: {e}")

        current_parts.append(prompt)
        contents.append({"role": "user", "parts": current_parts})

        # Map deprecated or sunset model IDs to active 2026 Gemini models
        model_name = self.model
        deprecated_aliases = {
            "gemini-1.5-flash": "gemini-3.5-flash",
            "gemini-1.5-pro": "gemini-3.5-flash",
            "gemini-2.5-flash": "gemini-3.5-flash",
            "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
            "gemini-pro": "gemini-flash-lite-latest",
        }
        active_model = deprecated_aliases.get(model_name, model_name)

        model_kwargs: Dict[str, Any] = {"generation_config": generation_config}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        try:
            model_client = genai.GenerativeModel(active_model, **model_kwargs)
            response = model_client.generate_content(contents, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as e:
            err_msg = str(e)
            # If model was 404 or unavailable, attempt fallback to gemini-flash-lite-latest
            if "404" in err_msg and active_model != "gemini-flash-lite-latest":
                try:
                    fallback_client = genai.GenerativeModel("gemini-flash-lite-latest", **model_kwargs)
                    response = fallback_client.generate_content(contents, stream=True)
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                    return
                except Exception:
                    pass

            if "API_KEY_INVALID" in err_msg or "expired" in err_msg.lower() or "not valid" in err_msg.lower():
                raise MissingProviderCredentialsError(f"Invalid Gemini API Key: {err_msg}")
            raise ProviderError(f"Gemini API Error: {err_msg}")

    async def _stream_openai_compatible(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_label: str = "OpenAI",
    ) -> AsyncIterator[str]:
        """Stream response using OpenAI or OpenAI-compatible (Grok / xAI) chat completions."""
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ProviderError(f"Missing OpenAI dependency: {e}")

        client_kwargs: Dict[str, Any] = {"api_key": self.get_api_key()}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = AsyncOpenAI(**client_kwargs)
        messages: List[Dict[str, Any]] = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if history:
            for msg in history:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        # Format current user message with optional image
        if image_bytes:
            mime = image_mime_type or "image/png"
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            image_url = f"data:{mime};base64,{b64_img}"
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.3,
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as e:
            err_msg = str(e)
            if "invalid_api_key" in err_msg.lower() or "authentication" in err_msg.lower() or "401" in err_msg or "incorrect api key" in err_msg.lower():
                raise MissingProviderCredentialsError(f"Invalid {provider_label} API Key: {err_msg}")
            raise ProviderError(f"{provider_label} API Error: {err_msg}")
