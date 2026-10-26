"""Persistent conversation and session management for StockMind AI."""

from __future__ import annotations

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConversationStore:
    """Manages chat sessions, messages, and uploaded chart images."""

    def __init__(self, storage_file: Optional[Union[str, Path]] = None, uploads_dir: Optional[Union[str, Path]] = None):
        self.storage_file = Path(storage_file or "artifacts/conversations.json")
        self.uploads_dir = Path(uploads_dir or "artifacts/uploads")
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._conversations: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(self._conversations, f, indent=2)
        except Exception as e:
            print(f"[!] Warning: Failed to persist conversations: {e}")

    def create_conversation(self, title: str = "New Conversation") -> Dict[str, Any]:
        """Create a new chat conversation."""
        conv_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        conv = {
            "id": conv_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._conversations[conv_id] = conv
        self._save()
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation by ID."""
        return self._conversations.get(conv_id)

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations ordered by recent activity."""
        convs = list(self._conversations.values())
        convs.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return [
            {
                "id": c["id"],
                "title": c.get("title", "Untitled Chat"),
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
                "message_count": len(c.get("messages", [])),
            }
            for c in convs
        ]

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        image_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a message to an existing conversation."""
        if conv_id not in self._conversations:
            self.create_conversation(title="Financial Research")
            conv = self._conversations[conv_id]
        else:
            conv = self._conversations[conv_id]

        msg_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        msg = {
            "id": msg_id,
            "role": role,
            "content": content,
            "image_url": image_url,
            "timestamp": now,
        }
        conv["messages"].append(msg)
        conv["updated_at"] = now

        # Auto-update conversation title from first user query
        if role == "user" and len(conv["messages"]) <= 2:
            snippet = content.strip().split("\n")[0][:45]
            if snippet:
                conv["title"] = snippet + ("..." if len(content) > 45 else "")

        self._save()
        return msg

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete conversation and its associated metadata."""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            self._save()
            return True
        return False

    def save_upload_image(self, file_bytes: bytes, original_filename: str) -> Tuple[str, str]:
        """Save an uploaded chart image and return its relative URL and filepath."""
        ext = Path(original_filename).suffix or ".png"
        unique_name = f"{uuid.uuid4()}{ext}"
        target_path = self.uploads_dir / unique_name
        target_path.write_bytes(file_bytes)
        relative_url = f"/api/uploads/{unique_name}"
        return relative_url, str(target_path)
