"""Unified workout cache backed by Supabase video_workout_cache table."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TABLE = "video_workout_cache"


def _get_supabase_client():
    """Return Supabase client or None if not configured."""
    try:
        from supabase import create_client
    except ImportError:
        return None
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        logger.error("Failed to create Supabase client: %s", e)
        return None


class UnifiedCacheService:
    """Static methods for reading/writing the unified workout cache."""

    @staticmethod
    def get(video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        """Return cached workout_data or None on miss."""
        client = _get_supabase_client()
        if not client:
            return None
        try:
            result = (
                client.table(_TABLE)
                .select("workout_data")
                .eq("video_id", video_id)
                .eq("platform", platform)
                .single()
                .execute()
            )
            if result.data:
                logger.info("Cache HIT: %s/%s", platform, video_id)
                return result.data["workout_data"]
            return None
        except Exception as e:
            if "no rows" in str(e).lower():
                return None
            logger.error("Cache read error for %s/%s: %s", platform, video_id, e)
            return None

    @staticmethod
    def save(video_id: str, platform: str, workout_data: Dict[str, Any]) -> None:
        """Insert into cache, silently ignore duplicates."""
        client = _get_supabase_client()
        if not client:
            return
        try:
            client.table(_TABLE).insert({
                "video_id": video_id,
                "platform": platform,
                "workout_data": workout_data,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception as e:
            if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                logger.debug("Already cached: %s/%s", platform, video_id)
            else:
                logger.error("Cache save error for %s/%s: %s", platform, video_id, e)
