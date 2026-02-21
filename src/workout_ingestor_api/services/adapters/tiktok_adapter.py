"""TikTok platform adapter — uses TikTokService (oEmbed + yt-dlp)."""
from __future__ import annotations

import logging
from .base import PlatformAdapter, MediaContent, PlatformFetchError
from . import register_adapter
from workout_ingestor_api.services.tiktok_service import TikTokService

logger = logging.getLogger(__name__)


class TikTokAdapter(PlatformAdapter):
    @staticmethod
    def platform_name() -> str:
        return "tiktok"

    def fetch(self, url: str, source_id: str) -> MediaContent:
        try:
            metadata = TikTokService.extract_metadata(url)
        except Exception as e:
            raise PlatformFetchError(f"TikTok fetch failed for {source_id}: {e}") from e

        # Build primary text from title + hashtags
        parts = [metadata.title] if metadata.title else []
        if metadata.hashtags:
            parts.append(" ".join(f"#{h}" for h in metadata.hashtags))
        primary_text = "\n".join(parts).strip()

        if not primary_text:
            raise PlatformFetchError(f"TikTok video {source_id} has no extractable text")

        title = (metadata.title or f"TikTok by @{metadata.author_name}")[:80]

        return MediaContent(
            primary_text=primary_text,
            title=title,
            media_metadata={
                "author": metadata.author_name,
                "video_id": source_id,
                "video_duration_sec": getattr(metadata, "duration_seconds", None),
            },
        )


register_adapter(TikTokAdapter)
