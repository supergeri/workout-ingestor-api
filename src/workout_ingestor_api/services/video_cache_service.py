"""
Multi-platform video workout cache service.

Handles caching of video workout metadata for YouTube, Instagram, and TikTok
to avoid redundant API calls and processing for previously ingested videos.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from workout_ingestor_api.services.video_url_normalizer import Platform

logger = logging.getLogger(__name__)


def get_supabase_client():
    """Get Supabase client instance."""
    try:
        from supabase import create_client, Client
    except ImportError:
        logger.warning("Supabase library not installed. Caching will be disabled.")
        return None

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not configured. Video caching will be disabled.")
        return None

    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None


class VideoCacheService:
    """
    Service for caching video workout metadata across platforms.

    Supports YouTube, Instagram, and TikTok videos.
    """

    TABLE_NAME = "video_workout_cache"

    @staticmethod
    def get_cached_video(video_id: str, platform: Platform) -> Optional[Dict[str, Any]]:
        """
        Get a cached video workout by video ID and platform.

        Args:
            video_id: Platform-specific video ID
            platform: Video platform (youtube, instagram, tiktok)

        Returns:
            Cached video data or None if not found
        """
        supabase = get_supabase_client()
        if not supabase:
            return None

        try:
            result = supabase.table(VideoCacheService.TABLE_NAME) \
                .select("*") \
                .eq("video_id", video_id) \
                .eq("platform", platform) \
                .single() \
                .execute()

            if result.data:
                logger.info(f"Cache HIT for {platform} video_id: {video_id}")
                return result.data
            return None
        except Exception as e:
            if "no rows" in str(e).lower() or "0 rows" in str(e).lower():
                logger.info(f"Cache MISS for {platform} video_id: {video_id}")
                return None
            logger.error(f"Error fetching cached video for {platform}/{video_id}: {e}")
            return None

    @staticmethod
    def get_cached_video_by_url(normalized_url: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached video workout by normalized URL.

        Args:
            normalized_url: Normalized video URL

        Returns:
            Cached video data or None if not found
        """
        supabase = get_supabase_client()
        if not supabase:
            return None

        try:
            result = supabase.table(VideoCacheService.TABLE_NAME) \
                .select("*") \
                .eq("normalized_url", normalized_url) \
                .single() \
                .execute()

            if result.data:
                logger.info(f"Cache HIT for normalized_url: {normalized_url}")
                return result.data
            return None
        except Exception as e:
            if "no rows" in str(e).lower() or "0 rows" in str(e).lower():
                logger.info(f"Cache MISS for normalized_url: {normalized_url}")
                return None
            logger.error(f"Error fetching cached video for url {normalized_url}: {e}")
            return None

    @staticmethod
    def save_cached_video(
        video_id: str,
        platform: Platform,
        source_url: str,
        normalized_url: str,
        oembed_data: Optional[Dict[str, Any]] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
        workout_data: Optional[Dict[str, Any]] = None,
        processing_method: Optional[str] = None,
        ingested_by: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Save a video workout to the cache.

        Args:
            video_id: Platform-specific video ID
            platform: Video platform (youtube, instagram, tiktok)
            source_url: Original URL submitted by user
            normalized_url: Normalized URL for lookups
            oembed_data: oEmbed response data (thumbnail, title, author, etc.)
            video_metadata: Additional video metadata
            workout_data: Extracted workout structure (if any)
            processing_method: How the workout was processed
            ingested_by: Optional user ID who first ingested this video

        Returns:
            Saved cache entry or None if failed
        """
        supabase = get_supabase_client()
        if not supabase:
            return None

        try:
            now = datetime.now(timezone.utc).isoformat()

            data = {
                "video_id": video_id,
                "platform": platform,
                "source_url": source_url,
                "normalized_url": normalized_url,
                "oembed_data": oembed_data or {},
                "video_metadata": video_metadata or {},
                "workout_data": workout_data or {},
                "processing_method": processing_method,
                "ingested_at": now,
                "ingested_by": ingested_by,
            }

            result = supabase.table(VideoCacheService.TABLE_NAME) \
                .insert(data) \
                .execute()

            if result.data and len(result.data) > 0:
                logger.info(f"Cached video saved for {platform}/{video_id}")
                return result.data[0]
            return None
        except Exception as e:
            error_str = str(e)
            # Handle duplicate key violation (video already cached)
            if "duplicate" in error_str.lower() or "unique" in error_str.lower():
                logger.info(f"Video already cached for {platform}/{video_id}")
                return VideoCacheService.get_cached_video(video_id, platform)
            logger.error(f"Failed to cache video for {platform}/{video_id}: {e}")
            return None

    @staticmethod
    def update_cached_video(
        video_id: str,
        platform: Platform,
        workout_data: Optional[Dict[str, Any]] = None,
        oembed_data: Optional[Dict[str, Any]] = None,
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing cached video.

        Args:
            video_id: Platform-specific video ID
            platform: Video platform
            workout_data: Updated workout structure (optional)
            oembed_data: Updated oEmbed data (optional)
            video_metadata: Updated video metadata (optional)

        Returns:
            True if update successful, False otherwise
        """
        supabase = get_supabase_client()
        if not supabase:
            return False

        try:
            update_data = {
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            if workout_data is not None:
                update_data["workout_data"] = workout_data
            if oembed_data is not None:
                update_data["oembed_data"] = oembed_data
            if video_metadata is not None:
                update_data["video_metadata"] = video_metadata

            result = supabase.table(VideoCacheService.TABLE_NAME) \
                .update(update_data) \
                .eq("video_id", video_id) \
                .eq("platform", platform) \
                .execute()

            return result.data is not None and len(result.data) > 0
        except Exception as e:
            logger.error(f"Failed to update cached video for {platform}/{video_id}: {e}")
            return False

    @staticmethod
    def increment_cache_hit(video_id: str, platform: Platform) -> bool:
        """
        Increment the cache hit counter for a video.

        Args:
            video_id: Platform-specific video ID
            platform: Video platform

        Returns:
            True if successful, False otherwise
        """
        supabase = get_supabase_client()
        if not supabase:
            return False

        try:
            result = supabase.table(VideoCacheService.TABLE_NAME) \
                .select("cache_hits") \
                .eq("video_id", video_id) \
                .eq("platform", platform) \
                .single() \
                .execute()

            if result.data:
                current_hits = result.data.get("cache_hits", 0) or 0
                supabase.table(VideoCacheService.TABLE_NAME) \
                    .update({
                        "cache_hits": current_hits + 1,
                        "last_accessed_at": datetime.now(timezone.utc).isoformat()
                    }) \
                    .eq("video_id", video_id) \
                    .eq("platform", platform) \
                    .execute()
                return True
            return False
        except Exception as e:
            logger.debug(f"Failed to increment cache hit for {platform}/{video_id}: {e}")
            return False

    @staticmethod
    def delete_cached_video(video_id: str, platform: Platform) -> bool:
        """
        Delete a cached video by video ID and platform.

        Args:
            video_id: Platform-specific video ID
            platform: Video platform

        Returns:
            True if successful, False otherwise
        """
        supabase = get_supabase_client()
        if not supabase:
            return False

        try:
            result = supabase.table(VideoCacheService.TABLE_NAME) \
                .delete() \
                .eq("video_id", video_id) \
                .eq("platform", platform) \
                .execute()

            deleted = result.data is not None and len(result.data) > 0
            if deleted:
                logger.info(f"Deleted cached video for {platform}/{video_id}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete cached video for {platform}/{video_id}: {e}")
            return False

    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """
        Get cache statistics by platform.

        Returns:
            Dict with total_cached, total_hits, breakdown by platform
        """
        supabase = get_supabase_client()
        if not supabase:
            return {"error": "Supabase not configured"}

        try:
            # Get all records for stats
            result = supabase.table(VideoCacheService.TABLE_NAME) \
                .select("platform, cache_hits") \
                .execute()

            if not result.data:
                return {
                    "total_cached": 0,
                    "total_cache_hits": 0,
                    "by_platform": {}
                }

            # Calculate stats
            by_platform: Dict[str, Dict[str, int]] = {}
            total_cached = 0
            total_hits = 0

            for row in result.data:
                platform = row.get("platform", "unknown")
                hits = row.get("cache_hits", 0) or 0

                if platform not in by_platform:
                    by_platform[platform] = {"count": 0, "hits": 0}

                by_platform[platform]["count"] += 1
                by_platform[platform]["hits"] += hits
                total_cached += 1
                total_hits += hits

            return {
                "total_cached": total_cached,
                "total_cache_hits": total_hits,
                "by_platform": by_platform
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    @staticmethod
    def search_cached_videos(
        platform: Optional[Platform] = None,
        search_query: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search cached videos with optional filters.

        Args:
            platform: Filter by platform
            search_query: Search in title/author
            limit: Max results to return

        Returns:
            List of cached video records
        """
        supabase = get_supabase_client()
        if not supabase:
            return []

        try:
            query = supabase.table(VideoCacheService.TABLE_NAME) \
                .select("*") \
                .order("cache_hits", desc=True) \
                .limit(limit)

            if platform:
                query = query.eq("platform", platform)

            # Note: Full-text search would require additional setup
            # For now, we just return filtered by platform

            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to search cached videos: {e}")
            return []
