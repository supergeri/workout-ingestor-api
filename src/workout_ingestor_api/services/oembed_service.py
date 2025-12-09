"""
oEmbed service for fetching video metadata from various platforms.

Supports:
- Instagram oEmbed API
- YouTube oEmbed API
- TikTok oEmbed API
"""

import logging
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import quote

import requests

from workout_ingestor_api.services.video_url_normalizer import (
    Platform,
    extract_instagram_shortcode,
    extract_youtube_video_id,
    extract_tiktok_video_id,
    get_full_url,
)

logger = logging.getLogger(__name__)


@dataclass
class OEmbedResponse:
    """Standardized oEmbed response across platforms."""
    success: bool
    platform: Platform
    video_id: Optional[str]

    # Video metadata
    title: Optional[str] = None
    author_name: Optional[str] = None
    author_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    thumbnail_width: Optional[int] = None
    thumbnail_height: Optional[int] = None

    # Platform-specific
    html: Optional[str] = None  # Embed HTML
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[int] = None
    post_type: Optional[str] = None  # reel, post, video, shorts

    # Error info
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class OEmbedService:
    """Service for fetching oEmbed metadata from video platforms."""

    # oEmbed endpoint URLs
    INSTAGRAM_OEMBED_URL = "https://graph.facebook.com/v18.0/instagram_oembed"
    YOUTUBE_OEMBED_URL = "https://www.youtube.com/oembed"
    TIKTOK_OEMBED_URL = "https://www.tiktok.com/oembed"

    # Timeout for HTTP requests
    REQUEST_TIMEOUT = 10

    # Facebook App Token for Instagram oEmbed (optional, increases rate limits)
    # Set via environment variable: FACEBOOK_APP_TOKEN
    _facebook_app_token: Optional[str] = None

    @classmethod
    def set_facebook_app_token(cls, token: str):
        """Set Facebook app token for Instagram oEmbed requests."""
        cls._facebook_app_token = token

    @classmethod
    def fetch_instagram_oembed(cls, url: str) -> OEmbedResponse:
        """
        Fetch oEmbed data from Instagram.

        Instagram's oEmbed API requires either:
        1. A Facebook App Token (for higher rate limits)
        2. Public access (limited, may fail)

        Args:
            url: Instagram post/reel URL

        Returns:
            OEmbedResponse with metadata or error
        """
        shortcode = extract_instagram_shortcode(url)
        if not shortcode:
            return OEmbedResponse(
                success=False,
                platform="instagram",
                video_id=None,
                error="Invalid Instagram URL - could not extract shortcode"
            )

        # Build the full URL for oEmbed request
        full_url = f"https://www.instagram.com/p/{shortcode}/"

        try:
            # Try Facebook Graph API oEmbed endpoint first (more reliable)
            params = {
                "url": full_url,
                "omitscript": "true",
                "maxwidth": "658",
            }

            # Add access token if available
            if cls._facebook_app_token:
                params["access_token"] = cls._facebook_app_token

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }

            response = requests.get(
                cls.INSTAGRAM_OEMBED_URL,
                params=params,
                headers=headers,
                timeout=cls.REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                return cls._parse_instagram_oembed(data, shortcode)

            # Try alternative: direct Instagram oEmbed (less reliable but no auth needed)
            alt_url = f"https://api.instagram.com/oembed/?url={quote(full_url)}"
            alt_response = requests.get(
                alt_url,
                headers=headers,
                timeout=cls.REQUEST_TIMEOUT
            )

            if alt_response.status_code == 200:
                data = alt_response.json()
                return cls._parse_instagram_oembed(data, shortcode)

            # Both failed
            return OEmbedResponse(
                success=False,
                platform="instagram",
                video_id=shortcode,
                error=f"Instagram oEmbed failed: HTTP {response.status_code}"
            )

        except requests.Timeout:
            return OEmbedResponse(
                success=False,
                platform="instagram",
                video_id=shortcode,
                error="Instagram oEmbed request timed out"
            )
        except requests.RequestException as e:
            return OEmbedResponse(
                success=False,
                platform="instagram",
                video_id=shortcode,
                error=f"Instagram oEmbed request failed: {str(e)}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error fetching Instagram oEmbed for {url}")
            return OEmbedResponse(
                success=False,
                platform="instagram",
                video_id=shortcode,
                error=f"Unexpected error: {str(e)}"
            )

    @classmethod
    def _parse_instagram_oembed(cls, data: Dict[str, Any], shortcode: str) -> OEmbedResponse:
        """Parse Instagram oEmbed response into standardized format."""
        # Extract title from caption (HTML content)
        title = data.get("title")
        if not title and data.get("html"):
            # Try to extract caption from HTML
            html = data.get("html", "")
            # Look for caption in blockquote
            caption_match = re.search(r'<p[^>]*>([^<]+)</p>', html)
            if caption_match:
                title = caption_match.group(1)[:100]  # Limit to 100 chars

        # Determine post type from HTML content
        post_type = "post"
        html_content = data.get("html", "").lower()
        if "reel" in html_content:
            post_type = "reel"
        elif "video" in html_content or "igtv" in html_content:
            post_type = "video"

        return OEmbedResponse(
            success=True,
            platform="instagram",
            video_id=shortcode,
            title=title,
            author_name=data.get("author_name"),
            author_url=data.get("author_url"),
            thumbnail_url=data.get("thumbnail_url"),
            thumbnail_width=data.get("thumbnail_width"),
            thumbnail_height=data.get("thumbnail_height"),
            html=data.get("html"),
            width=data.get("width"),
            height=data.get("height"),
            post_type=post_type,
        )

    @classmethod
    def fetch_youtube_oembed(cls, url: str) -> OEmbedResponse:
        """
        Fetch oEmbed data from YouTube.

        YouTube's oEmbed is publicly accessible without authentication.

        Args:
            url: YouTube video URL

        Returns:
            OEmbedResponse with metadata or error
        """
        video_id = extract_youtube_video_id(url)
        if not video_id:
            return OEmbedResponse(
                success=False,
                platform="youtube",
                video_id=None,
                error="Invalid YouTube URL - could not extract video ID"
            )

        full_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            params = {
                "url": full_url,
                "format": "json",
            }

            response = requests.get(
                cls.YOUTUBE_OEMBED_URL,
                params=params,
                timeout=cls.REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                return OEmbedResponse(
                    success=True,
                    platform="youtube",
                    video_id=video_id,
                    title=data.get("title"),
                    author_name=data.get("author_name"),
                    author_url=data.get("author_url"),
                    thumbnail_url=data.get("thumbnail_url"),
                    thumbnail_width=data.get("thumbnail_width"),
                    thumbnail_height=data.get("thumbnail_height"),
                    html=data.get("html"),
                    width=data.get("width"),
                    height=data.get("height"),
                    post_type="shorts" if "/shorts/" in url else "video",
                )

            return OEmbedResponse(
                success=False,
                platform="youtube",
                video_id=video_id,
                error=f"YouTube oEmbed failed: HTTP {response.status_code}"
            )

        except requests.Timeout:
            return OEmbedResponse(
                success=False,
                platform="youtube",
                video_id=video_id,
                error="YouTube oEmbed request timed out"
            )
        except requests.RequestException as e:
            return OEmbedResponse(
                success=False,
                platform="youtube",
                video_id=video_id,
                error=f"YouTube oEmbed request failed: {str(e)}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error fetching YouTube oEmbed for {url}")
            return OEmbedResponse(
                success=False,
                platform="youtube",
                video_id=video_id,
                error=f"Unexpected error: {str(e)}"
            )

    @classmethod
    def fetch_tiktok_oembed(cls, url: str) -> OEmbedResponse:
        """
        Fetch oEmbed data from TikTok.

        TikTok's oEmbed is publicly accessible.

        Args:
            url: TikTok video URL

        Returns:
            OEmbedResponse with metadata or error
        """
        video_id = extract_tiktok_video_id(url)
        if not video_id:
            return OEmbedResponse(
                success=False,
                platform="tiktok",
                video_id=None,
                error="Invalid TikTok URL - could not extract video ID"
            )

        try:
            params = {
                "url": url,  # TikTok oEmbed accepts the original URL
            }

            response = requests.get(
                cls.TIKTOK_OEMBED_URL,
                params=params,
                timeout=cls.REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()

                # TikTok provides duration in seconds
                duration = None
                if "duration" in data:
                    duration = int(data["duration"])

                return OEmbedResponse(
                    success=True,
                    platform="tiktok",
                    video_id=video_id,
                    title=data.get("title"),
                    author_name=data.get("author_name"),
                    author_url=data.get("author_url"),
                    thumbnail_url=data.get("thumbnail_url"),
                    thumbnail_width=data.get("thumbnail_width"),
                    thumbnail_height=data.get("thumbnail_height"),
                    html=data.get("html"),
                    width=data.get("width"),
                    height=data.get("height"),
                    duration_seconds=duration,
                    post_type="video",
                )

            return OEmbedResponse(
                success=False,
                platform="tiktok",
                video_id=video_id,
                error=f"TikTok oEmbed failed: HTTP {response.status_code}"
            )

        except requests.Timeout:
            return OEmbedResponse(
                success=False,
                platform="tiktok",
                video_id=video_id,
                error="TikTok oEmbed request timed out"
            )
        except requests.RequestException as e:
            return OEmbedResponse(
                success=False,
                platform="tiktok",
                video_id=video_id,
                error=f"TikTok oEmbed request failed: {str(e)}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error fetching TikTok oEmbed for {url}")
            return OEmbedResponse(
                success=False,
                platform="tiktok",
                video_id=video_id,
                error=f"Unexpected error: {str(e)}"
            )

    @classmethod
    def fetch_oembed(cls, url: str, platform: Optional[Platform] = None) -> OEmbedResponse:
        """
        Fetch oEmbed data for any supported platform.

        Auto-detects platform if not specified.

        Args:
            url: Video URL
            platform: Optional platform override

        Returns:
            OEmbedResponse with metadata or error
        """
        from workout_ingestor_api.services.video_url_normalizer import detect_platform

        if platform is None:
            platform = detect_platform(url)

        if platform == "youtube":
            return cls.fetch_youtube_oembed(url)
        elif platform == "instagram":
            return cls.fetch_instagram_oembed(url)
        elif platform == "tiktok":
            return cls.fetch_tiktok_oembed(url)
        else:
            return OEmbedResponse(
                success=False,
                platform="unknown",
                video_id=None,
                error=f"Unsupported platform for URL: {url}"
            )
