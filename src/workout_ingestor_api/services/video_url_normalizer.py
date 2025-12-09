"""
Multi-platform video URL normalization utilities.

Handles URL detection, parsing, and normalization for:
- YouTube (youtube.com, youtu.be)
- Instagram (instagram.com reels, posts)
- TikTok (tiktok.com, vm.tiktok.com)
"""

from urllib.parse import urlparse, parse_qs
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
import re


# Platform type alias
Platform = Literal["youtube", "instagram", "tiktok", "unknown"]


@dataclass
class VideoUrlInfo:
    """Parsed video URL information."""
    platform: Platform
    video_id: Optional[str]
    normalized_url: Optional[str]
    original_url: str
    post_type: Optional[str] = None  # e.g., "reel", "post", "shorts", "video"


# Parameters to strip from URLs (tracking, playlist, timestamp)
STRIP_PARAMS = {'t', 'list', 'index', 'si', 'utm_source', 'utm_medium', 'utm_campaign', 'feature', 'igsh'}


# --------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------

def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.

    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://youtube.com/v/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - VIDEO_ID (bare ID, 11 characters)
    """
    if not url:
        return None

    url = url.strip()

    # Check if it's already a bare video ID (11 alphanumeric chars + _ and -)
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url

    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower()

        # youtu.be/VIDEO_ID
        if hostname == 'youtu.be':
            video_id = parsed.path.lstrip('/')
            video_id = video_id.split('/')[0]
            if video_id and len(video_id) == 11:
                return video_id
            return None

        # youtube.com variants
        if 'youtube' in hostname:
            # Standard watch URL: /watch?v=VIDEO_ID
            query = parse_qs(parsed.query)
            if 'v' in query and query['v']:
                return query['v'][0]

            path_parts = parsed.path.split('/')

            # Embed URL: /embed/VIDEO_ID
            if 'embed' in path_parts:
                idx = path_parts.index('embed')
                if idx + 1 < len(path_parts) and path_parts[idx + 1]:
                    return path_parts[idx + 1]

            # Old-style URL: /v/VIDEO_ID
            if 'v' in path_parts:
                idx = path_parts.index('v')
                if idx + 1 < len(path_parts) and path_parts[idx + 1]:
                    return path_parts[idx + 1]

            # Shorts URL: /shorts/VIDEO_ID
            if 'shorts' in path_parts:
                idx = path_parts.index('shorts')
                if idx + 1 < len(path_parts) and path_parts[idx + 1]:
                    return path_parts[idx + 1]

        return None
    except Exception:
        return None


def normalize_youtube_url(url: str) -> Optional[str]:
    """Normalize YouTube URL to canonical form: youtube.com/watch?v=VIDEO_ID"""
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return None
    return f"youtube.com/watch?v={video_id}"


def get_youtube_post_type(url: str) -> Optional[str]:
    """Detect if the YouTube URL is a shorts or regular video."""
    if '/shorts/' in url.lower():
        return "shorts"
    return "video"


# --------------------------------------------------------------------------
# Instagram
# --------------------------------------------------------------------------

# Instagram shortcode regex - matches /p/, /reel/, /tv/
INSTAGRAM_SHORTCODE_RE = re.compile(
    r"(?:instagram\.com|instagr\.am)/(?:p|reel|tv|reels)/([A-Za-z0-9_-]+)",
    re.IGNORECASE
)


def extract_instagram_shortcode(url: str) -> Optional[str]:
    """
    Extract Instagram shortcode from URL.

    Supported formats:
    - https://www.instagram.com/p/SHORTCODE/
    - https://www.instagram.com/reel/SHORTCODE/
    - https://www.instagram.com/reels/SHORTCODE/
    - https://www.instagram.com/tv/SHORTCODE/
    - https://instagr.am/p/SHORTCODE/
    """
    if not url:
        return None

    url = url.strip()
    match = INSTAGRAM_SHORTCODE_RE.search(url)
    if match:
        return match.group(1)
    return None


def normalize_instagram_url(url: str) -> Optional[str]:
    """Normalize Instagram URL to canonical form: instagram.com/reel/SHORTCODE"""
    shortcode = extract_instagram_shortcode(url)
    if not shortcode:
        return None

    # Determine post type from original URL
    post_type = get_instagram_post_type(url)
    return f"instagram.com/{post_type}/{shortcode}"


def get_instagram_post_type(url: str) -> str:
    """Detect Instagram post type from URL."""
    url_lower = url.lower()
    if '/reel/' in url_lower or '/reels/' in url_lower:
        return "reel"
    if '/tv/' in url_lower:
        return "tv"
    return "p"  # Default to post


# --------------------------------------------------------------------------
# TikTok
# --------------------------------------------------------------------------

# TikTok video ID regex - matches /@user/video/ID or /v/ID or direct ID
TIKTOK_VIDEO_ID_RE = re.compile(
    r"(?:tiktok\.com|vm\.tiktok\.com)/(?:@[\w.-]+/video/|v/)?(\d+)",
    re.IGNORECASE
)

# Short TikTok URLs (vm.tiktok.com/CODE)
TIKTOK_SHORT_CODE_RE = re.compile(
    r"vm\.tiktok\.com/([A-Za-z0-9]+)",
    re.IGNORECASE
)


def extract_tiktok_video_id(url: str) -> Optional[str]:
    """
    Extract TikTok video ID from URL.

    Supported formats:
    - https://www.tiktok.com/@user/video/1234567890123456789
    - https://vm.tiktok.com/ZMxxxxxxx/
    - https://www.tiktok.com/v/1234567890123456789

    Note: Short URLs (vm.tiktok.com) return the short code, not the full video ID.
    These would need to be resolved via redirect to get the full ID.
    """
    if not url:
        return None

    url = url.strip()

    # Try to extract full video ID first
    match = TIKTOK_VIDEO_ID_RE.search(url)
    if match:
        return match.group(1)

    # Try short URL code
    short_match = TIKTOK_SHORT_CODE_RE.search(url)
    if short_match:
        return f"short:{short_match.group(1)}"

    return None


def normalize_tiktok_url(url: str) -> Optional[str]:
    """Normalize TikTok URL to canonical form."""
    video_id = extract_tiktok_video_id(url)
    if not video_id:
        return None

    # For short URLs, keep as-is since we can't resolve without HTTP request
    if video_id.startswith("short:"):
        short_code = video_id[6:]
        return f"vm.tiktok.com/{short_code}"

    return f"tiktok.com/video/{video_id}"


# --------------------------------------------------------------------------
# Platform Detection & Unified Parsing
# --------------------------------------------------------------------------

def detect_platform(url: str) -> Platform:
    """
    Detect which platform a URL belongs to.

    Returns: "youtube", "instagram", "tiktok", or "unknown"
    """
    if not url:
        return "unknown"

    url_lower = url.lower().strip()

    # YouTube
    if any(domain in url_lower for domain in ['youtube.com', 'youtu.be']):
        return "youtube"

    # Instagram
    if any(domain in url_lower for domain in ['instagram.com', 'instagr.am']):
        return "instagram"

    # TikTok
    if 'tiktok.com' in url_lower:
        return "tiktok"

    return "unknown"


def parse_video_url(url: str) -> VideoUrlInfo:
    """
    Parse any video URL and return structured information.

    This is the main entry point for URL parsing.

    Args:
        url: Video URL from any supported platform

    Returns:
        VideoUrlInfo with platform, video_id, normalized_url, etc.
    """
    original_url = url.strip() if url else ""
    platform = detect_platform(original_url)

    if platform == "youtube":
        video_id = extract_youtube_video_id(original_url)
        normalized = normalize_youtube_url(original_url)
        post_type = get_youtube_post_type(original_url)
        return VideoUrlInfo(
            platform=platform,
            video_id=video_id,
            normalized_url=normalized,
            original_url=original_url,
            post_type=post_type
        )

    elif platform == "instagram":
        video_id = extract_instagram_shortcode(original_url)
        normalized = normalize_instagram_url(original_url)
        post_type = get_instagram_post_type(original_url)
        return VideoUrlInfo(
            platform=platform,
            video_id=video_id,
            normalized_url=normalized,
            original_url=original_url,
            post_type=post_type
        )

    elif platform == "tiktok":
        video_id = extract_tiktok_video_id(original_url)
        normalized = normalize_tiktok_url(original_url)
        return VideoUrlInfo(
            platform=platform,
            video_id=video_id,
            normalized_url=normalized,
            original_url=original_url,
            post_type="video"
        )

    else:
        return VideoUrlInfo(
            platform="unknown",
            video_id=None,
            normalized_url=None,
            original_url=original_url,
            post_type=None
        )


def get_full_url(platform: Platform, video_id: str) -> Optional[str]:
    """
    Get the full URL for a video given its platform and ID.

    Args:
        platform: The video platform
        video_id: Platform-specific video ID

    Returns:
        Full URL string
    """
    if platform == "youtube":
        return f"https://www.youtube.com/watch?v={video_id}"
    elif platform == "instagram":
        return f"https://www.instagram.com/reel/{video_id}/"
    elif platform == "tiktok":
        if video_id.startswith("short:"):
            return f"https://vm.tiktok.com/{video_id[6:]}/"
        return f"https://www.tiktok.com/video/{video_id}"
    return None
