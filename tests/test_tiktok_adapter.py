from unittest.mock import patch, MagicMock
import pytest
from workout_ingestor_api.services.adapters.tiktok_adapter import TikTokAdapter
from workout_ingestor_api.services.adapters.base import MediaContent, PlatformFetchError
from workout_ingestor_api.services.tiktok_service import TikTokVideoMetadata


MOCK_METADATA = TikTokVideoMetadata(
    video_id="7575571317500546322",
    url="https://www.tiktok.com/@user/video/7575571317500546322",
    title="5 rounds: squats push-ups burpees #fitness #workout",
    author_name="fitnessguru",
    author_url="https://tiktok.com/@fitnessguru",
    hashtags=["fitness", "workout"],
)

EMPTY_METADATA = TikTokVideoMetadata(
    video_id="000",
    url="https://www.tiktok.com/@x/video/000",
    title="",
    author_name="x",
    author_url="",
    hashtags=[],
)


def test_platform_name():
    assert TikTokAdapter.platform_name() == "tiktok"


def test_fetch_uses_title_as_primary_text():
    with patch(
        "workout_ingestor_api.services.adapters.tiktok_adapter.TikTokService.extract_metadata",
        return_value=MOCK_METADATA,
    ):
        adapter = TikTokAdapter()
        result = adapter.fetch(
            "https://tiktok.com/@user/video/7575571317500546322",
            "7575571317500546322",
        )
        assert isinstance(result, MediaContent)
        assert "squats" in result.primary_text


def test_fetch_raises_on_empty_title():
    with patch(
        "workout_ingestor_api.services.adapters.tiktok_adapter.TikTokService.extract_metadata",
        return_value=EMPTY_METADATA,
    ):
        adapter = TikTokAdapter()
        with pytest.raises(PlatformFetchError):
            adapter.fetch("https://tiktok.com/@x/video/000", "000")


def test_fetch_raises_on_service_failure():
    with patch(
        "workout_ingestor_api.services.adapters.tiktok_adapter.TikTokService.extract_metadata",
        side_effect=RuntimeError("Network error"),
    ):
        adapter = TikTokAdapter()
        with pytest.raises(PlatformFetchError):
            adapter.fetch("https://tiktok.com/@user/video/123", "123")


def test_metadata_has_author():
    with patch(
        "workout_ingestor_api.services.adapters.tiktok_adapter.TikTokService.extract_metadata",
        return_value=MOCK_METADATA,
    ):
        adapter = TikTokAdapter()
        result = adapter.fetch(
            "https://tiktok.com/@user/video/7575571317500546322",
            "7575571317500546322",
        )
        assert result.media_metadata["author"] == "fitnessguru"
