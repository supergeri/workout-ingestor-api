"""Integration test for Instagram Reel ingestion (requires APIFY_API_TOKEN)."""
import os
import pytest
from workout_ingestor_api.services.instagram_reel_service import InstagramReelService

SKIP_REASON = "Set APIFY_API_TOKEN and OPENAI_API_KEY to run integration tests"


@pytest.mark.skipif(
    not os.getenv("APIFY_API_TOKEN") or not os.getenv("OPENAI_API_KEY"),
    reason=SKIP_REASON,
)
def test_real_reel_ingestion():
    """Test with an actual public Instagram Reel URL."""
    url = "https://www.instagram.com/reel/DRHiuniDM1K/"

    result = InstagramReelService.ingest_reel(url=url)

    assert "title" in result
    assert "blocks" in result
    assert len(result["blocks"]) > 0
    assert result["_provenance"]["mode"] == "instagram_reel"

    # Verify chapters (video_start_sec) are present on at least some exercises
    has_timestamps = False
    for block in result["blocks"]:
        for ex in block.get("exercises", []):
            if ex.get("video_start_sec") is not None:
                has_timestamps = True
                break
    assert has_timestamps, "Expected at least one exercise with video_start_sec"
