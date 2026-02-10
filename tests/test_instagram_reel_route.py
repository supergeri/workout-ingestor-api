"""Tests for /ingest/instagram_reel endpoint."""
import pytest
from unittest.mock import patch, MagicMock


MOCK_WORKOUT_RESPONSE = {
    "title": "HIIT Workout",
    "workout_type": "hiit",
    "blocks": [{"label": "Circuit", "exercises": [{"name": "Squats"}]}],
    "source": "https://www.instagram.com/reel/DRHiuniDM1K/",
    "_provenance": {
        "mode": "instagram_reel",
        "source_url": "https://www.instagram.com/reel/DRHiuniDM1K/",
        "shortcode": "DRHiuniDM1K",
        "extraction_method": "apify_transcript",
    },
}


def test_ingest_instagram_reel_success(client):
    """POST /ingest/instagram_reel should return structured workout."""
    with patch(
        "workout_ingestor_api.services.instagram_reel_service.InstagramReelService.ingest_reel",
        return_value=MOCK_WORKOUT_RESPONSE,
    ), patch(
        "workout_ingestor_api.services.instagram_reel_cache_service.InstagramReelCacheService.get_cached_workout",
        return_value=None,
    ), patch(
        "workout_ingestor_api.services.instagram_reel_cache_service.InstagramReelCacheService.save_workout",
        return_value=True,
    ):
        resp = client.post(
            "/ingest/instagram_reel",
            json={"url": "https://www.instagram.com/reel/DRHiuniDM1K/"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "HIIT Workout"
    assert data["_provenance"]["mode"] == "instagram_reel"


def test_ingest_instagram_reel_invalid_url(client):
    """Should reject non-Instagram URLs."""
    resp = client.post(
        "/ingest/instagram_reel",
        json={"url": "https://www.youtube.com/watch?v=abc"},
    )
    assert resp.status_code == 400


def test_ingest_instagram_reel_missing_url(client):
    """Should return 422 when URL is missing."""
    resp = client.post("/ingest/instagram_reel", json={})
    assert resp.status_code == 422
