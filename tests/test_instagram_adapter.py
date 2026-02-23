import os
import tempfile
from unittest.mock import MagicMock, patch, call
import pytest
from workout_ingestor_api.services.adapters.instagram_adapter import InstagramAdapter
from workout_ingestor_api.services.adapters.base import MediaContent, PlatformFetchError

REEL_WITH_TRANSCRIPT = {
    "caption": "X4 Rounds\nSquat 3x10\n#hyrox",
    "transcript": "Today we do four rounds of squats",
    "videoDuration": 30.0,
    "ownerUsername": "coach_anna",
    "shortCode": "DEaDjHLtHwA",
}

REEL_NO_TRANSCRIPT = {
    "caption": "X4 Rounds\nSquat 3x10",
    "transcript": None,
    "videoDuration": 14.7,
    "ownerUsername": "commando_charlie",
    "shortCode": "DEaDjHLtHwA",
}


def test_platform_name():
    assert InstagramAdapter.platform_name() == "instagram"


def test_fetch_uses_transcript_when_available():
    with patch(
        "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
        return_value=REEL_WITH_TRANSCRIPT,
    ):
        result = InstagramAdapter().fetch("https://instagram.com/reel/DEaDjHLtHwA/", "DEaDjHLtHwA")
        assert isinstance(result, MediaContent)
        assert result.primary_text == "Today we do four rounds of squats"
        assert result.secondary_texts == ["X4 Rounds\nSquat 3x10\n#hyrox"]


def test_fetch_falls_back_to_caption_when_no_transcript():
    with patch(
        "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
        return_value=REEL_NO_TRANSCRIPT,
    ):
        result = InstagramAdapter().fetch("https://instagram.com/p/DEaDjHLtHwA/", "DEaDjHLtHwA")
        assert result.primary_text == "X4 Rounds\nSquat 3x10"


def test_fetch_raises_when_no_text():
    empty = {"caption": "", "transcript": None, "videoDuration": 10, "ownerUsername": "x", "shortCode": "abc"}
    with patch(
        "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
        return_value=empty,
    ):
        with pytest.raises(PlatformFetchError, match="no transcript or caption"):
            InstagramAdapter().fetch("https://instagram.com/p/abc/", "abc")


def test_fetch_raises_on_apify_error():
    with patch(
        "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
        side_effect=RuntimeError("Apify down"),
    ):
        with pytest.raises(PlatformFetchError):
            InstagramAdapter().fetch("https://instagram.com/p/abc/", "abc")


def test_media_metadata_populated():
    with patch(
        "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
        return_value=REEL_NO_TRANSCRIPT,
    ):
        result = InstagramAdapter().fetch("https://instagram.com/p/DEaDjHLtHwA/", "DEaDjHLtHwA")
        assert result.media_metadata["creator"] == "commando_charlie"
        assert result.media_metadata["video_duration_sec"] == 14.7


def test_secondary_texts_empty_when_caption_is_whitespace_only():
    data = {
        "caption": "   ",  # whitespace-only
        "transcript": "Today we do four rounds of squats",
        "videoDuration": 30.0,
        "ownerUsername": "coach_anna",
        "shortCode": "DEaDjHLtHwA",
    }
    with patch(
        "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
        return_value=data,
    ):
        adapter = InstagramAdapter()
        result = adapter.fetch("https://instagram.com/reel/DEaDjHLtHwA/", "DEaDjHLtHwA")
        assert result.primary_text == "Today we do four rounds of squats"
        assert result.secondary_texts == []


# ---------------------------------------------------------------------------
# Sidecar (carousel) tests — AMA-742
# ---------------------------------------------------------------------------

SIDECAR_REEL = {
    "type": "Sidecar",
    "caption": "Full body circuit 4 rounds",
    "transcript": None,
    "videoDuration": None,
    "ownerUsername": "coach_sidecar",
    "shortCode": "SC1234",
    "childPosts": [
        {"videoUrl": "https://example.com/clip1.mp4"},
        {"videoUrl": "https://example.com/clip2.mp4"},
        {"videoUrl": "https://example.com/clip3.mp4"},
    ],
}

SIDECAR_REEL_NO_VIDEO_URL = {
    "type": "Sidecar",
    "caption": "No video URLs here",
    "transcript": None,
    "videoDuration": None,
    "ownerUsername": "coach_x",
    "shortCode": "SC_NOVID",
    "childPosts": [
        {"imageUrl": "https://example.com/img1.jpg"},
    ],
}


def _make_frame_tuples(tmpdir: str, names: list[str]) -> list[tuple[str, float]]:
    """Create real (empty) PNG files in tmpdir and return (path, ts) tuples."""
    result = []
    for i, name in enumerate(names):
        path = os.path.join(tmpdir, name)
        open(path, "wb").close()
        result.append((path, float(i)))
    return result


class TestSidecarHandling:
    """Tests for Sidecar (carousel) post handling."""

    def _patch_stack(self, vision_text="Squat 3x10\nDeadlift 4x6"):
        """Return a context manager that patches all external calls for Sidecar path."""
        apify_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
            return_value=SIDECAR_REEL,
        )
        # httpx.stream is a context manager returning a response object
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = iter([b"fakevideobytes"])
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        httpx_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.httpx.stream",
            return_value=mock_stream_cm,
        )
        # KeyframeService returns list of (path, ts) tuples
        fake_frames = [("/tmp/frame_00000_0.00s.png", 0.0), ("/tmp/frame_00001_2.00s.png", 2.0)]
        keyframe_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.KeyframeService.extract_periodic_frames",
            return_value=[0.0, 2.0],
        )
        extract_frames_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.KeyframeService.extract_frames_at_timestamps",
            return_value=fake_frames,
        )
        vision_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.VisionService.extract_text_from_images",
            return_value=vision_text,
        )
        return apify_patch, httpx_patch, keyframe_patch, extract_frames_patch, vision_patch

    def test_sidecar_primary_text_is_vision_output(self):
        """Vision-extracted text becomes primary_text for Sidecar posts."""
        apify, httpx, kf_periodic, kf_frames, vision = self._patch_stack("Squat 3x10\nDeadlift 4x6")
        with apify, httpx, kf_periodic, kf_frames, vision:
            result = InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        assert result.primary_text == "Squat 3x10\nDeadlift 4x6"

    def test_sidecar_caption_in_secondary_texts(self):
        """Caption is placed in secondary_texts for Sidecar posts."""
        apify, httpx, kf_periodic, kf_frames, vision = self._patch_stack()
        with apify, httpx, kf_periodic, kf_frames, vision:
            result = InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        assert "Full body circuit 4 rounds" in result.secondary_texts

    def test_sidecar_had_vision_flag_in_metadata(self):
        """had_vision=True is set in media_metadata for Sidecar posts."""
        apify, httpx, kf_periodic, kf_frames, vision = self._patch_stack()
        with apify, httpx, kf_periodic, kf_frames, vision:
            result = InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        assert result.media_metadata.get("had_vision") is True

    def test_sidecar_child_post_count_in_metadata(self):
        """sidecar_child_count is set in media_metadata."""
        apify, httpx, kf_periodic, kf_frames, vision = self._patch_stack()
        with apify, httpx, kf_periodic, kf_frames, vision:
            result = InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        assert result.media_metadata.get("sidecar_child_count") == 3

    def test_sidecar_max_8_child_posts(self):
        """At most 8 child posts are processed even if more are present."""
        reel_with_many = dict(SIDECAR_REEL)
        reel_with_many["childPosts"] = [
            {"videoUrl": f"https://example.com/clip{i}.mp4"} for i in range(12)
        ]
        apify_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
            return_value=reel_with_many,
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = iter([b"fakevideobytes"])
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        httpx_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.httpx.stream",
            return_value=mock_stream_cm,
        )
        kf_periodic = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.KeyframeService.extract_periodic_frames",
            return_value=[0.0],
        )
        fake_frames = [("/tmp/frame_00000_0.00s.png", 0.0)]
        kf_frames = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.KeyframeService.extract_frames_at_timestamps",
            return_value=fake_frames,
        )
        vision_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.VisionService.extract_text_from_images",
            return_value="some workout text",
        )
        with apify_patch, httpx_patch as mock_httpx, kf_periodic, kf_frames, vision_patch:
            InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        # httpx.stream should be called exactly 8 times (capped at _MAX_SIDECAR_CLIPS)
        assert mock_httpx.call_count == 8

    def test_sidecar_vision_failure_falls_back_to_caption(self):
        """If VisionService raises, fall back to caption-only behaviour."""
        apify_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
            return_value=SIDECAR_REEL,
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = iter([b"fakevideobytes"])
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        httpx_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.httpx.stream",
            return_value=mock_stream_cm,
        )
        kf_periodic = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.KeyframeService.extract_periodic_frames",
            return_value=[0.0],
        )
        kf_frames = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.KeyframeService.extract_frames_at_timestamps",
            return_value=[("/tmp/frame_00000_0.00s.png", 0.0)],
        )
        vision_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.VisionService.extract_text_from_images",
            side_effect=ValueError("No API key"),
        )
        with apify_patch, httpx_patch, kf_periodic, kf_frames, vision_patch:
            result = InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        # Should fall back to caption
        assert result.primary_text == "Full body circuit 4 rounds"
        assert result.media_metadata.get("had_vision") is not True

    def test_sidecar_temp_cleanup_on_vision_failure(self):
        """shutil.rmtree is called even when VisionService raises mid-pipeline."""
        apify_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
            return_value=SIDECAR_REEL,
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = iter([b"fakevideobytes"])
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        httpx_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.httpx.stream",
            return_value=mock_stream_cm,
        )
        kf_periodic = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.KeyframeService.extract_periodic_frames",
            return_value=[0.0],
        )
        kf_frames = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.KeyframeService.extract_frames_at_timestamps",
            return_value=[("/tmp/frame_00000_0.00s.png", 0.0)],
        )
        vision_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.VisionService.extract_text_from_images",
            side_effect=RuntimeError("Vision API unavailable"),
        )
        rmtree_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.shutil.rmtree",
        )
        with apify_patch, httpx_patch, kf_periodic, kf_frames, vision_patch, rmtree_patch as mock_rmtree:
            result = InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        # Cleanup must run even though vision raised
        mock_rmtree.assert_called_once()
        # Fallback to caption because vision failed
        assert result.primary_text == "Full body circuit 4 rounds"

    def test_sidecar_child_posts_without_video_url_are_skipped(self):
        """Child posts without a videoUrl are skipped gracefully."""
        apify_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
            return_value=SIDECAR_REEL_NO_VIDEO_URL,
        )
        with apify_patch:
            # No video URLs means no frames, so we expect fallback to caption
            result = InstagramAdapter().fetch("https://instagram.com/p/SC_NOVID/", "SC_NOVID")
        assert result.primary_text == "No video URLs here"

    def test_non_sidecar_reel_unchanged(self):
        """Regular (non-Sidecar) reel behaviour is completely unchanged."""
        apify_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
            return_value=REEL_WITH_TRANSCRIPT,
        )
        vision_spy = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.VisionService.extract_text_from_images",
        )
        with apify_patch, vision_spy as mock_vision:
            result = InstagramAdapter().fetch("https://instagram.com/reel/DEaDjHLtHwA/", "DEaDjHLtHwA")
        # VisionService must NOT be called for regular reels
        mock_vision.assert_not_called()
        assert result.primary_text == "Today we do four rounds of squats"

    def test_sidecar_result_is_media_content(self):
        """Sidecar path returns a proper MediaContent instance."""
        apify, httpx, kf_periodic, kf_frames, vision = self._patch_stack()
        with apify, httpx, kf_periodic, kf_frames, vision:
            result = InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        assert isinstance(result, MediaContent)

    def test_sidecar_with_empty_child_posts_list(self):
        """Sidecar type with empty childPosts list falls back to caption/transcript path."""
        reel_empty_children = dict(SIDECAR_REEL)
        reel_empty_children["childPosts"] = []
        apify_patch = patch(
            "workout_ingestor_api.services.adapters.instagram_adapter.ApifyService.fetch_reel_data",
            return_value=reel_empty_children,
        )
        with apify_patch:
            result = InstagramAdapter().fetch("https://instagram.com/p/SC1234/", "SC1234")
        # Falls back to caption because childPosts is empty
        assert result.primary_text == "Full body circuit 4 rounds"
