"""Instagram platform adapter — fetches via Apify."""
from __future__ import annotations

import logging
import shutil
import tempfile

import httpx

from workout_ingestor_api.services.apify_service import ApifyService
from workout_ingestor_api.services.keyframe_service import KeyframeService
from workout_ingestor_api.services.vision_service import VisionService
from .base import PlatformAdapter, MediaContent, PlatformFetchError
from . import register_adapter

logger = logging.getLogger(__name__)

_MAX_SIDECAR_CLIPS = 8


class InstagramAdapter(PlatformAdapter):
    """Fetches Instagram posts/reels/TV via Apify."""

    @staticmethod
    def platform_name() -> str:
        return "instagram"

    def fetch(self, url: str, source_id: str) -> MediaContent:
        try:
            reel = ApifyService.fetch_reel_data(url)
        except Exception as e:
            raise PlatformFetchError(f"Instagram fetch failed for {source_id}: {e}") from e

        caption: str = reel.get("caption") or ""
        transcript: str = reel.get("transcript") or ""
        duration: float | None = reel.get("videoDuration")
        creator: str = reel.get("ownerUsername", "unknown")

        # ------------------------------------------------------------------
        # Sidecar (carousel) path — extract on-screen text via vision
        # ------------------------------------------------------------------
        is_sidecar = reel.get("type") == "Sidecar"
        child_posts = reel.get("childPosts") or []
        video_children = [
            cp for cp in child_posts if cp.get("videoUrl")
        ][:_MAX_SIDECAR_CLIPS]

        if is_sidecar and video_children:
            vision_text = self._extract_sidecar_text(video_children)
            if vision_text:
                title = (caption.split("\n")[0] if caption else f"Instagram by @{creator}")[:80]
                return MediaContent(
                    primary_text=vision_text,
                    secondary_texts=[caption.strip()] if caption.strip() else [],
                    title=title,
                    media_metadata={
                        "video_duration_sec": duration,
                        "creator": creator,
                        "shortcode": source_id,
                        "had_transcript": False,
                        "had_vision": True,
                        "sidecar_child_count": len(video_children),
                    },
                )
            # VisionService failed — log already done inside helper; fall through
            logger.warning(
                "Sidecar vision extraction failed for %s — falling back to caption", source_id
            )

        # ------------------------------------------------------------------
        # Regular reel path (unchanged behaviour)
        # ------------------------------------------------------------------
        primary_text = transcript.strip() if transcript.strip() else caption.strip()
        if not primary_text:
            raise PlatformFetchError(f"Instagram post {source_id} has no transcript or caption.")

        title = (caption.split("\n")[0] if caption else f"Instagram by @{creator}")[:80]

        return MediaContent(
            primary_text=primary_text,
            secondary_texts=[caption.strip()] if transcript.strip() and caption.strip() else [],
            title=title,
            media_metadata={
                "video_duration_sec": duration,
                "creator": creator,
                "shortcode": source_id,
                "had_transcript": bool(transcript.strip()),
                "had_vision": False,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_sidecar_text(self, video_children: list[dict]) -> str | None:
        """
        Download child video clips, extract keyframes, and run VisionService.

        Returns the extracted text string, or None if extraction fails.
        """
        tmpdir = tempfile.mkdtemp(prefix="sidecar_")
        try:
            all_frame_paths: list[str] = []

            for idx, child in enumerate(video_children):
                video_url: str = child["videoUrl"]
                clip_path = f"{tmpdir}/clip_{idx:02d}.mp4"
                frames_dir = f"{tmpdir}/frames_{idx:02d}"

                # Download the MP4
                try:
                    with httpx.stream("GET", video_url, follow_redirects=True, timeout=30) as response:
                        response.raise_for_status()
                        with open(clip_path, "wb") as fh:
                            for chunk in response.iter_bytes(chunk_size=1 << 20):
                                fh.write(chunk)
                except Exception as exc:
                    logger.warning("Failed to download clip %d (%s): %s", idx, video_url, exc)
                    continue

                # Extract keyframes — 1-2 frames per short clip at 0.5 fps
                try:
                    timestamps = KeyframeService.extract_periodic_frames(clip_path, fps=0.5)
                    frame_tuples = KeyframeService.extract_frames_at_timestamps(
                        clip_path, timestamps, frames_dir
                    )
                    all_frame_paths.extend(fp for fp, _ in frame_tuples)
                except Exception as exc:
                    logger.warning("Keyframe extraction failed for clip %d: %s", idx, exc)
                    continue

            if not all_frame_paths:
                logger.warning("No frames extracted from any Sidecar child clips.")
                return None

            # Run vision model on all collected frames
            vision_text = VisionService.extract_text_from_images(
                all_frame_paths,
                provider="openai",
                model="gpt-4o-mini",
            )
            return vision_text.strip() if vision_text and vision_text.strip() else None

        except Exception as exc:
            logger.warning("Sidecar vision pipeline error: %s", exc)
            return None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


register_adapter(InstagramAdapter)
