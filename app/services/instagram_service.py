"""Helpers for downloading Instagram posts via Instaloader."""

from __future__ import annotations

import os
import re
from typing import List

import instaloader
from instaloader import Post
from instaloader.exceptions import (
    BadCredentialsException,
    ConnectionException,
    InstaloaderException,
    TwoFactorAuthRequiredException,
)


SHORTCODE_RE = re.compile(r"(?:instagram\.com/(?:p|reel|tv)/|/p/)([A-Za-z0-9_-]+)")


class InstagramServiceError(RuntimeError):
    """Raised when Instaloader cannot download the requested media."""


class InstagramService:
    """Wrapper around Instaloader to fetch Instagram post images."""

    @staticmethod
    def _extract_shortcode(url: str) -> str:
        match = SHORTCODE_RE.search(url)
        if not match:
            raise InstagramServiceError("Could not extract Instagram shortcode from the provided URL.")
        return match.group(1)

    @staticmethod
    def download_post_images(
        username: str,
        password: str,
        url: str,
        target_dir: str,
    ) -> List[str]:
        """Download all images for a single Instagram post using the given credentials."""

        os.makedirs(target_dir, exist_ok=True)
        shortcode = InstagramService._extract_shortcode(url)

        loader = instaloader.Instaloader(
            download_videos=False,
            download_video_thumbnails=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            post_metadata_txt_pattern="",
            dirname_pattern=target_dir,
            filename_pattern=f"{shortcode}_{{index}}",
            quiet=True,
        )

        try:
            loader.login(username, password)
        except TwoFactorAuthRequiredException as exc:
            raise InstagramServiceError(
                "Two-factor authentication is required for this account, which is not supported yet."
            ) from exc
        except BadCredentialsException as exc:
            raise InstagramServiceError("Instagram login failed: invalid username or password.") from exc
        except ConnectionException as exc:
            raise InstagramServiceError(f"Instagram login failed: {exc}.") from exc

        try:
            post = Post.from_shortcode(loader.context, shortcode)
        except InstaloaderException as exc:
            raise InstagramServiceError(f"Could not load Instagram post: {exc}.") from exc

        nodes = []
        if post.typename == "GraphSidecar":
            nodes.extend(post.get_sidecar_nodes())
        else:
            nodes.append(post)

        image_paths: List[str] = []
        for idx, node in enumerate(nodes):
            if getattr(node, "is_video", False):
                continue

            filepath_root = os.path.join(target_dir, f"{shortcode}_{idx}")
            media_url = getattr(node, "display_url", None) or getattr(node, "url", None)
            if not media_url:
                continue

            media_date = getattr(node, "date_utc", getattr(post, "date_utc", None))
            downloaded = loader.download_pic(filepath_root, media_url, media_date)
            if not downloaded:
                continue

            basename = os.path.basename(filepath_root)
            for entry in os.listdir(target_dir):
                if entry.startswith(basename):
                    candidate = os.path.join(target_dir, entry)
                    if os.path.isfile(candidate):
                        image_paths.append(candidate)
                        break

        if not image_paths:
            raise InstagramServiceError("No images were downloaded for the Instagram post.")

        return image_paths



