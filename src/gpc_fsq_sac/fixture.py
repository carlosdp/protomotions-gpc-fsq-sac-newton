"""Explicit, checksum-verified acquisition of the public mini MotionLib."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from .constants import (
    MOTION_BYTES,
    MOTION_COUNT,
    MOTION_FILENAME,
    MOTION_MANIFEST_SHA256,
    MOTION_SHA256,
    MOTION_URL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_motion_file(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != MOTION_BYTES:
        raise ValueError(f"unexpected motion file size: {size} != {MOTION_BYTES}")
    if digest != MOTION_SHA256:
        raise ValueError(f"unexpected motion file sha256: {digest} != {MOTION_SHA256}")
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "expected_motion_count": MOTION_COUNT,
        "expected_manifest_sha256": MOTION_MANIFEST_SHA256,
        "source": MOTION_URL,
    }


def manifest_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        motion_files = payload.get("motion_files")
    else:
        motion_files = getattr(payload, "motion_files", None)
    if motion_files is None:
        raise ValueError("MotionLib payload has no motion_files manifest")
    return [str(item) for item in motion_files]


def verify_motion_manifest(path: Path) -> dict:
    import torch

    metadata = verify_motion_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = manifest_from_payload(payload)
    manifest_sha256 = hashlib.sha256(
        ("\n".join(manifest) + "\n").encode("utf-8")
    ).hexdigest()
    if len(manifest) != MOTION_COUNT:
        raise ValueError(f"unexpected motion count: {len(manifest)} != {MOTION_COUNT}")
    if manifest_sha256 != MOTION_MANIFEST_SHA256:
        raise ValueError(
            "unexpected fixed-order motion manifest sha256: "
            f"{manifest_sha256} != {MOTION_MANIFEST_SHA256}"
        )
    metadata["motion_count"] = len(manifest)
    metadata["manifest_sha256"] = manifest_sha256
    metadata["motion_files"] = manifest
    return metadata


def fetch_motion_file(destination: Path, *, accept_license: bool) -> dict:
    if not accept_license:
        raise ValueError(
            "BONES-SEED license acknowledgement is required; pass "
            "--accept-bones-seed-license after reviewing "
            "https://bones.studio/info/seed-license"
        )
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(MOTION_URL, partial)
    metadata = verify_motion_manifest(partial)
    partial.replace(destination)
    metadata["path"] = str(destination)
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def default_motion_path() -> Path:
    return Path("data") / MOTION_FILENAME
