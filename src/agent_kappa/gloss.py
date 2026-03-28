"""
Auto-download and manage the gloss binary for pretty terminal output.

On first use of --pretty, downloads the correct gloss binary from GitHub
to ~/.agent-kappa/bin/gloss. Subsequent runs use the cached binary.
"""

from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

GLOSS_VERSION = "v0.1.0"
GLOSS_REPO = "nalalou/gloss"
GLOSS_DIR = Path.home() / ".agent-kappa" / "bin"
GLOSS_PATH = GLOSS_DIR / "gloss"

PLATFORM_MAP = {
    ("Darwin", "arm64"): "gloss-darwin-arm64",
    ("Darwin", "x86_64"): "gloss-darwin-amd64",
    ("Linux", "x86_64"): "gloss-linux-amd64",
}


def _get_asset_name() -> str:
    system = platform.system()
    machine = platform.machine()
    key = (system, machine)
    if key not in PLATFORM_MAP:
        raise RuntimeError(
            f"No gloss binary available for {system}/{machine}. "
            f"Supported: {', '.join(f'{s}/{m}' for s, m in PLATFORM_MAP)}"
        )
    return PLATFORM_MAP[key]


def _download_url(asset_name: str) -> str:
    return f"https://github.com/{GLOSS_REPO}/releases/download/{GLOSS_VERSION}/{asset_name}"


def ensure_gloss() -> Path:
    """Ensure gloss binary is available. Downloads on first use."""
    if GLOSS_PATH.exists():
        return GLOSS_PATH

    asset_name = _get_asset_name()
    url = _download_url(asset_name)

    print(f"Downloading gloss {GLOSS_VERSION} ({asset_name})...", file=sys.stderr)
    GLOSS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, GLOSS_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to download gloss from {url}: {e}")

    # Make executable
    GLOSS_PATH.chmod(GLOSS_PATH.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed gloss to {GLOSS_PATH}", file=sys.stderr)
    return GLOSS_PATH


def run_with_gloss(cmd: list[str]) -> int:
    """Run a command and pipe its output through gloss watch."""
    gloss = ensure_gloss()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    gloss_proc = subprocess.Popen(
        [str(gloss), "watch"],
        stdin=proc.stdout,
    )
    proc.stdout.close()
    gloss_proc.wait()
    proc.wait()
    return proc.returncode
