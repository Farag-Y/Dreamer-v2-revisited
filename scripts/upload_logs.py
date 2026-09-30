#!/usr/bin/env python3
"""Upload log files to R2 under a run prefix. Used by remote_run.sh on Vast.ai.

Usage:
    uv run python scripts/upload_logs.py <run_id> <file> [<file> ...]
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cloud_storage import get_client


def main() -> None:
    run_id, *files = sys.argv[1:]
    bucket = OmegaConf.load(ROOT / "conf" / "config.yaml").r2_bucket
    client = get_client()
    for file in map(Path, files):
        if file.exists():
            client.upload_file(str(file), bucket, f"{run_id}/{file.name}")
            print(f"[R2] {file.name} → {run_id}/{file.name}")


if __name__ == "__main__":
    main()
