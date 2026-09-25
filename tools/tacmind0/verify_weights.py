#!/usr/bin/env python3
"""Verify bundled TacMind-0 pretrained files against the checked-in SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "vtla" / "frameworks" / "tacmind0" / "weights_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for asset in data["files"]:
        path = ROOT / asset["path"]
        if not path.is_file():
            failures.append(f"missing: {asset['path']}")
            continue
        if path.stat().st_size != asset["bytes"]:
            failures.append(f"size mismatch: {asset['path']}")
            continue
        actual = sha256(path)
        if actual != asset["sha256"]:
            failures.append(f"SHA-256 mismatch: {asset['path']}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(data['files'])} files from {data['source_repo']} commit {data['source_commit']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
