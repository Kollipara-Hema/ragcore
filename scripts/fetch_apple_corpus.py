"""
Copy the 5 Apple corpus files from an external staging directory into
data/apple_demo/.

Pre-flight: every source file must exist before any copy runs. On any missing
source the script prints the full set of missing names and exits non-zero,
leaving data/apple_demo/ untouched. Post-copy: each destination file's size is
verified against its source.

Provenance and exclusions are documented in data/README.md.

Usage:
    python scripts/fetch_apple_corpus.py
    python scripts/fetch_apple_corpus.py --source /custom/source/dir
    python scripts/fetch_apple_corpus.py --dest /custom/dest/dir
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SOURCE = Path("/Users/hemakollipara/Desktop/files_ragcore")
DEFAULT_DEST = REPO_ROOT / "data" / "apple_demo"

# Authoritative list of the 5 corpus files. Kept in sync with
# scripts/smoke_ingest_apple.py (loader-validation gate).
CORPUS_FILES: tuple[str, ...] = (
    "apple__sec__form_10k__2025.pdf",
    "apple__sec__q4_earnings_release__2025.html",
    "apple__corporate__environmental_progress_report__2025.pdf",
    "apple__sec__financial_metrics_quarterly__2026.csv",
    "apple__sec__financial_metrics_annual__2026.csv",
)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the Apple corpus files into data/apple_demo/.",
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help=f"Directory containing the 5 corpus files (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()

    source_dir: Path = args.source
    dest_dir: Path = args.dest

    print(f"source: {source_dir}")
    print(f"dest:   {dest_dir}")
    print()

    # Pre-flight: verify ALL sources exist before any copy. Avoids partial
    # population on a missing-file failure.
    missing = [name for name in CORPUS_FILES if not (source_dir / name).is_file()]
    if missing:
        print(f"ERROR: {len(missing)} source file(s) missing in {source_dir}:")
        for name in missing:
            print(f"  - {name}")
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int, int, str]] = []
    for name in CORPUS_FILES:
        src = source_dir / name
        dst = dest_dir / name
        src_size = src.stat().st_size
        shutil.copy2(src, dst)
        dst_size = dst.stat().st_size
        status = "ok" if dst_size == src_size else "SIZE MISMATCH"
        rows.append((name, src_size, dst_size, status))
        if status != "ok":
            print(f"ERROR: size mismatch for {name}: src={src_size} dst={dst_size}")
            return 2

    # Summary table
    name_w = max(len(r[0]) for r in rows)
    print(f"{'file':<{name_w}}  {'size':>10}  status")
    print("-" * (name_w + 22))
    total = 0
    for name, src_size, _, status in rows:
        print(f"{name:<{name_w}}  {_human_size(src_size):>10}  {status}")
        total += src_size
    print("-" * (name_w + 22))
    print(f"{'TOTAL':<{name_w}}  {_human_size(total):>10}")
    print()
    print(f"copied {len(rows)} files into {dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
