from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nailfold_report.data.manifest import build_manifest, write_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases, files = build_manifest(args.data_root.resolve())
    write_manifest(cases, files, args.output_dir.resolve())
    print(f"wrote {len(cases)} exam directories and {len(files)} file records")


if __name__ == "__main__":
    main()
