from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from nailfold_report.print_report import render_report_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(render_report_file(args.report_json, args.output).resolve())


if __name__ == "__main__":
    main()
