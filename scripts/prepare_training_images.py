from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


def convert_one(job: tuple[Path, Path, int, int]) -> tuple[str, str]:
    source, destination, long_edge, quality = job
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, "JPEG", quality=quality, subsampling=0, optimize=True)
        return str(source), "ok"
    except Exception as error:
        return str(source), f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--long-edge", type=int, default=512)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    sources = sorted(args.data_root.glob("recovered_archive*/[0-9]*/CAPorg*.jpg"))
    jobs = [
        (
            source,
            args.output_root / source.relative_to(args.data_root),
            args.long_edge,
            args.quality,
        )
        for source in sources
    ]
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(convert_one, jobs):
            if result[1] != "ok":
                errors.append(result)
    print({"sources": len(sources), "errors": len(errors)})
    for source, error in errors:
        print(source, error)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
