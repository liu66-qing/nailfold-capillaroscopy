from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path


MAGIC = b"NFD1"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def create(base: Path, target: Path, output: Path, block_size: int) -> None:
    changed = []
    with base.open("rb") as left, target.open("rb") as right:
        index = 0
        while True:
            old = left.read(block_size)
            new = right.read(block_size)
            if not new:
                break
            if old != new:
                changed.append((index, zlib.compress(new, level=9)))
            index += 1
    header = {
        "block_size": block_size,
        "target_size": target.stat().st_size,
        "base_sha256": digest(base),
        "target_sha256": digest(target),
        "changed_blocks": len(changed),
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with output.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(encoded)))
        stream.write(encoded)
        for index, data in changed:
            stream.write(struct.pack("<II", index, len(data)))
            stream.write(data)
    print(json.dumps({**header, "delta_bytes": output.stat().st_size}))


def apply(base: Path, delta: Path, output: Path) -> None:
    with delta.open("rb") as stream:
        if stream.read(4) != MAGIC:
            raise ValueError("invalid delta magic")
        header = json.loads(stream.read(struct.unpack("<I", stream.read(4))[0]))
        if digest(base) != header["base_sha256"]:
            raise ValueError("base SHA-256 mismatch")
        with base.open("rb") as source, output.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        with output.open("r+b") as target:
            for _ in range(header["changed_blocks"]):
                index, length = struct.unpack("<II", stream.read(8))
                data = zlib.decompress(stream.read(length))
                target.seek(index * header["block_size"])
                target.write(data)
            target.truncate(header["target_size"])
    if digest(output) != header["target_sha256"]:
        raise ValueError("reconstructed target SHA-256 mismatch")
    print(json.dumps(header))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    creator = subparsers.add_parser("create")
    creator.add_argument("--base", type=Path, required=True)
    creator.add_argument("--target", type=Path, required=True)
    creator.add_argument("--output", type=Path, required=True)
    creator.add_argument("--block-size", type=int, default=65536)
    applier = subparsers.add_parser("apply")
    applier.add_argument("--base", type=Path, required=True)
    applier.add_argument("--delta", type=Path, required=True)
    applier.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        create(args.base, args.target, args.output, args.block_size)
    else:
        apply(args.base, args.delta, args.output)


if __name__ == "__main__":
    main()
