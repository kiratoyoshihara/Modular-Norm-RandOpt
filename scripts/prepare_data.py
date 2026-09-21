#!/usr/bin/env python3
"""Prepare canonical inputs from the upstream processed-data archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_if_missing(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing input differs; refusing to overwrite: {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)


def split_countdown(source, output):
    rows = json.loads(source.read_text())
    if len(rows) != 2200:
        raise ValueError("Countdown source must contain 2,200 examples")
    random.Random(42).shuffle(rows)
    for split, subset in (("train", rows[:200]), ("validation", rows[200:700]), ("test", rows[700:])):
        # This serialization also reproduces the file checksums used by ES.
        content = json.dumps(subset, indent=2, ensure_ascii=False).encode()
        write_if_missing(output / f"countdown_{split}.json", content)


def prepare_mbpp(root):
    from datasets import DatasetDict, load_from_disk
    source = load_from_disk(str(root / "mbpp_full"))
    for directory, split, dataset in (("mbpp_train_200", "train", source["train"].select(range(200))),
                                       ("mbpp_test", "test", source["test"])):
        target = root / directory
        if not target.exists():
            DatasetDict({split: dataset}).save_to_disk(str(target))
        else:
            existing = load_from_disk(str(target))
            if existing[split].to_list() != dataset.to_list():
                raise ValueError(f"Existing MBPP split differs: {target}")


def check(root, tasks):
    manifest = json.loads((ROOT / "configs/data_manifest.json").read_text())
    for task in tasks:
        for relative, expected in manifest[task].items():
            path = root / relative
            if expected.startswith("records:"):
                from datasets import load_from_disk
                data = load_from_disk(str(path))
                rows = {key: data[key].to_list() for key in data}
                actual = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
                if actual != expected.removeprefix("records:"):
                    raise ValueError(f"Dataset content mismatch: {path}")
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Missing input: {path}; obtain the processed-data archive linked in README")
            if sha(path) != expected:
                raise ValueError(f"Input checksum mismatch: {path}")
    print(f"Verified: {', '.join(tasks)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--source", type=Path, help="Unpacked upstream data directory; copy only requested task files")
    p.add_argument("--tasks", default="countdown,gsm8k", help="Comma-separated task names, or all")
    p.add_argument("--download-countdown", action="store_true", help="Download only the public Countdown source")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    root = args.data_root.resolve()
    manifest = json.loads((ROOT / "configs/data_manifest.json").read_text())
    tasks = list(manifest) if args.tasks == "all" else args.tasks.split(",")
    if not tasks or any(task not in manifest for task in tasks):
        p.error(f"Tasks must be drawn from {list(manifest)}")
    try:
        if not args.check_only:
            if args.source:
                # Copy only whitelisted scientific inputs, never arbitrary archive files.
                source = args.source.resolve()
                names = {name for task in tasks for name in manifest[task]}
                if "countdown" in tasks:
                    names.add("countdown/countdown.json")
                for name in sorted(names):
                    origin, target = source / name, root / name
                    if origin.is_file():
                        write_if_missing(target, origin.read_bytes())
                if "mbpp" in tasks and (source / "mbpp_full").is_dir() and not (root / "mbpp_full").exists():
                    shutil.copytree(source / "mbpp_full", root / "mbpp_full")
            if args.download_countdown:
                source = root / "countdown/countdown.json"
                if not source.exists():
                    url = "https://raw.githubusercontent.com/VsonicV/es-at-scale/574a9d134da1ffce2a8bb812019899e5c96b588a/archive/countdown/data/countdown.json"
                    with urllib.request.urlopen(url, timeout=60) as response:
                        payload = response.read(10 * 2**20)
                    expected = json.loads((ROOT / "configs/data_sources.json").read_text())["countdown_sha256"]
                    if hashlib.sha256(payload).hexdigest() != expected:
                        raise ValueError("Downloaded Countdown source differs from the expected version")
                    write_if_missing(source, payload)
            if "countdown" in tasks:
                paths = [root / f"countdown/countdown_{split}.json" for split in ("train", "validation", "test")]
                if not all(path.is_file() for path in paths):
                    split_countdown(root / "countdown/countdown.json", root / "countdown")
            if "mbpp" in tasks:
                prepare_mbpp(root)
        check(root, tasks)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
