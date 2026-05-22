#!/usr/bin/env python3
"""Create DocNLC pretraining filelist from a Hybrid dataset folder.

Expected folder structure:

Hybrid/
├── GT/
└── Degraded/
    ├── Blur/
    ├── Noise/
    ├── Shadow/
    ├── Watermark/
    └── WithBack/

Each output line has the format expected by data/multitask_dataset.py:

GT|WithBack|Blur|Noise|Shadow|Watermark
"""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

BLUR_SUFFIXES = ["_b0", "_b1", "_b2"]
NOISE_SUFFIXES = ["_n0.1", "_n0.2", "_n0.05"]
SHADOW_SUFFIXES = ["_s0", "_s1", "_s2"]
WATERMARK_SUFFIXES = ["_w0", "_w1", "_w2"]


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )


def stem_map(folder: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in image_files(folder):
        if path.stem in mapping:
            duplicates.append(path.stem)
        mapping[path.stem] = path
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate stems in {folder}: {sample}")
    return mapping


def infer_gt_path(withback_stem: str, gt_by_stem: dict[str, Path], gt_stems_by_len: list[str]) -> Path | None:
    """Match a WithBack stem to the longest GT stem prefix.

    The original generator creates WithBack names by appending a background id to
    the GT stem, so longest-prefix matching is more robust than splitting on '_'.
    """
    for gt_stem in gt_stems_by_len:
        if withback_stem == gt_stem or withback_stem.startswith(gt_stem + "_"):
            return gt_by_stem[gt_stem]
    return None


def make_filelist(root: Path, output: Path, strict: bool = False) -> tuple[int, list[str]]:
    gt_dir = root / "GT"
    degraded_dir = root / "Degraded"
    withback_dir = degraded_dir / "WithBack"
    blur_dir = degraded_dir / "Blur"
    noise_dir = degraded_dir / "Noise"
    shadow_dir = degraded_dir / "Shadow"
    watermark_dir = degraded_dir / "Watermark"

    required_dirs = [gt_dir, withback_dir, blur_dir, noise_dir, shadow_dir, watermark_dir]
    missing_dirs = [str(p) for p in required_dirs if not p.is_dir()]
    if missing_dirs:
        raise FileNotFoundError("Missing required folders:\n" + "\n".join(missing_dirs))

    gt_by_stem = stem_map(gt_dir)
    blur_by_stem = stem_map(blur_dir)
    noise_by_stem = stem_map(noise_dir)
    shadow_by_stem = stem_map(shadow_dir)
    watermark_by_stem = stem_map(watermark_dir)
    withback_files = image_files(withback_dir)
    gt_stems_by_len = sorted(gt_by_stem, key=len, reverse=True)

    lines: list[str] = []
    warnings: list[str] = []

    for withback_path in withback_files:
        base = withback_path.stem
        gt_path = infer_gt_path(base, gt_by_stem, gt_stems_by_len)
        if gt_path is None:
            warnings.append(f"No GT matched for WithBack: {withback_path.name}")
            continue

        for idx in range(3):
            expected = {
                "Blur": blur_by_stem.get(base + BLUR_SUFFIXES[idx]),
                "Noise": noise_by_stem.get(base + NOISE_SUFFIXES[idx]),
                "Shadow": shadow_by_stem.get(base + SHADOW_SUFFIXES[idx]),
                "Watermark": watermark_by_stem.get(base + WATERMARK_SUFFIXES[idx]),
            }
            missing = [name for name, path in expected.items() if path is None]
            if missing:
                warnings.append(f"Missing {', '.join(missing)} for WithBack variant {base}, index {idx}")
                continue

            paths = [
                gt_path,
                withback_path,
                expected["Blur"],
                expected["Noise"],
                expected["Shadow"],
                expected["Watermark"],
            ]
            lines.append("|".join(str(p.resolve()) for p in paths))

    if strict and warnings:
        sample = "\n".join(warnings[:20])
        raise RuntimeError(f"Found {len(warnings)} matching problems. Sample:\n{sample}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines), warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Path to Hybrid dataset root.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pretraining_data_path.txt"),
        help="Output txt path. Default: ./pretraining_data_path.txt",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any expected match is missing.")
    args = parser.parse_args()

    total, warnings = make_filelist(args.root, args.output, strict=args.strict)
    print(f"Wrote {total} lines to {args.output}")
    if warnings:
        print(f"Skipped {len(warnings)} incomplete matches. First examples:")
        for warning in warnings[:20]:
            print("  -", warning)


if __name__ == "__main__":
    main()
