from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CLASS_TO_LABEL = {"closed": 0, "open": 1, "irrelevant": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced 3-class eye-state dataset using MRL and FoodSeg103."
    )
    parser.add_argument("--mrl-root", default="dataset/mrl", help="MRL dataset root.")
    parser.add_argument(
        "--foodseg-root",
        default="dataset/FoodSeg103/Images/img_dir",
        help="FoodSeg103 image root containing train/ and test/.",
    )
    parser.add_argument("--output", default="dataset/eye3_foodseg", help="Output dataset root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foodseg-val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--method",
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help="How to materialize images in the output dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove output if it exists.")
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing directory: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def materialize(src: Path, dst: Path, method: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if method == "copy":
        shutil.copy2(src, dst)
    elif method == "hardlink":
        dst.hardlink_to(src)
    else:
        dst.symlink_to(src.resolve())


def add_samples(
    *,
    rows: list[dict[str, str | int]],
    sources: list[Path],
    output_root: Path,
    split: str,
    class_name: str,
    method: str,
) -> None:
    label = CLASS_TO_LABEL[class_name]
    for index, src in enumerate(sources):
        dst = output_root / split / class_name / f"{split}_{class_name}_{index:06d}{src.suffix.lower()}"
        materialize(src, dst, method)
        rows.append(
            {
                "split": split,
                "class_name": class_name,
                "label": label,
                "source_path": str(src),
                "target_path": str(dst),
            }
        )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    mrl_root = Path(args.mrl_root)
    foodseg_root = Path(args.foodseg_root)
    output_root = Path(args.output)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite to rebuild: {output_root}")
        shutil.rmtree(output_root)

    mrl_sources = {
        "train": {
            "closed": list_images(mrl_root / "train" / "sleepy"),
            "open": list_images(mrl_root / "train" / "awake"),
        },
        "val": {
            "closed": list_images(mrl_root / "val" / "sleepy"),
            "open": list_images(mrl_root / "val" / "awake"),
        },
        "test": {
            "closed": list_images(mrl_root / "test" / "sleepy"),
            "open": list_images(mrl_root / "test" / "awake"),
        },
    }
    foodseg_train = list_images(foodseg_root / "train")
    foodseg_test = list_images(foodseg_root / "test")

    rng.shuffle(foodseg_train)
    rng.shuffle(foodseg_test)
    foodseg_val_count = round(len(foodseg_train) * args.foodseg_val_ratio)
    split_counts = {
        "train": len(foodseg_train) - foodseg_val_count,
        "val": foodseg_val_count,
        "test": len(foodseg_test),
    }
    irrelevant_sources = {
        "train": foodseg_train[: split_counts["train"]],
        "val": foodseg_train[split_counts["train"] :],
        "test": foodseg_test,
    }

    rows: list[dict[str, str | int]] = []
    summary: dict[str, dict[str, int]] = {}
    for split, count in split_counts.items():
        summary[split] = {}
        for class_name in ("closed", "open"):
            candidates = list(mrl_sources[split][class_name])
            if len(candidates) < count:
                raise RuntimeError(
                    f"Not enough MRL {split}/{class_name} images: need {count}, have {len(candidates)}"
                )
            rng.shuffle(candidates)
            selected = candidates[:count]
            add_samples(
                rows=rows,
                sources=selected,
                output_root=output_root,
                split=split,
                class_name=class_name,
                method=args.method,
            )
            summary[split][class_name] = len(selected)

        selected_irrelevant = irrelevant_sources[split]
        add_samples(
            rows=rows,
            sources=selected_irrelevant,
            output_root=output_root,
            split=split,
            class_name="irrelevant",
            method=args.method,
        )
        summary[split]["irrelevant"] = len(selected_irrelevant)

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "class_name", "label", "source_path", "target_path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "class_to_label": CLASS_TO_LABEL,
        "seed": args.seed,
        "method": args.method,
        "mrl_root": str(mrl_root),
        "foodseg_root": str(foodseg_root),
        "foodseg_val_ratio": args.foodseg_val_ratio,
        "summary": summary,
        "total_images": len(rows),
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
