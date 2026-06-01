from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
from typing import Any

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS


class RandomBluePurpleTint:
    """Apply a mild blue/purple camera tint to an RGB PIL image."""

    def __init__(self, p: float = 0.25, strength: tuple[float, float] = (0.08, 0.22)):
        self.p = float(p)
        self.strength = (float(strength[0]), float(strength[1]))

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() > self.p:
            return image

        strength = random.uniform(*self.strength)
        image = image.convert("RGB")
        red, green, blue = image.split()
        red = red.point(lambda value: min(255, int(value * (1.0 + strength * 0.75))))
        green = green.point(lambda value: max(0, int(value * (1.0 - strength * 0.35))))
        blue = blue.point(lambda value: min(255, int(value * (1.0 + strength))))
        return Image.merge("RGB", (red, green, blue))


class EyeImageFolder(Dataset):
    """ImageFolder-style dataset with explicit class-to-label mapping."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        class_map: dict[str, int],
        transform=None,
        transforms_by_label: dict[int, Any] | None = None,
    ):
        self.root = Path(root) / split
        self.transform = transform
        self.transforms_by_label = transforms_by_label or {}
        self.samples: list[tuple[Path, int]] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Missing split directory: {self.root}")

        for class_dir, label in class_map.items():
            folder = self.root / class_dir
            if not folder.exists():
                raise FileNotFoundError(f"Missing class directory: {folder}")
            for path in sorted(folder.iterdir()):
                if is_image_file(path):
                    self.samples.append((path, int(label)))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        transform = self.transforms_by_label.get(label, self.transform)
        if transform is not None:
            image = transform(image)
        return image, label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]


def _range_from_config(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return (number, number)
    if len(value) != 2:
        raise ValueError(f"Expected a 2-value range, got: {value}")
    return (float(value[0]), float(value[1]))


def build_transform(image_size: int, train: bool, augment_config: dict[str, Any] | None = None):
    resize = transforms.Resize((image_size, image_size), antialias=True)
    common = [
        resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    if not train:
        return transforms.Compose(common)

    augment_config = augment_config or {}
    train_steps: list[Any] = [resize]

    rotation_degrees = float(augment_config.get("rotation_degrees", 0))
    if rotation_degrees > 0:
        train_steps.append(
            transforms.RandomRotation(
                degrees=(-rotation_degrees, rotation_degrees),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    train_steps.append(transforms.RandomHorizontalFlip(p=0.5))

    blue_purple_p = float(augment_config.get("blue_purple_p", 0))
    if blue_purple_p > 0:
        train_steps.append(
            RandomBluePurpleTint(
                p=blue_purple_p,
                strength=_range_from_config(
                    augment_config.get("blue_purple_strength"),
                    default=(0.08, 0.22),
                ),
            )
        )

    grayscale_p = float(augment_config.get("grayscale_p", 0))
    if grayscale_p > 0:
        train_steps.append(transforms.RandomGrayscale(p=grayscale_p))

    low_contrast_p = float(augment_config.get("low_contrast_p", 0))
    if low_contrast_p > 0:
        train_steps.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=_range_from_config(
                            augment_config.get("brightness"), default=(0.75, 1.15)
                        ),
                        contrast=_range_from_config(
                            augment_config.get("contrast"), default=(0.45, 1.0)
                        ),
                        saturation=_range_from_config(
                            augment_config.get("saturation"), default=(0.0, 1.0)
                        ),
                    )
                ],
                p=low_contrast_p,
            )
        )

    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(train_steps)


def build_transforms_by_label(
    cfg: dict[str, Any],
    class_map: dict[str, int],
    image_size: int,
) -> dict[int, Any]:
    augment_by_class = cfg["data"].get("train_augment_by_class", {})
    return {
        int(label): build_transform(
            image_size=image_size,
            train=True,
            augment_config=augment_by_class.get(class_name, {}),
        )
        for class_name, label in class_map.items()
        if class_name in augment_by_class
    }


def build_dataset(cfg: dict[str, Any], split: str, train: bool = False) -> Dataset:
    image_size = int(cfg["data"].get("image_size", 224))
    transform = build_transform(
        image_size=image_size,
        train=train,
        augment_config=cfg["data"].get("train_augment"),
    )
    datasets = []
    for item in cfg["data"]["datasets"]:
        class_map = item["class_map"]
        transforms_by_label = (
            build_transforms_by_label(cfg, class_map=class_map, image_size=image_size)
            if train
            else None
        )
        datasets.append(
            EyeImageFolder(
                root=item["root"],
                split=split,
                class_map=class_map,
                transform=transform,
                transforms_by_label=transforms_by_label,
            )
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def label_counts(dataset: Dataset, num_classes: int = 2) -> list[int]:
    labels: list[int] = []
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            labels.extend(getattr(child, "labels"))
    else:
        labels.extend(getattr(dataset, "labels"))
    counter = Counter(labels)
    return [counter.get(i, 0) for i in range(num_classes)]


def build_loader(cfg: dict[str, Any], split: str, train: bool = False) -> DataLoader:
    dataset = build_dataset(cfg, split=split, train=train)
    batch_size_key = "batch_size" if train else "eval_batch_size"
    batch_size = int(cfg["data"].get(batch_size_key, cfg["data"].get("batch_size", 64)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        drop_last=False,
    )
