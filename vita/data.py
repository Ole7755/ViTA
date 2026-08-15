from __future__ import annotations

from collections import Counter
from io import BytesIO
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


class RandomJPEGCompression:
    """Round-trip a PIL image through JPEG to simulate camera compression artifacts."""

    def __init__(self, p: float = 0.35, quality: tuple[int, int] = (25, 85)):
        self.p = float(p)
        self.quality = (int(quality[0]), int(quality[1]))
        if not 1 <= self.quality[0] <= self.quality[1] <= 100:
            raise ValueError(f"JPEG quality must satisfy 1 <= min <= max <= 100, got {quality}")

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() > self.p:
            return image

        quality = random.randint(*self.quality)
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()


class RandomScalePad:
    """Shrink an image and paste it at a random position on an average-color canvas."""

    def __init__(self, p: float = 0.5, scale: tuple[float, float] = (0.3, 0.7)):
        self.p = float(p)
        self.scale = (float(scale[0]), float(scale[1]))
        if not 0 <= self.p <= 1:
            raise ValueError(f"Scale-pad probability must be between 0 and 1, got {p}")
        if not 0 < self.scale[0] <= self.scale[1] <= 1:
            raise ValueError(
                "Scale-pad range must satisfy 0 < min <= max <= 1, "
                f"got {scale}"
            )

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() > self.p:
            return image

        image = image.convert("RGB")
        width, height = image.size
        scale = random.uniform(*self.scale)
        scaled_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        scaled = image.resize(scaled_size, Image.Resampling.BILINEAR)

        # Eye crops are usually surrounded by skin. An image-derived fill color
        # avoids introducing the pure-black padding shortcut of affine transforms.
        fill = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
        canvas = Image.new("RGB", (width, height), color=fill)
        left = random.randint(0, width - scaled_size[0])
        top = random.randint(0, height - scaled_size[1])
        canvas.paste(scaled, (left, top))
        return canvas


class RandomDownsampleUpsample:
    """Temporarily reduce image resolution and resize back to simulate low-resolution input."""

    def __init__(self, p: float = 0.35, scale: tuple[float, float] = (0.4, 0.8)):
        self.p = float(p)
        self.scale = (float(scale[0]), float(scale[1]))
        if not 0 <= self.p <= 1:
            raise ValueError(f"Downsample probability must be between 0 and 1, got {p}")
        if not 0 < self.scale[0] <= self.scale[1] < 1:
            raise ValueError(
                "Downsample range must satisfy 0 < min <= max < 1, "
                f"got {scale}"
            )

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() > self.p:
            return image

        image = image.convert("RGB")
        width, height = image.size
        scale = random.uniform(*self.scale)
        reduced_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        reduced = image.resize(reduced_size, Image.Resampling.BOX)
        return reduced.resize((width, height), Image.Resampling.BILINEAR)


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


def _int_range_from_config(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    low, high = _range_from_config(value, default=default)
    return (int(round(low)), int(round(high)))


def _merge_augment_config(
    base_config: dict[str, Any] | None,
    class_config: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base_config or {})
    merged.update(class_config or {})
    return merged


def build_transform(image_size: int, train: bool, augment_config: dict[str, Any] | None = None):
    # Keep the deterministic resize contract explicit because the deployed
    # blink-call classifier must reproduce this preprocessing.
    resize = transforms.Resize(
        (image_size, image_size),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
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
        rotation_p = float(augment_config.get("rotation_p", 1.0))
        train_steps.append(
            transforms.RandomApply(
                [
                    transforms.RandomRotation(
                        degrees=(-rotation_degrees, rotation_degrees),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                ],
                p=rotation_p,
            )
        )

    train_steps.append(transforms.RandomHorizontalFlip(p=0.5))

    scale_pad_p = float(augment_config.get("scale_pad_p", 0))
    if scale_pad_p > 0:
        train_steps.append(
            RandomScalePad(
                p=scale_pad_p,
                scale=_range_from_config(
                    augment_config.get("scale_pad_scale"), default=(0.3, 0.7)
                ),
            )
        )

    downsample_p = float(augment_config.get("downsample_p", 0))
    if downsample_p > 0:
        train_steps.append(
            RandomDownsampleUpsample(
                p=downsample_p,
                scale=_range_from_config(
                    augment_config.get("downsample_scale"), default=(0.4, 0.8)
                ),
            )
        )

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

    gaussian_blur_p = float(augment_config.get("gaussian_blur_p", 0))
    if gaussian_blur_p > 0:
        kernel_size = int(augment_config.get("gaussian_blur_kernel_size", 7))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("gaussian_blur_kernel_size must be a positive odd integer")
        train_steps.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=kernel_size,
                        sigma=_range_from_config(
                            augment_config.get("gaussian_blur_sigma"), default=(0.1, 2.0)
                        ),
                    )
                ],
                p=gaussian_blur_p,
            )
        )

    jpeg_compression_p = float(augment_config.get("jpeg_compression_p", 0))
    if jpeg_compression_p > 0:
        train_steps.append(
            RandomJPEGCompression(
                p=jpeg_compression_p,
                quality=_int_range_from_config(
                    augment_config.get("jpeg_quality"), default=(25, 85)
                ),
            )
        )

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

    random_erasing_p = float(augment_config.get("random_erasing_p", 0))
    train_steps.extend(
        [
            transforms.ToTensor(),
            *(
                [
                    transforms.RandomErasing(
                        p=random_erasing_p,
                        scale=_range_from_config(
                            augment_config.get("random_erasing_scale"), default=(0.02, 0.08)
                        ),
                        ratio=_range_from_config(
                            augment_config.get("random_erasing_ratio"), default=(0.3, 3.3)
                        ),
                        value=augment_config.get("random_erasing_value", 0),
                    )
                ]
                if random_erasing_p > 0
                else []
            ),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(train_steps)


def build_transforms_by_label(
    cfg: dict[str, Any],
    class_map: dict[str, int],
    image_size: int,
) -> dict[int, Any]:
    base_augment = cfg["data"].get("train_augment", {})
    augment_by_class = cfg["data"].get("train_augment_by_class", {})
    return {
        int(label): build_transform(
            image_size=image_size,
            train=True,
            augment_config=_merge_augment_config(
                base_config=base_augment,
                class_config=augment_by_class.get(class_name, {}),
            ),
        )
        for class_name, label in class_map.items()
        if class_name in augment_by_class
    }


def build_datasets(cfg: dict[str, Any], split: str, train: bool = False) -> list[Dataset]:
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
    return datasets


def build_dataset(cfg: dict[str, Any], split: str, train: bool = False) -> Dataset:
    datasets = build_datasets(cfg, split=split, train=train)
    if not datasets:
        raise ValueError("No datasets configured under data.datasets")
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


def _build_loader(cfg: dict[str, Any], dataset: Dataset, train: bool) -> DataLoader:
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


def build_loader(cfg: dict[str, Any], split: str, train: bool = False) -> DataLoader:
    return _build_loader(cfg, build_dataset(cfg, split=split, train=train), train=train)


def build_loaders_by_dataset(
    cfg: dict[str, Any], split: str, train: bool = False
) -> list[DataLoader]:
    """Build one loader per configured dataset, preserving config order."""

    return [
        _build_loader(cfg, dataset, train=train)
        for dataset in build_datasets(cfg, split=split, train=train)
    ]
