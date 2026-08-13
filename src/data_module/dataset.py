import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import torch
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_MAP = {"benign": 0, "malignant": 1, "normal": 2}
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Reviewer E, point 1 (patient-level leakage) ---------------------------
# The public BUSI release (Al-Dhabyani et al., 2020) ships images named only
# "<class> (<index>).png" with no patient identifier, examination date, or
# any other field linking images to the 600 source patients. We verified
# this directly against the raw/ directory: filenames carry no recoverable
# grouping key. A patient-wise/group-aware split is therefore not
# implementable from the public data as distributed, and this is stated as
# an explicit limitation in the manuscript rather than silently using an
# image-level split. See plan/revision-plan-infotel.md, open question 1.
PATIENT_ID_RECOVERABLE = False


def load_paths_labels(root_dir: str) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    for cls_name, cls_idx in CLASS_MAP.items():
        cls_dir = os.path.join(root_dir, cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.endswith(".png") and "mask" not in fname:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    return paths, labels


def speckle_reduce(img_array: np.ndarray) -> np.ndarray:
    img_float = img_array.astype(np.float32) / 255.0
    denoised = denoise_tv_chambolle(img_float, weight=0.1, channel_axis=-1)
    return (denoised * 255).astype(np.uint8)


def _denoise_and_save(src_path: str, dst_path: str) -> None:
    if os.path.exists(dst_path):
        return
    img = Image.open(src_path).convert("RGB")
    denoised = Image.fromarray(speckle_reduce(np.array(img)))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    denoised.save(dst_path)


def ensure_denoised_cache(root_dir: str, cache_dir: str, max_workers: int = 8) -> str:
    """Pre-cache TV-Chambolle-denoised images to disk.

    `denoise_tv_chambolle` costs ~180ms/image on CPU; recomputing it in
    every __getitem__ call turns a ~35-50 min run into ~7 hours (documented
    in the original notebook's own comments). Precompute once, reuse the
    cache across all seeds/epochs.
    """
    paths, _ = load_paths_labels(root_dir)
    jobs = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for src in paths:
            rel = os.path.relpath(src, root_dir)
            dst = os.path.join(cache_dir, rel)
            jobs.append(pool.submit(_denoise_and_save, src, dst))
        for _ in as_completed(jobs):
            pass
    return cache_dir


class BUSIDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
        apply_denoise: bool = True,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.apply_denoise = apply_denoise

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.apply_denoise:
            img = Image.fromarray(speckle_reduce(np.array(img)))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_transforms(is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    split_seed: int = 42,
    run_seed: Optional[int] = None,
    num_workers: int = 0,
    apply_denoise: bool = True,
    denoise_cache_dir: Optional[str] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders.

    `split_seed` fixes the stratified 70/15/15 image-level partition — this
    is held IDENTICAL across all 5 evaluation seeds, so the test set is the
    same set of images in every run (Reviewer E, point 2). `run_seed`, if
    given, seeds the DataLoader shuffling generator so that only the
    training order (and, upstream, model init) varies across evaluation
    seeds — the partition itself never changes.

    When `apply_denoise` is True, images are read from a pre-cached
    denoised copy (built once via `ensure_denoised_cache`) rather than
    denoised on-the-fly per __getitem__ call — see `ensure_denoised_cache`
    docstring for why.
    """
    read_dir = root_dir
    if apply_denoise:
        cache_dir = denoise_cache_dir or f"{root_dir.rstrip('/')}_denoised"
        ensure_denoised_cache(root_dir, cache_dir)
        read_dir = cache_dir

    paths, labels = load_paths_labels(read_dir)

    idx = list(range(len(paths)))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, labels, test_size=0.30, stratify=labels, random_state=split_seed
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=split_seed
    )

    def subset(indices: list[int], is_train: bool) -> BUSIDataset:
        p = [paths[i] for i in indices]
        l = [labels[i] for i in indices]
        # Denoising, if requested, was already applied when building the
        # cache above (or read_dir == root_dir and none is applied) — the
        # Dataset itself must never denoise again here.
        return BUSIDataset(p, l, transform=get_transforms(is_train), apply_denoise=False)

    train_ds = subset(idx_train, True)
    val_ds = subset(idx_val, False)
    test_ds = subset(idx_test, False)

    generator = None
    if run_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(run_seed)

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
