import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from data_module import CLASS_NAMES


def measure_inference_time(model: nn.Module, device: torch.device, n_runs: int = 100) -> float:
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times[10:]))


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_preds), np.array(all_labels)


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — appropriate for small-n proportions (test set n=117)."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - adj) / denom, (centre + adj) / denom)


def per_class_recall_specificity(
    labels: np.ndarray, preds: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per-class recall (sensitivity) and specificity with Wilson CIs.

    Addresses Reviewer E point 8: macro metrics alone hide malignant-class
    performance, which is the clinically critical category.
    """
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    out: dict[str, dict[str, float]] = {}
    total = cm.sum()
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        recall_lo, recall_hi = wilson_ci(int(tp), int(tp + fn))
        spec_lo, spec_hi = wilson_ci(int(tn), int(tn + fp))

        out[name] = {
            "recall": float(recall),
            "recall_ci_lo": recall_lo,
            "recall_ci_hi": recall_hi,
            "specificity": float(specificity),
            "specificity_ci_lo": spec_lo,
            "specificity_ci_hi": spec_hi,
        }
    return out


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
