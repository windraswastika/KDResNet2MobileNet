from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    max_epochs: int = 100
    early_stop_patience: int = 15


@dataclass(frozen=True)
class TrainHistory:
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    stopped_epoch: int = 0
    best_val_loss: float = float("inf")


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float,
    alpha: float,
) -> torch.Tensor:
    """Hinton-style combined loss: L = alpha*CE + (1-alpha)*T^2*KL(teacher||student).

    alpha weights the hard-label term; (1 - alpha) weights the soft-label
    (distillation) term, matching the code convention audited against the
    original notebook. The manuscript's combined-loss equation must be
    filled in with exactly this form (Phase E, item 18 in the revision plan).
    """
    ce = F.cross_entropy(student_logits, labels)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    return alpha * ce + (1 - alpha) * kd


def unfreeze_schedule(model: nn.Module, epoch: int, head_param_name: str = "fc") -> None:
    if epoch < 10:
        for name, param in model.named_parameters():
            param.requires_grad = head_param_name in name
    else:
        for param in model.parameters():
            param.requires_grad = True


def _run_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    train_step: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    val_step: Callable[[nn.Module, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    save_path: str,
    per_epoch_hook: Optional[Callable[[nn.Module, int], None]] = None,
    log_prefix: str = "",
) -> TrainHistory:
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)

    val_loss_hist: list[float] = []
    val_acc_hist: list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0
    stopped_epoch = cfg.max_epochs

    for epoch in range(1, cfg.max_epochs + 1):
        if per_epoch_hook is not None:
            per_epoch_hook(model, epoch)

        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = train_step(model, imgs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                loss, correct_batch = val_step(model, imgs, labels)
                val_loss_sum += loss.item()
                correct += int(correct_batch.item())

        val_loss = val_loss_sum / len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        scheduler.step(val_loss)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(f"{log_prefix}Epoch {epoch:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"{log_prefix}Early stopping at epoch {epoch}")
                stopped_epoch = epoch
                break

    return TrainHistory(
        val_loss=val_loss_hist,
        val_acc=val_acc_hist,
        stopped_epoch=stopped_epoch,
        best_val_loss=best_val_loss,
    )


def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
    cfg: TrainConfig = TrainConfig(),
) -> TrainHistory:
    criterion = nn.CrossEntropyLoss()

    def train_step(m: nn.Module, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return criterion(m(imgs), labels)

    def val_step(m: nn.Module, imgs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = m(imgs)
        return criterion(out, labels), (out.argmax(1) == labels).sum()

    return _run_loop(
        model, train_loader, val_loader, device, cfg,
        train_step, val_step, save_path,
        per_epoch_hook=lambda m, e: unfreeze_schedule(m, e),
        log_prefix="[teacher] ",
    )


def train_student_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
    cfg: TrainConfig = TrainConfig(),
) -> TrainHistory:
    """Plain fine-tuning of the student architecture, no teacher involved.

    This is the MobileNetV3-Small-without-KD baseline requested by both
    reviewers (Reviewer A point 4, Reviewer E point 3) — it isolates the
    architecture/transfer-learning effect from the knowledge-transfer effect.
    """
    criterion = nn.CrossEntropyLoss()

    def train_step(m: nn.Module, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return criterion(m(imgs), labels)

    def val_step(m: nn.Module, imgs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = m(imgs)
        return criterion(out, labels), (out.argmax(1) == labels).sum()

    return _run_loop(
        model, train_loader, val_loader, device, cfg,
        train_step, val_step, save_path,
        log_prefix="[student-baseline] ",
    )


def train_student_kd(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
    T: float,
    alpha: float,
    cfg: TrainConfig = TrainConfig(),
) -> TrainHistory:
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    def train_step(m: nn.Module, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_logits = teacher(imgs)
        s_logits = m(imgs)
        return kd_loss(s_logits, t_logits, labels, T, alpha)

    def val_step(m: nn.Module, imgs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = m(imgs)
        return F.cross_entropy(out, labels), (out.argmax(1) == labels).sum()

    return _run_loop(
        student, train_loader, val_loader, device, cfg,
        train_step, val_step, save_path,
        log_prefix=f"[student-kd T={T} a={alpha}] ",
    )
