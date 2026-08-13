import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data_module import get_dataloaders  # noqa: E402
from model_module import get_student, get_teacher  # noqa: E402
from trainer_module import TrainConfig, train_student_baseline, train_student_kd, train_teacher  # noqa: E402
from run_grid_search import get_device, run_grid_search  # noqa: E402


def _train_cfg(cfg: DictConfig) -> TrainConfig:
    return TrainConfig(
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        scheduler_factor=cfg.train.scheduler_factor,
        scheduler_patience=cfg.train.scheduler_patience,
        max_epochs=cfg.train.max_epochs,
        early_stop_patience=cfg.train.early_stop_patience,
    )


def _save_history(history, path: str) -> None:
    with open(path, "w") as f:
        json.dump(asdict(history), f, indent=2)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    tcfg = _train_cfg(cfg)

    # ---- Phase 1: teacher, all seeds -----------------------------------
    for seed in cfg.seeds:
        path = os.path.join(cfg.checkpoint_dir, f"teacher_seed{seed}.pth")
        if os.path.exists(path):
            print(f"[skip] {path} exists")
            continue
        from utils import set_seed

        set_seed(seed)
        train_loader, val_loader, _ = get_dataloaders(
            cfg.root_dir, batch_size=cfg.data.batch_size,
            split_seed=cfg.split_seed, run_seed=seed, num_workers=cfg.data.num_workers,
        )
        teacher = get_teacher().to(device)
        history = train_teacher(teacher, train_loader, val_loader, device, path, tcfg)
        _save_history(history, os.path.join(cfg.output_dir, f"history_teacher_seed{seed}.json"))

    # ---- Phase 2: grid search (once, seed-0 teacher) --------------------
    best_hparams_path = os.path.join(cfg.output_dir, "best_hparams.json")
    if os.path.exists(best_hparams_path):
        with open(best_hparams_path) as f:
            best_hparams = json.load(f)
        print(f"[skip] grid search already done: {best_hparams}")
    else:
        teacher0_path = os.path.join(cfg.checkpoint_dir, "teacher_seed0.pth")
        best_hparams = run_grid_search(cfg, device, teacher0_path)
    best_T, best_alpha = best_hparams["T"], best_hparams["alpha"]

    # ---- Phase 3: student baseline (no KD), all seeds --------------------
    for seed in cfg.seeds:
        path = os.path.join(cfg.checkpoint_dir, f"student_baseline_seed{seed}.pth")
        if os.path.exists(path):
            print(f"[skip] {path} exists")
            continue
        from utils import set_seed

        set_seed(seed)
        train_loader, val_loader, _ = get_dataloaders(
            cfg.root_dir, batch_size=cfg.data.batch_size,
            split_seed=cfg.split_seed, run_seed=seed, num_workers=cfg.data.num_workers,
        )
        student = get_student().to(device)
        history = train_student_baseline(student, train_loader, val_loader, device, path, tcfg)
        _save_history(history, os.path.join(cfg.output_dir, f"history_student_baseline_seed{seed}.json"))

    # ---- Phase 4: student KD, all seeds ----------------------------------
    for seed in cfg.seeds:
        path = os.path.join(cfg.checkpoint_dir, f"student_kd_seed{seed}.pth")
        if os.path.exists(path):
            print(f"[skip] {path} exists")
            continue
        from utils import set_seed

        set_seed(seed)
        teacher_path = os.path.join(cfg.checkpoint_dir, f"teacher_seed{seed}.pth")
        train_loader, val_loader, _ = get_dataloaders(
            cfg.root_dir, batch_size=cfg.data.batch_size,
            split_seed=cfg.split_seed, run_seed=seed, num_workers=cfg.data.num_workers,
        )
        teacher = get_teacher().to(device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        student = get_student().to(device)
        history = train_student_kd(
            student, teacher, train_loader, val_loader, device, path,
            T=best_T, alpha=best_alpha, cfg=tcfg,
        )
        _save_history(history, os.path.join(cfg.output_dir, f"history_student_kd_seed{seed}.json"))

    # ---- Phase 5: denoising ablation (student KD, denoise off), all seeds -
    if cfg.run_denoise_ablation:
        for seed in cfg.seeds:
            path = os.path.join(cfg.checkpoint_dir, f"student_kd_nodenoise_seed{seed}.pth")
            if os.path.exists(path):
                print(f"[skip] {path} exists")
                continue
            from utils import set_seed

            set_seed(seed)
            teacher_path = os.path.join(cfg.checkpoint_dir, f"teacher_seed{seed}.pth")
            train_loader, val_loader, _ = get_dataloaders(
                cfg.root_dir, batch_size=cfg.data.batch_size,
                split_seed=cfg.split_seed, run_seed=seed, num_workers=cfg.data.num_workers,
                apply_denoise=False,
            )
            teacher = get_teacher().to(device)
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))
            student = get_student().to(device)
            history = train_student_kd(
                student, teacher, train_loader, val_loader, device, path,
                T=best_T, alpha=best_alpha, cfg=tcfg,
            )
            _save_history(history, os.path.join(cfg.output_dir, f"history_student_kd_nodenoise_seed{seed}.json"))

    print("\nAll Phase A runs complete.")


if __name__ == "__main__":
    main()
