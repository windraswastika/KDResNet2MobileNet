import json
import os
import sys
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data_module import get_dataloaders  # noqa: E402
from model_module import get_student, get_teacher  # noqa: E402
from trainer_module import kd_loss  # noqa: E402
from utils import set_seed  # noqa: E402


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_grid_search(cfg: DictConfig, device: torch.device, teacher_path: str) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(0)
    train_loader, val_loader, _ = get_dataloaders(
        cfg.root_dir,
        batch_size=cfg.data.batch_size,
        split_seed=cfg.split_seed,
        run_seed=0,
        num_workers=cfg.data.num_workers,
    )

    teacher = get_teacher().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    results = []
    for T in cfg.grid_search.temperatures:
        for alpha in cfg.grid_search.alphas:
            set_seed(0)
            student = get_student().to(device)
            optimizer = torch.optim.Adam(student.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

            for _ in range(cfg.grid_search.epochs):
                student.train()
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    with torch.no_grad():
                        t_logits = teacher(imgs)
                    s_logits = student(imgs)
                    loss = kd_loss(s_logits, t_logits, labels, T, alpha)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            student.eval()
            correct = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += (student(imgs).argmax(1) == labels).sum().item()
            val_acc = correct / len(val_loader.dataset)
            results.append({"T": T, "alpha": alpha, "val_acc": val_acc})
            print(f"T={T} alpha={alpha} val_acc={val_acc:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.output_dir, "grid_search_results.csv"), index=False)

    best = df.loc[df["val_acc"].idxmax()]
    best_hparams = {"T": int(best["T"]), "alpha": float(best["alpha"]), "val_acc": float(best["val_acc"])}
    with open(os.path.join(cfg.output_dir, "best_hparams.json"), "w") as f:
        json.dump(best_hparams, f, indent=2)

    print(f"\nBest: T={best_hparams['T']}, alpha={best_hparams['alpha']}, val_acc={best_hparams['val_acc']:.4f}")
    return best_hparams


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device()
    teacher_path = os.path.join(cfg.checkpoint_dir, "teacher_seed0.pth")
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"{teacher_path} not found. Run run_multiseed.py teacher training "
            "for seed 0 first (grid search uses the seed-0 teacher checkpoint)."
        )
    run_grid_search(cfg, device, teacher_path)


if __name__ == "__main__":
    main()
