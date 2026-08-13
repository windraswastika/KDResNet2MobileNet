import json
import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data_module import CLASS_NAMES, get_dataloaders  # noqa: E402
from eval_module import (  # noqa: E402
    compute_metrics,
    evaluate_model,
    measure_inference_time,
    per_class_recall_specificity,
    tost_equivalence,
    variance_stability_test,
    wilcoxon_test,
)
from model_module import count_params, get_student, get_teacher  # noqa: E402
from run_grid_search import get_device  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


ARMS = {
    "teacher": ("teacher_seed{}.pth", get_teacher),
    "student_baseline": ("student_baseline_seed{}.pth", get_student),
    "student_kd": ("student_kd_seed{}.pth", get_student),
    "student_kd_nodenoise": ("student_kd_nodenoise_seed{}.pth", get_student),
}


def multi_seed_metrics(cfg: DictConfig, device: torch.device, arm: str, test_loader) -> pd.DataFrame:
    template, model_fn = ARMS[arm]
    rows = []
    for seed in cfg.seeds:
        path = os.path.join(cfg.checkpoint_dir, template.format(seed))
        if not os.path.exists(path):
            print(f"[missing] {path}")
            continue
        model = model_fn().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        preds, labels = evaluate_model(model, test_loader, device)
        m = compute_metrics(labels, preds)
        m["seed"] = seed
        m["arm"] = arm
        rows.append(m)
    return pd.DataFrame(rows)


def _test_loader_for(cfg: DictConfig, arm: str):
    """student_kd_nodenoise was trained on non-denoised images — it must be
    evaluated on a non-denoised test set too, or the comparison is invalid."""
    apply_denoise = arm != "student_kd_nodenoise"
    _, _, loader = get_dataloaders(
        cfg.root_dir, batch_size=cfg.data.batch_size,
        split_seed=cfg.split_seed, run_seed=None, num_workers=cfg.data.num_workers,
        apply_denoise=apply_denoise,
    )
    return loader


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device()
    os.makedirs(cfg.output_dir, exist_ok=True)
    test_loader = _test_loader_for(cfg, "teacher")  # denoised (default) test set
    test_loader_nodenoise = _test_loader_for(cfg, "student_kd_nodenoise")

    # ---- Table 1: multi-seed classification performance -------------------
    dfs = [
        multi_seed_metrics(
            cfg, device, arm,
            test_loader_nodenoise if arm == "student_kd_nodenoise" else test_loader,
        )
        for arm in ARMS
    ]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(os.path.join(cfg.output_dir, "per_seed_metrics.csv"), index=False)

    summary = df_all.groupby("arm")[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(cfg.output_dir, "table1_multiseed_summary.csv"))
    print("\n=== Table 1: Multi-seed classification performance ===")
    print(summary)

    # ---- Statistical tests: teacher vs student_kd --------------------------
    teacher_f1 = df_all[df_all.arm == "teacher"].sort_values("seed")["f1"].to_numpy()
    kd_f1 = df_all[df_all.arm == "student_kd"].sort_values("seed")["f1"].to_numpy()
    baseline_f1 = df_all[df_all.arm == "student_baseline"].sort_values("seed")["f1"].to_numpy()

    stats_report = {}
    if len(teacher_f1) == len(kd_f1) and len(teacher_f1) > 0:
        wres = wilcoxon_test(teacher_f1, kd_f1)
        tost = tost_equivalence(teacher_f1, kd_f1, margin=cfg.equivalence_margin)
        var_test = variance_stability_test(teacher_f1, kd_f1)
        stats_report["teacher_vs_student_kd"] = {
            "wilcoxon": {"statistic": wres.statistic, "pvalue": wres.pvalue, "effect_size_r": wres.effect_size_r,
                         "interpretation": wres.interpretation()},
            "tost_equivalence": {"pvalue": tost.pvalue, "margin": cfg.equivalence_margin,
                                  "mean_diff": tost.mean_diff, "interpretation": tost.interpretation()},
            "variance_stability": var_test,
        }
        print("\n=== Wilcoxon (teacher vs student_kd, F1) ===")
        print(wres.interpretation())
        print("\n=== TOST equivalence (teacher vs student_kd, F1) ===")
        print(tost.interpretation())

    if len(baseline_f1) == len(kd_f1) and len(kd_f1) > 0:
        wres_ablation = wilcoxon_test(baseline_f1, kd_f1)
        stats_report["student_baseline_vs_student_kd"] = {
            "wilcoxon": {"statistic": wres_ablation.statistic, "pvalue": wres_ablation.pvalue,
                         "effect_size_r": wres_ablation.effect_size_r,
                         "interpretation": wres_ablation.interpretation()},
        }
        print("\n=== Wilcoxon (student_baseline vs student_kd, F1) — isolates KD contribution ===")
        print(wres_ablation.interpretation())

    nodenoise_f1 = df_all[df_all.arm == "student_kd_nodenoise"].sort_values("seed")["f1"].to_numpy()
    if len(nodenoise_f1) == len(kd_f1) and len(kd_f1) > 0:
        wres_denoise = wilcoxon_test(nodenoise_f1, kd_f1)
        stats_report["student_kd_nodenoise_vs_student_kd"] = {
            "wilcoxon": {"statistic": wres_denoise.statistic, "pvalue": wres_denoise.pvalue,
                         "effect_size_r": wres_denoise.effect_size_r,
                         "interpretation": wres_denoise.interpretation()},
        }
        print("\n=== Wilcoxon (student_kd_nodenoise vs student_kd, F1) — isolates denoising contribution ===")
        print(wres_denoise.interpretation())

    with open(os.path.join(cfg.output_dir, "statistical_tests.json"), "w") as f:
        json.dump(stats_report, f, indent=2)

    # ---- Table 2: efficiency ------------------------------------------------
    teacher_eval = get_teacher().to(device)
    student_eval = get_student().to(device)
    eff_rows = [
        {"model": "Teacher (ResNet-50)", "params_M": count_params(teacher_eval),
         "inference_ms": measure_inference_time(teacher_eval, device)},
        {"model": "Student (MobileNetV3-Sm.)", "params_M": count_params(student_eval),
         "inference_ms": measure_inference_time(student_eval, device)},
    ]
    df_eff = pd.DataFrame(eff_rows)
    df_eff.to_csv(os.path.join(cfg.output_dir, "table2_efficiency.csv"), index=False)
    print("\n=== Table 2: Efficiency ===")
    print(df_eff)

    def pooled_preds_labels(arm: str) -> tuple[list[int], list[int]]:
        template, model_fn = ARMS[arm]
        loader = test_loader_nodenoise if arm == "student_kd_nodenoise" else test_loader
        pooled_preds, pooled_labels = [], []
        for seed in cfg.seeds:
            path = os.path.join(cfg.checkpoint_dir, template.format(seed))
            if not os.path.exists(path):
                continue
            model = model_fn().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            preds, labels = evaluate_model(model, loader, device)
            pooled_preds.extend(preds.tolist())
            pooled_labels.extend(labels.tolist())
        return pooled_preds, pooled_labels

    # ---- Per-class recall/specificity (pooled across seeds) ----------------
    per_class_rows = []
    for arm in ARMS:
        pooled_preds, pooled_labels = pooled_preds_labels(arm)
        if not pooled_preds:
            continue
        pc = per_class_recall_specificity(np.array(pooled_labels), np.array(pooled_preds))
        for cls_name, vals in pc.items():
            per_class_rows.append({"arm": arm, "class": cls_name, **vals})
    pd.DataFrame(per_class_rows).to_csv(os.path.join(cfg.output_dir, "per_class_metrics_pooled.csv"), index=False)

    # ---- Figure 5: confusion matrices, POOLED across seeds (not one run) ----
    fig, axes = plt.subplots(1, len(ARMS), figsize=(6 * len(ARMS), 5))
    for ax, arm in zip(axes, ARMS):
        pooled_preds, pooled_labels = pooled_preds_labels(arm)
        if not pooled_preds:
            continue
        cm = confusion_matrix(pooled_labels, pooled_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{arm} (pooled over {len(cfg.seeds)} seeds)")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "fig5_confusion_matrices_pooled.png"), dpi=150)
    plt.close()

    print(f"\nAll outputs written to {cfg.output_dir}/")


if __name__ == "__main__":
    main()
