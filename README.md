# Knowledge Distillation: ResNet-50 → MobileNetV3-Small (BUSI Breast Ultrasound)

A Knowledge Distillation (KD) pipeline compressing a ResNet-50 teacher into a
MobileNetV3-Small student for three-class breast ultrasound classification (BUSI
dataset), with a real no-KD baseline, a denoising ablation, 5-seed evaluation, and a
TOST equivalence test.

## Status

**Headline finding**: the KD student is statistically equivalent to the teacher (TOST,
±5pp margin, p=0.0498) but does **not** show a statistically significant advantage over
a plain fine-tuned MobileNetV3-Small trained without a teacher (Wilcoxon p=0.625). Full
analysis: [`outputs/analysis-report.md`](outputs/analysis-report.md).

## Project layout

```
src/
├── data_module/      # dataset, stratified split, TV-Chambolle denoising + cache
├── model_module/      # teacher/student factory+registry
├── trainer_module/    # training loops: teacher, student-KD, student-baseline (no-KD)
├── eval_module/        # metrics, Wilcoxon, TOST equivalence, Levene's test
└── utils/              # seeding

run/
├── conf/               # Hydra configs (data/train/grid-search/seeds)
└── pipeline/
    ├── run_multiseed.py      # orchestrates teacher/baseline/KD/denoise-ablation x 5 seeds
    ├── run_grid_search.py    # T x alpha grid search (once, seed-0 teacher)
    └── run_evaluation.py     # Table 1/2, Wilcoxon+TOST+Levene, per-class metrics, figures

outputs/                 # real experiment results (metrics, stats, figures, hydra logs)
implementasi_kd_breast_ultrasound.ipynb      # original exploratory notebook
```

Not included in this repo (see `.gitignore`): the BUSI raw image dataset (`raw/`), the
denoised-image cache (`raw_denoised/`), and trained model checkpoints (`checkpoints/`,
~540MB) — all are regenerable by running the pipeline below against your own copy of
[BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

## Reproducing the results

```bash
uv sync
# place the BUSI dataset at raw/{benign,malignant,normal}/*.png
uv run python3 run/pipeline/run_multiseed.py
uv run python3 run/pipeline/run_evaluation.py
```

Default config (`run/conf/config.yaml`): 5 evaluation seeds (0–4), fixed split seed 42,
12-combination grid search (T ∈ {2,4,6,8}, α ∈ {0.3,0.5,0.7}), denoising ablation on.

## Key documents

- [`outputs/analysis-report.md`](outputs/analysis-report.md) — strict statistical analysis of the real results.
