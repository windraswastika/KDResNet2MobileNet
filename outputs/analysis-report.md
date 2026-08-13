# Phase A Results — Strict Analysis (real, corrected pipeline)

Generated from `run/pipeline/run_evaluation.py` against 20 checkpoints (5 seeds ×
{teacher, student-baseline-no-KD, student-KD, student-KD-no-denoise}). Test partition
is fixed (`split_seed=42`) and identical across all seeds/arms. Raw data:
`per_seed_metrics.csv`, `table1_multiseed_summary.csv`, `statistical_tests.json`,
`per_class_metrics_pooled.csv`, `table2_efficiency.csv`, `grid_search_results.csv`,
`fig5_confusion_matrices_pooled.png`.

## Comparison questions (locked before analysis)

1. Does student-KD match teacher performance? (primary claim of the manuscript)
2. Does KD improve over a plain fine-tuned MobileNetV3-Small (no-KD baseline)? (the
   missing comparison Reviewers A/E both required — this is what actually tells us
   whether "knowledge transfer" is doing anything)
3. Does denoising contribute to student-KD performance?
4. What are the real per-class recall/specificity, especially malignant?
5. What is the real efficiency trade-off?

Primary metric: macro F1 (matches manuscript's stated primary metric). Unit of
analysis: seed (n=5, paired across arms — same 5 seeds, same fixed test set).

## 1. Descriptive statistics (Table 1, real, 5 seeds, fixed test set)

| Arm | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Teacher (ResNet-50) | 83.59% ± 4.25% | 82.90% ± 2.93% | 83.07% ± 7.37% | 82.41% ± 5.62% |
| Student, no-KD (baseline) | 82.74% ± 1.64% | 82.46% ± 2.38% | 81.78% ± 1.94% | **81.88% ± 2.17%** |
| Student, KD (main arm) | 82.05% ± 1.60% | 82.42% ± 1.61% | 80.10% ± 2.46% | 80.93% ± 2.04% |
| Student, KD, no denoise | 81.37% ± 0.72% | 80.23% ± 1.52% | 80.39% ± 1.22% | 80.11% ± 1.11% |

## 2. Headline finding — the manuscript's central claim is not supported

**Student-KD does not outperform the no-KD baseline.** Across every metric, the plain
fine-tuned MobileNetV3-Small (no teacher involved) scores *equal to or higher than*
the KD-trained student (F1 81.88% vs 80.93%; accuracy 82.74% vs 82.05%). This is the
opposite of what the manuscript needs to claim, and it is exactly the comparison
Reviewer A (point 4) and Reviewer E (point 3) said was missing and "essential... not
[to be] postponed to future work."

- Wilcoxon (baseline vs KD, F1): statistic=5.0, **p=0.625**, effect size r=0.30 (n=5
  pairs). Not statistically significant — but the *direction* is against KD, not for it.
- With only 5 seeds this cannot be called "no difference" either — power is too low
  (min achievable p=0.0312). The honest statement is: **no evidence that KD helps, and
  the point estimate favors not using KD.**

Also note the manuscript's second claim — "student KD slightly exceeds teacher"
(+0.34% acc, +0.65% F1 in the original text) — is **reversed** here: the teacher now
outperforms student-KD by ~1.5–1.9 points on every metric.

- Wilcoxon (teacher vs student-KD, F1): statistic=4.0, p=0.4375, effect size r=0.42.
- TOST equivalence (margin ±5pp, F1): **p=0.0498 < 0.05 → equivalence confirmed**, but
  only just under the threshold, and the confirmed claim is "student is within 5 points
  of teacher," not "student matches or exceeds teacher." Report it as: performance is
  practically comparable, on the small side of the pre-registered margin.

## 3. Variance / stability claim

Teacher F1 std (5.62%) is numerically much higher than any student variant (~1.1–2.2%),
consistent with the manuscript's "KD as implicit regularizer" narrative *in direction*.
However:

- Levene's test (teacher vs student-KD): statistic=2.36, **p=0.163 — not significant**.
  With n=5 per group this test has very low power; the variance difference is
  suggestive, not statistically established. State it as an observation, not a proven
  effect, and flag the small-n limitation explicitly if kept in the Discussion.

## 4. Denoising ablation (C9)

Student-KD-with-denoising (F1 80.93%) vs student-KD-without-denoising (F1 80.11%):
a small positive gap, consistent direction with the manuscript's motivation for
denoising, but **not statistically significant** (Wilcoxon p=0.4375, r=0.42, n=5).
Report as "a small, non-significant improvement" — do not claim denoising is proven
necessary from this evidence alone.

## 5. Per-class recall / specificity (pooled over 5 seeds, 117×5 test predictions)

| Arm | Malignant recall | Malignant recall 95% CI |
|---|---|---|
| Teacher | 73.5% | [66.1%, 79.9%] |
| Student, no-KD | 66.5% | [58.7%, 73.4%] |
| Student, KD | 65.8% | [58.0%, 72.8%] |
| Student, KD, no denoise | 67.1% | [59.4%, 74.0%] |

Malignant is the lowest-recall class in every arm, consistent with the manuscript's
narrative about under-representation (26.9%) driving this. But **the teacher has
meaningfully higher malignant recall than any student variant** — this cuts against
any claim of the student being clinically preferable, and should temper the
"clinically deployable" language regardless of the separate overclaiming issue
Reviewer E raised (C7). Full recall/specificity table with CIs for all 3 classes ×
4 arms: `per_class_metrics_pooled.csv`.

## 6. Efficiency (Table 2, real)

| Model | Params (M) | Inference (ms) |
|---|---|---|
| Teacher (ResNet-50) | 23.51 | 9.27 |
| Student (MobileNetV3-Sm.) | 1.52 | 3.71 |

Ratio: 15.5× fewer parameters (matches manuscript exactly), 2.50× faster inference
(manuscript claimed 1.84×; single-run timing on MPS, expect run-to-run variance —
don't over-interpret the exact ratio, but the qualitative "much faster" claim holds).

## 7. Grid search (real, 12 combinations, seed-0 teacher)

Best: **T=2, α=0.5** (88.89% val acc), tied with T=4, α=0.7 (88.89%). Full table in
`grid_search_results.csv`. There is **no monotonic trend with temperature** — this
directly contradicts the manuscript's claimed narrative ("higher temperatures tend to
yield better performance... highly diffuse soft targets are more informative"). That
paragraph in the Discussion needs to be rewritten to match this table, not the old
narrative built around T=8.

## 8. Limitations / blockers to flag explicitly

- n=5 seeds throughout — every non-significant Wilcoxon result is underpowered, not
  evidence of equivalence (min p=0.0625 with 5 pairs, consistent with the manuscript's
  own stated limitation).
- Grid search was run once (seed-0 teacher only), not per-seed — if reviewers want
  per-seed grid search this analysis does not cover that; current manuscript text
  should describe the once-only protocol accurately, matching what was actually run.
- Inference timing is a single measurement pass per model (100 runs, 10 warm-up
  discarded), not repeated across seeds — treat Table 2 latency as approximate.
- TOST margin (±5pp) is a placeholder, not yet justified against a clinically
  meaningful threshold — the equivalence conclusion in §2 is only as good as this
  margin choice.
- Patient-level leakage (Reviewer E point 1) is unresolved by this analysis — BUSI's
  public release has no recoverable patient identifier (confirmed by inspecting
  `raw/` filenames), so this remains a stated dataset limitation, not something these
  numbers can correct for.


