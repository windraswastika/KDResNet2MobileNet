from dataclasses import dataclass

import numpy as np
from scipy.stats import levene, wilcoxon
from statsmodels.stats.weightstats import ttost_paired


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    pvalue: float
    effect_size_r: float
    n_pairs: int

    def interpretation(self, alpha: float = 0.05) -> str:
        """Correct, non-overclaiming interpretation (Reviewer E point 4).

        A non-significant Wilcoxon result means "no statistically significant
        difference was detected" — it does NOT establish equivalence. Use
        `tost_equivalence` below if an equivalence claim is actually needed.
        """
        if self.pvalue < alpha:
            return f"Statistically significant difference detected (p={self.pvalue:.4f} < {alpha})."
        return (
            f"No statistically significant difference was detected "
            f"(p={self.pvalue:.4f} >= {alpha}). This does NOT establish "
            f"equivalence; with n={self.n_pairs} paired seeds the minimum "
            f"achievable p-value is {2 ** (-self.n_pairs):.4f}, so statistical "
            f"power is limited. See TOST equivalence test for an equivalence claim."
        )


@dataclass(frozen=True)
class TOSTResult:
    pvalue: float
    low: float
    upp: float
    mean_diff: float

    def interpretation(self, alpha: float = 0.05) -> str:
        if self.pvalue < alpha:
            return (
                f"Equivalence confirmed within margin [{self.low}, {self.upp}] "
                f"(TOST p={self.pvalue:.4f} < {alpha}, mean diff={self.mean_diff:.4f})."
            )
        return (
            f"Equivalence NOT confirmed within margin [{self.low}, {self.upp}] "
            f"(TOST p={self.pvalue:.4f} >= {alpha}, mean diff={self.mean_diff:.4f}). "
            "Do not claim equivalence based on this result."
        )


def wilcoxon_test(baseline: np.ndarray, proposed: np.ndarray) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test with a matched-pairs rank-biserial effect size."""
    stat, p = wilcoxon(baseline, proposed)
    n = len(baseline)
    # z from normal approximation, then r = z / sqrt(n) (Rosenthal, 1991).
    _, p_approx = wilcoxon(baseline, proposed, mode="approx")
    from scipy.stats import norm

    z = norm.ppf(1 - p_approx / 2) if p_approx > 0 else 0.0
    effect_size_r = z / np.sqrt(n) if n > 0 else float("nan")
    return WilcoxonResult(statistic=float(stat), pvalue=float(p), effect_size_r=float(effect_size_r), n_pairs=n)


def tost_equivalence(baseline: np.ndarray, proposed: np.ndarray, margin: float) -> TOSTResult:
    """Two one-sided tests (TOST) for equivalence, requested by the user in
    place of relying on Wilcoxon non-significance for the equivalence claim.

    `margin` is the pre-defined equivalence bound (e.g. 0.05 = 5 percentage
    points of accuracy/F1). The equivalence interval is [-margin, +margin]
    on (baseline - proposed). This choice of margin must be justified in the
    manuscript (e.g., smallest clinically meaningful accuracy difference),
    not left implicit.
    """
    pvalue, *_ = ttost_paired(baseline, proposed, -margin, margin)
    return TOSTResult(
        pvalue=float(pvalue),
        low=-margin,
        upp=margin,
        mean_diff=float(np.mean(baseline) - np.mean(proposed)),
    )


def variance_stability_test(baseline: np.ndarray, proposed: np.ndarray) -> dict[str, float]:
    """Levene's test comparing dispersion of the two seed-level metric
    distributions — statistical backing for any "lower variance = more
    stable" claim (Reviewer E point 8), instead of eyeballing std devs.
    """
    stat, p = levene(baseline, proposed)
    return {
        "levene_statistic": float(stat),
        "levene_pvalue": float(p),
        "baseline_std": float(np.std(baseline, ddof=1)),
        "proposed_std": float(np.std(proposed, ddof=1)),
    }
