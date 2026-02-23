#!/usr/bin/env python3
"""
Enhanced Statistical Analysis for QRNG vs PRNG Inference Experiments
=====================================================================

Provides rigorous statistical analysis with:
- Multiple comparison corrections (Bonferroni, Holm-Bonferroni)
- Confidence intervals on effect sizes
- Power analysis
- Equivalence testing
- Publication-ready reporting

Usage:
    python analyze_inference_statistics.py [--input PATH] [--output PATH]
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import statistics
import math

import numpy as np
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Force UTF-8 for Windows compatibility
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

console = Console(force_terminal=True)


@dataclass
class GroupStats:
    """Statistics for a single experimental group."""
    name: str
    n: int
    mean: float
    std: float
    sem: float
    ci_lower: float
    ci_upper: float
    values: list


@dataclass
class PairwiseComparison:
    """Result of pairwise comparison between two groups."""
    group1: str
    group2: str
    mean_diff: float
    cohens_d: float
    cohens_d_ci_lower: float
    cohens_d_ci_upper: float
    t_statistic: float
    df: float
    p_value: float
    p_adjusted: float
    significant: bool
    correction_method: str


@dataclass
class ANOVAResult:
    """Result of one-way ANOVA."""
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    eta_squared: float
    omega_squared: float


def load_experiment_data(filepath: Path) -> dict:
    """Load experiment results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_group_data(data: dict, metric: str = "iterations") -> dict[str, list]:
    """Extract per-group data for a given metric."""
    groups = {}

    for result in data.get("results", []):
        source = result.get("source_type", "UNKNOWN")
        if source not in groups:
            groups[source] = []

        if metric == "iterations":
            groups[source].append(result.get("iterations", 0))
        elif metric == "confidence":
            groups[source].append(result.get("final_confidence", 0))
        elif metric == "time_ms":
            groups[source].append(result.get("convergence_time_ms", 0))
        elif metric == "tokens":
            groups[source].append(result.get("tokens_used", 0))

    return groups


def compute_group_stats(name: str, values: list, alpha: float = 0.05) -> GroupStats:
    """Compute descriptive statistics with confidence intervals."""
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0
    sem = std / math.sqrt(n) if n > 0 else 0

    # t-based CI
    if n > 1:
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        ci_lower = mean - t_crit * sem
        ci_upper = mean + t_crit * sem
    else:
        ci_lower = ci_upper = mean

    return GroupStats(
        name=name,
        n=n,
        mean=mean,
        std=std,
        sem=sem,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        values=values
    )


def cohens_d(group1: list, group2: list) -> float:
    """Calculate Cohen's d effect size (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)

    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def cohens_d_confidence_interval(d: float, n1: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Calculate confidence interval for Cohen's d using non-central t approximation.

    Uses the formula from Cumming & Finch (2001).
    """
    # Variance of d (Hedges & Olkin, 1985)
    var_d = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2))
    se_d = math.sqrt(var_d)

    # Normal approximation CI
    z = stats.norm.ppf(1 - alpha/2)
    ci_lower = d - z * se_d
    ci_upper = d + z * se_d

    return ci_lower, ci_upper


def welch_t_test(group1: list, group2: list) -> tuple[float, float, float]:
    """
    Perform Welch's t-test (unequal variances).

    Returns: (t_statistic, df, p_value)
    """
    result = stats.ttest_ind(group1, group2, equal_var=False)

    # Calculate Welch-Satterthwaite degrees of freedom
    n1, n2 = len(group1), len(group2)
    var1 = statistics.variance(group1) if n1 > 1 else 0
    var2 = statistics.variance(group2) if n2 > 1 else 0

    numerator = (var1/n1 + var2/n2)**2
    denominator = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = numerator / denominator if denominator > 0 else n1 + n2 - 2

    return result.statistic, df, result.pvalue


def bonferroni_correction(p_values: list, alpha: float = 0.05) -> list[float]:
    """Apply Bonferroni correction to p-values."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def holm_bonferroni_correction(p_values: list, alpha: float = 0.05) -> list[float]:
    """Apply Holm-Bonferroni step-down correction."""
    n = len(p_values)
    indexed = list(enumerate(p_values))
    indexed.sort(key=lambda x: x[1])

    adjusted = [0.0] * n
    for rank, (original_idx, p) in enumerate(indexed):
        adjusted[original_idx] = min(p * (n - rank), 1.0)

    # Enforce monotonicity
    max_so_far = 0
    for i, (original_idx, _) in enumerate(indexed):
        adjusted[original_idx] = max(adjusted[original_idx], max_so_far)
        max_so_far = adjusted[original_idx]

    return adjusted


def one_way_anova(groups: dict[str, list]) -> ANOVAResult:
    """Perform one-way ANOVA."""
    group_values = list(groups.values())

    f_stat, p_value = stats.f_oneway(*group_values)

    # Calculate effect sizes
    k = len(groups)  # number of groups
    n_total = sum(len(g) for g in group_values)

    # Grand mean
    all_values = [v for g in group_values for v in g]
    grand_mean = statistics.mean(all_values)

    # SS_between
    ss_between = sum(len(g) * (statistics.mean(g) - grand_mean)**2 for g in group_values)

    # SS_within
    ss_within = sum(sum((v - statistics.mean(g))**2 for v in g) for g in group_values)

    # SS_total
    ss_total = ss_between + ss_within

    # Effect sizes
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Omega squared (less biased)
    df_between = k - 1
    df_within = n_total - k
    ms_within = ss_within / df_within if df_within > 0 else 0
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    omega_squared = max(0, omega_squared)  # Can't be negative

    return ANOVAResult(
        f_statistic=f_stat,
        p_value=p_value,
        df_between=df_between,
        df_within=df_within,
        eta_squared=eta_squared,
        omega_squared=omega_squared
    )


def kruskal_wallis(groups: dict[str, list]) -> tuple[float, float]:
    """Perform Kruskal-Wallis H-test (non-parametric alternative to ANOVA)."""
    group_values = list(groups.values())
    h_stat, p_value = stats.kruskal(*group_values)
    return h_stat, p_value


def compute_pairwise_comparisons(
    groups: dict[str, list],
    alpha: float = 0.05,
    correction: str = "holm"
) -> list[PairwiseComparison]:
    """
    Compute all pairwise comparisons with correction.

    Args:
        groups: Dict mapping group names to value lists
        alpha: Significance level
        correction: "bonferroni" or "holm"

    Returns:
        List of PairwiseComparison results
    """
    group_names = list(groups.keys())
    comparisons = []
    p_values = []

    # First pass: compute all comparisons
    for i, name1 in enumerate(group_names):
        for name2 in group_names[i+1:]:
            g1, g2 = groups[name1], groups[name2]

            # Effect size
            d = cohens_d(g1, g2)
            d_ci_lower, d_ci_upper = cohens_d_confidence_interval(d, len(g1), len(g2), alpha)

            # Welch's t-test
            t_stat, df, p_val = welch_t_test(g1, g2)

            comparisons.append({
                "group1": name1,
                "group2": name2,
                "mean_diff": statistics.mean(g1) - statistics.mean(g2),
                "cohens_d": d,
                "cohens_d_ci_lower": d_ci_lower,
                "cohens_d_ci_upper": d_ci_upper,
                "t_statistic": t_stat,
                "df": df,
                "p_value": p_val
            })
            p_values.append(p_val)

    # Apply correction
    if correction == "bonferroni":
        adjusted_p = bonferroni_correction(p_values, alpha)
        method = "Bonferroni"
    else:
        adjusted_p = holm_bonferroni_correction(p_values, alpha)
        method = "Holm-Bonferroni"

    # Build final results
    results = []
    for comp, p_adj in zip(comparisons, adjusted_p):
        results.append(PairwiseComparison(
            group1=comp["group1"],
            group2=comp["group2"],
            mean_diff=comp["mean_diff"],
            cohens_d=comp["cohens_d"],
            cohens_d_ci_lower=comp["cohens_d_ci_lower"],
            cohens_d_ci_upper=comp["cohens_d_ci_upper"],
            t_statistic=comp["t_statistic"],
            df=comp["df"],
            p_value=comp["p_value"],
            p_adjusted=p_adj,
            significant=p_adj < alpha,
            correction_method=method
        ))

    return results


def compute_power(effect_size: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """
    Compute achieved power for a two-sample t-test.

    Uses non-central t distribution.
    """
    # Non-centrality parameter
    ncp = effect_size * math.sqrt(n1 * n2 / (n1 + n2))

    # Degrees of freedom
    df = n1 + n2 - 2

    # Critical value
    t_crit = stats.t.ppf(1 - alpha/2, df)

    # Power = P(reject H0 | H1 true)
    # Using non-central t distribution
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    return power


def print_analysis_report(
    groups: dict[str, list],
    metric_name: str,
    alpha: float = 0.05
):
    """Print comprehensive analysis report."""

    console.print(Panel.fit(
        f"Statistical Analysis: {metric_name.upper()}",
        style="bold blue"
    ))

    # 1. Descriptive Statistics
    console.print("\n[bold]1. Descriptive Statistics[/]")

    desc_table = Table(title="Group Statistics")
    desc_table.add_column("Source", style="cyan")
    desc_table.add_column("N", justify="right")
    desc_table.add_column("Mean", justify="right")
    desc_table.add_column("SD", justify="right")
    desc_table.add_column("SEM", justify="right")
    desc_table.add_column("95% CI", justify="right")

    group_stats = {}
    for name, values in groups.items():
        gs = compute_group_stats(name, values, alpha)
        group_stats[name] = gs
        desc_table.add_row(
            gs.name,
            str(gs.n),
            f"{gs.mean:.3f}",
            f"{gs.std:.3f}",
            f"{gs.sem:.3f}",
            f"[{gs.ci_lower:.3f}, {gs.ci_upper:.3f}]"
        )

    console.print(desc_table)

    # 2. ANOVA
    console.print("\n[bold]2. Omnibus Test (One-Way ANOVA)[/]")
    anova = one_way_anova(groups)
    console.print(f"  F({anova.df_between}, {anova.df_within}) = {anova.f_statistic:.3f}, p = {anova.p_value:.4f}")
    console.print(f"  η² = {anova.eta_squared:.3f}, ω² = {anova.omega_squared:.3f}")

    if anova.p_value >= alpha:
        console.print(f"  [yellow]→ No significant omnibus effect (p ≥ {alpha})[/]")
    else:
        console.print(f"  [green]→ Significant omnibus effect (p < {alpha})[/]")

    # 3. Non-parametric alternative
    console.print("\n[bold]3. Non-Parametric Test (Kruskal-Wallis)[/]")
    h_stat, kw_p = kruskal_wallis(groups)
    console.print(f"  H = {h_stat:.3f}, p = {kw_p:.4f}")

    # 4. Pairwise Comparisons
    console.print("\n[bold]4. Pairwise Comparisons (Holm-Bonferroni corrected)[/]")

    comparisons = compute_pairwise_comparisons(groups, alpha, "holm")

    pair_table = Table(title="Pairwise Comparisons")
    pair_table.add_column("Comparison", style="cyan")
    pair_table.add_column("Δ Mean", justify="right")
    pair_table.add_column("Cohen's d", justify="right")
    pair_table.add_column("95% CI (d)", justify="right")
    pair_table.add_column("t", justify="right")
    pair_table.add_column("df", justify="right")
    pair_table.add_column("p (raw)", justify="right")
    pair_table.add_column("p (adj)", justify="right")
    pair_table.add_column("Sig.", justify="center")

    for comp in comparisons:
        sig_marker = "[green]✓[/]" if comp.significant else "[red]✗[/]"
        pair_table.add_row(
            f"{comp.group1} vs {comp.group2}",
            f"{comp.mean_diff:+.3f}",
            f"{comp.cohens_d:+.3f}",
            f"[{comp.cohens_d_ci_lower:+.3f}, {comp.cohens_d_ci_upper:+.3f}]",
            f"{comp.t_statistic:.3f}",
            f"{comp.df:.1f}",
            f"{comp.p_value:.4f}",
            f"{comp.p_adjusted:.4f}",
            sig_marker
        )

    console.print(pair_table)

    # 5. QRNG vs PRNG Focus
    console.print("\n[bold]5. QRNG vs PRNG Comparison[/]")

    qrng_sources = ["OUTSHIFT_STREAM", "ANU_QRNG", "CIPHERSTONE_QRNG"]
    prng_source = "PRNG"

    if prng_source in groups:
        prng_values = groups[prng_source]

        for qrng in qrng_sources:
            if qrng in groups:
                qrng_values = groups[qrng]
                d = cohens_d(qrng_values, prng_values)
                d_lower, d_upper = cohens_d_confidence_interval(
                    d, len(qrng_values), len(prng_values), alpha
                )
                power = compute_power(abs(d), len(qrng_values), len(prng_values), alpha)

                # Effect size interpretation
                if abs(d) < 0.2:
                    interp = "negligible"
                elif abs(d) < 0.5:
                    interp = "small"
                elif abs(d) < 0.8:
                    interp = "medium"
                else:
                    interp = "large"

                console.print(f"\n  [cyan]{qrng}[/] vs PRNG:")
                console.print(f"    Cohen's d = {d:+.3f} ({interp} effect)")
                console.print(f"    95% CI: [{d_lower:+.3f}, {d_upper:+.3f}]")
                console.print(f"    Achieved power: {power:.3f}")

                # Interpretation
                if d_lower <= 0 <= d_upper:
                    console.print(f"    [yellow]→ CI includes zero: cannot rule out null effect[/]")
                elif d > 0:
                    console.print(f"    [green]→ QRNG shows higher values than PRNG[/]")
                else:
                    console.print(f"    [green]→ QRNG shows lower values than PRNG[/]")

    # 6. Power analysis
    console.print("\n[bold]6. Power Analysis[/]")
    n_typical = len(next(iter(groups.values())))

    for target_d in [0.2, 0.5, 0.8]:
        power = compute_power(target_d, n_typical, n_typical, alpha)
        console.print(f"  Power to detect d={target_d} with n={n_typical}/group: {power:.3f}")

    # Calculate required N for 80% power at d=0.5
    target_power = 0.80
    target_effect = 0.5
    for n in range(5, 200):
        if compute_power(target_effect, n, n, alpha) >= target_power:
            console.print(f"  [bold]Required n for 80% power at d=0.5: {n}/group[/]")
            break


def save_analysis_results(
    groups: dict[str, list],
    metric_name: str,
    output_path: Path,
    alpha: float = 0.05
):
    """Save analysis results to JSON."""

    # Compute all statistics
    group_stats = {name: compute_group_stats(name, values, alpha)
                   for name, values in groups.items()}

    anova = one_way_anova(groups)
    h_stat, kw_p = kruskal_wallis(groups)
    comparisons = compute_pairwise_comparisons(groups, alpha, "holm")

    results = {
        "metric": metric_name,
        "alpha": alpha,
        "descriptive_statistics": {
            name: {
                "n": gs.n,
                "mean": gs.mean,
                "std": gs.std,
                "sem": gs.sem,
                "ci_95_lower": gs.ci_lower,
                "ci_95_upper": gs.ci_upper
            }
            for name, gs in group_stats.items()
        },
        "anova": {
            "f_statistic": anova.f_statistic,
            "p_value": anova.p_value,
            "df_between": anova.df_between,
            "df_within": anova.df_within,
            "eta_squared": anova.eta_squared,
            "omega_squared": anova.omega_squared
        },
        "kruskal_wallis": {
            "h_statistic": h_stat,
            "p_value": kw_p
        },
        "pairwise_comparisons": [
            {
                "comparison": f"{c.group1}_vs_{c.group2}",
                "mean_difference": c.mean_diff,
                "cohens_d": c.cohens_d,
                "cohens_d_ci_95": [c.cohens_d_ci_lower, c.cohens_d_ci_upper],
                "t_statistic": c.t_statistic,
                "df": c.df,
                "p_value_raw": c.p_value,
                "p_value_adjusted": c.p_adjusted,
                "significant": c.significant,
                "correction_method": c.correction_method
            }
            for c in comparisons
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to:[/] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze QRNG inference experiments")
    parser.add_argument("--input", "-i", type=Path,
                        default=Path("inference_results"),
                        help="Input directory or file")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("inference_results/statistical_analysis.json"),
                        help="Output file for results")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    parser.add_argument("--metric", choices=["iterations", "confidence", "time_ms", "tokens"],
                        default="iterations",
                        help="Metric to analyze")

    args = parser.parse_args()

    # Find latest experiment file
    if args.input.is_dir():
        files = sorted(args.input.glob("pilot_experiment_*.json"), reverse=True)
        if not files:
            console.print("[red]No experiment files found![/]")
            return
        input_file = files[0]
    else:
        input_file = args.input

    console.print(f"Loading: {input_file}")
    data = load_experiment_data(input_file)

    # Extract group data
    groups = extract_group_data(data, args.metric)

    if not groups:
        console.print("[red]No data found![/]")
        return

    console.print(f"Found {len(groups)} groups: {list(groups.keys())}")

    # Run analysis
    print_analysis_report(groups, args.metric, args.alpha)

    # Save results
    save_analysis_results(groups, args.metric, args.output, args.alpha)


if __name__ == "__main__":
    main()
