#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ci95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(n))


def _sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(n))


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = sub["success"].dropna()
        row = dict(zip(group_cols, keys))
        row.update(
            n=int(values.count()),
            mean_success=float(values.mean()) if len(values) else np.nan,
            sem_success=_sem(values),
            ci95_success=_ci95(values),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(clean: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    keys = ["env", "seed"]
    for env, sub in clean.groupby("env"):
        pivot = sub.pivot_table(
            index=keys, columns="variant", values="success", aggfunc="mean"
        )
        if baseline not in pivot:
            continue
        for variant in sorted(c for c in pivot.columns if c != baseline):
            diff = (pivot[variant] - pivot[baseline]).dropna()
            if diff.empty:
                continue
            rows.append(
                {
                    "env": env,
                    "baseline": baseline,
                    "variant": variant,
                    "n_pairs": int(diff.count()),
                    "mean_diff": float(diff.mean()),
                    "sem_diff": _sem(diff),
                    "ci95_diff": _ci95(diff),
                    "wins": int((diff > 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "ties": int((diff == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results/results.csv")
    ap.add_argument("--outdir", default="outputs/figs")
    ap.add_argument("--baseline", default="base")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    for col, default in (
        ("success", np.nan),
        ("proprio_noise", 0.0),
        ("tactile_dropout", 0.0),
        ("mass_scale", 1.0),
        ("friction_scale", 1.0),
        ("seed", np.nan),
    ):
        if col not in df:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df = df[df["success"].notna()].copy()

    sensory = df[(df["mass_scale"] == 1.0) & (df["friction_scale"] == 1.0)].copy()
    clean = sensory[
        (sensory["proprio_noise"] == 0.0) & (sensory["tactile_dropout"] == 0.0)
    ].copy()

    if clean.empty:
        print(f"wrote figures to: {outdir}")
        return

    clean_summary = _summary(clean, ["env", "variant"]).sort_values(["env", "variant"])
    clean_summary.to_csv(outdir / "clean_success_summary.csv", index=False)
    paired = _paired(clean, args.baseline)
    paired.to_csv(outdir / "paired_success_comparisons.csv", index=False)

    envs = sorted(clean_summary["env"].unique())
    variants = sorted(clean_summary["variant"].unique())
    x = np.arange(len(envs), dtype=float)
    width = min(0.8 / max(len(variants), 1), 0.25)

    plt.figure(figsize=(max(7, len(envs) * 1.2), 4.8))
    for i, variant in enumerate(variants):
        vals = []
        errs = []
        xpos = x + (i - (len(variants) - 1) / 2) * width
        for env in envs:
            row = clean_summary[
                (clean_summary["env"] == env) & (clean_summary["variant"] == variant)
            ]
            vals.append(float(row["mean_success"].iloc[0]) if not row.empty else 0.0)
            errs.append(float(row["ci95_success"].iloc[0]) if not row.empty else 0.0)
        plt.bar(xpos, vals, width=width, yerr=errs, capsize=3, label=variant, alpha=0.8)
        for env_idx, env in enumerate(envs):
            pts = clean[(clean["env"] == env) & (clean["variant"] == variant)]
            if pts.empty:
                continue
            jitter = np.linspace(-0.25, 0.25, len(pts)) * width
            plt.scatter(
                np.full(len(pts), xpos[env_idx]) + jitter,
                pts["success"],
                s=16,
                color="black",
                alpha=0.55,
                linewidths=0,
            )

    plt.xticks(x, envs, rotation=30, ha="right")
    plt.ylabel("Success")
    plt.title("Clean success by task and variant")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "clean_success_bar.png", dpi=200)
    plt.close()

    by_ev = sensory.groupby(["env", "variant"], dropna=False)
    for (env, variant), sub in by_ev:
        summary = _summary(sub, ["proprio_noise", "tactile_dropout"])
        if summary.empty:
            continue
        drops = sorted(summary["tactile_dropout"].unique())
        plt.figure(figsize=(6.0, 4.2))
        for drop in drops:
            pts = summary[summary["tactile_dropout"] == drop].sort_values("proprio_noise")
            plt.errorbar(
                pts["proprio_noise"],
                pts["mean_success"],
                yerr=pts["ci95_success"],
                marker="o",
                capsize=3,
                label=f"drop={drop}",
            )

        plt.xlabel("Proprio noise std")
        plt.ylabel("Success")
        plt.title(f"{env} / {variant} robustness")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / f"{env}_{variant}_robustness.png", dpi=200)
        plt.close()

    print(f"wrote figures to: {outdir}")


if __name__ == "__main__":
    main()
