#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def _sem(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _ci95(values):
    return 1.96 * _sem(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results/results.csv")
    ap.add_argument("--out", default="outputs/results/dynamics_summary.csv")
    ap.add_argument("--decimals", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "mass_scale" not in df:
        df["mass_scale"] = 1.0
    if "friction_scale" not in df:
        df["friction_scale"] = 1.0
    for col in ("success", "mass_scale", "friction_scale"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["success", "mass_scale", "friction_scale"])

    dyn = df[(df["mass_scale"] != 1.0) | (df["friction_scale"] != 1.0)].copy()
    summary = (
        dyn.groupby(["env", "variant", "mass_scale", "friction_scale"])["success"]
        .agg(success="mean", n="count", sem=_sem, ci95=_ci95)
        .reset_index()
        .sort_values(["env", "variant", "mass_scale", "friction_scale"])
        .reset_index(drop=True)
    )
    if not summary.empty:
        for col in ("success", "sem", "ci95"):
            summary[col] = summary[col].round(args.decimals)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
