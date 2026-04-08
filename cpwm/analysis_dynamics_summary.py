#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
        .mean()
        .reset_index()
        .sort_values(["env", "variant", "mass_scale", "friction_scale"])
        .reset_index(drop=True)
    )
    if not summary.empty:
        summary["success"] = summary["success"].round(args.decimals)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
