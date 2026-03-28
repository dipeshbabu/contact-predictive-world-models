#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results/results.csv")
    ap.add_argument("--out", default="outputs/results/contact_analysis.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["success"].notna()].copy()
    df["success"] = pd.to_numeric(df["success"], errors="coerce")

    clean = df[(df["proprio_noise"] == 0.0) & (df["tactile_dropout"] == 0.0)]
    pert = df[(df["proprio_noise"] > 0.0) | (df["tactile_dropout"] > 0.0)]

    clean_mean = clean.groupby(["env", "variant"])["success"].mean().reset_index().rename(columns={"success": "clean_success"})
    pert_mean = pert.groupby(["env", "variant"])["success"].mean().reset_index().rename(columns={"success": "perturbed_success"})

    merged = clean_mean.merge(pert_mean, on=["env", "variant"], how="outer")
    merged["robustness_gap"] = merged["clean_success"] - merged["perturbed_success"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
