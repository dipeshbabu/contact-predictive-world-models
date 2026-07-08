#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

KNOWN_VARIANTS = (
    "future1_noact",
    "proprio_only",
    "ppo_proprio",
    "ppo_tactile",
    "aux_contact",
    "future1",
    "future3",
    "future5",
    "masked",
    "tactile_group",
    "contact_frontier",
    "frontier_rgb",
    "frontier",
    "contact_onset",
    "contact_change",
    "both_onset",
    "both_change",
    "aux_contact_onset",
    "aux_contact_change",
    "contact1",
    "contact3",
    "current",
    "proprio",
    "contact",
    "noact",
    "recon",
    "both",
    "base",
    "aux",
)


def _sem(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _ci95(values):
    return 1.96 * _sem(values)


def read_last_probe_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return {}
    last = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            for key, value in row.items():
                if key.startswith("report/contact_probe/") or key.startswith(
                    "report_eval/contact_probe/"
                ):
                    try:
                        metric = key.replace("report/contact_probe/", "probe_")
                        metric = metric.replace("report_eval/contact_probe/", "probe_eval_")
                        last[metric] = float(value)
                    except Exception:
                        pass
    return last


def infer_run_fields(run_dir: str):
    name = Path(str(run_dir)).name
    match = re.match(r"(.+)_s(\d+)$", name)
    if not match:
        return None
    stem, seed = match.group(1), int(match.group(2))
    for variant in sorted(KNOWN_VARIANTS, key=len, reverse=True):
        suffix = f"_{variant}"
        if stem.endswith(suffix):
            return {
                "env": stem[: -len(suffix)],
                "variant": variant,
                "seed": seed,
            }
    env, variant = stem.rsplit("_", 1)
    return {
        "env": env,
        "variant": variant,
        "seed": seed,
    }


def norm_path(value):
    if value is None:
        return ""
    text = str(value)
    if not text or text == "nan":
        return ""
    return os.path.normcase(os.path.abspath(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results/results.csv")
    ap.add_argument("--out", default="outputs/results/contact_analysis.csv")
    ap.add_argument("--decimals", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "mass_scale" not in df:
        df["mass_scale"] = 1.0
    if "friction_scale" not in df:
        df["friction_scale"] = 1.0
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    df["proprio_noise"] = pd.to_numeric(df["proprio_noise"], errors="coerce").fillna(0.0)
    df["tactile_dropout"] = pd.to_numeric(df["tactile_dropout"], errors="coerce").fillna(0.0)
    df["mass_scale"] = pd.to_numeric(df["mass_scale"], errors="coerce").fillna(1.0)
    df["friction_scale"] = pd.to_numeric(df["friction_scale"], errors="coerce").fillna(1.0)
    if "run_dir" in df:
        inferred = df["run_dir"].map(infer_run_fields)
        for field in ("env", "variant", "seed"):
            values = inferred.map(lambda item: item.get(field) if item else None)
            df[field] = values.combine_first(df[field])
        df["_run_dir_norm"] = df["run_dir"].map(norm_path)
    else:
        df["_run_dir_norm"] = ""
    df = df[df["success"].notna()].copy()
    df = df[(df["mass_scale"] == 1.0) & (df["friction_scale"] == 1.0)]

    clean = df[(df["proprio_noise"] == 0.0) & (df["tactile_dropout"] == 0.0)]
    pert = df[(df["proprio_noise"] > 0.0) | (df["tactile_dropout"] > 0.0)]

    clean_mean = (
        clean.groupby(["env", "variant"])["success"]
        .agg(clean_success="mean", clean_n="count", clean_sem=_sem, clean_ci95=_ci95)
        .reset_index()
    )
    pert_mean = (
        pert.groupby(["env", "variant"])["success"]
        .agg(
            perturbed_success="mean",
            perturbed_n="count",
            perturbed_sem=_sem,
            perturbed_ci95=_ci95,
        )
        .reset_index()
    )
    if not pert.empty:
        robust_auc = (
            pert.groupby(["env", "variant", "seed"])["success"]
            .mean()
            .groupby(["env", "variant"])
            .agg(robustness_auc="mean", robustness_auc_sem=_sem, robustness_auc_ci95=_ci95)
            .reset_index()
        )
    else:
        robust_auc = pd.DataFrame(columns=["env", "variant", "robustness_auc"])

    merged = clean_mean.merge(pert_mean, on=["env", "variant"], how="outer")
    merged = merged.merge(robust_auc, on=["env", "variant"], how="outer")
    merged["robustness_gap"] = merged["clean_success"] - merged["perturbed_success"]
    numeric_cols = [
        "clean_success",
        "clean_sem",
        "clean_ci95",
        "perturbed_success",
        "perturbed_sem",
        "perturbed_ci95",
        "robustness_auc",
        "robustness_auc_sem",
        "robustness_auc_ci95",
        "robustness_gap",
    ]
    numeric_cols = [col for col in numeric_cols if col in merged]
    merged[numeric_cols] = merged[numeric_cols].round(args.decimals)

    probe_rows = []
    for run_dir in sorted(set(str(x) for x in df.get("run_dir", []) if str(x) and str(x) != "nan")):
        fields = infer_run_fields(run_dir)
        if fields is None:
            continue
        metrics = read_last_probe_metrics(Path(run_dir) / "metrics.jsonl")
        if not metrics:
            continue
        probe_rows.append({**fields, **metrics})

    if probe_rows:
        probes = pd.DataFrame(probe_rows)
        metric_cols = [c for c in probes.columns if c.startswith("probe_")]
        probes = (
            probes.groupby(["env", "variant"])[metric_cols]
            .agg(["mean", _sem, _ci95])
            .reset_index()
        )
        probes.columns = [
            "_".join(part for part in col if part)
            if isinstance(col, tuple)
            else col
            for col in probes.columns
        ]
        metric_cols = [c for c in probes.columns if c.startswith("probe_")]
        probes[metric_cols] = probes[metric_cols].round(args.decimals)
        merged = merged.merge(probes, on=["env", "variant"], how="outer")

    merged = merged.sort_values(["env", "variant"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
