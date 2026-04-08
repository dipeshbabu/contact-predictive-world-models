#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results/results.csv")
    ap.add_argument("--outdir", default="outputs/figs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["success_f"] = _to_float(r.get("success", ""))
            r["noise_f"] = _to_float(r.get("proprio_noise", "0"))
            r["drop_f"] = _to_float(r.get("tactile_dropout", "0"))
            r["mass_f"] = _to_float(r.get("mass_scale", "1"))
            r["fric_f"] = _to_float(r.get("friction_scale", "1"))
            if (
                r["success_f"] is None
                or r["noise_f"] is None
                or r["drop_f"] is None
                or r["mass_f"] is None
                or r["fric_f"] is None
            ):
                continue
            rows.append(r)

    sensory_rows = [
        r for r in rows if abs(r["mass_f"] - 1.0) < 1e-12 and abs(r["fric_f"] - 1.0) < 1e-12
    ]
    clean = [r for r in sensory_rows if abs(r["noise_f"]) < 1e-12 and abs(r["drop_f"]) < 1e-12]
    if clean:
        agg = defaultdict(list)
        for r in clean:
            agg[(r["env"], r["variant"])].append(r["success_f"])

        envs = sorted({e for (e, _) in agg.keys()})
        variants = sorted({v for (_, v) in agg.keys()})

        x = list(range(len(envs)))
        width = 0.35 if len(variants) == 2 else 0.25

        plt.figure()
        for i, v in enumerate(variants):
            vals = []
            for e in envs:
                xs = agg.get((e, v), [])
                vals.append(sum(xs) / len(xs) if xs else 0.0)
            xpos = [xi + (i - (len(variants) - 1) / 2) * width for xi in x]
            plt.bar(xpos, vals, width=width, label=v)

        plt.xticks(x, envs, rotation=30, ha="right")
        plt.ylabel("Success")
        plt.title("Clean success by task and variant")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "clean_success_bar.png", dpi=200)
        plt.close()

    by_ev = defaultdict(list)
    for r in sensory_rows:
        by_ev[(r["env"], r["variant"])].append(r)

    for (env, variant), rs in by_ev.items():
        bucket = defaultdict(list)
        for r in rs:
            bucket[(r["noise_f"], r["drop_f"])].append(r["success_f"])

        drops = sorted({d for (_, d) in bucket.keys()})
        plt.figure()
        for d in drops:
            pts = sorted(
                [(n, sum(bucket[(n, d)]) / len(bucket[(n, d)]))
                 for (n, dd) in bucket.keys() if dd == d]
            )
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=f"drop={d}")

        plt.xlabel("Proprio noise std")
        plt.ylabel("Success")
        plt.title(f"{env} / {variant} robustness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{env}_{variant}_robustness.png", dpi=200)
        plt.close()

    print(f"wrote figures to: {outdir}")


if __name__ == "__main__":
    main()
