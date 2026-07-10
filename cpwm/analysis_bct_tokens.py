#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _last_metrics(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, value in obj.items():
                if (
                    key.startswith("train/tactile_part_aux/")
                    or key.startswith("report/tactile_part_aux/")
                    or key.startswith("train/tactile_part_map_aux/")
                    or key.startswith("report/tactile_part_map_aux/")
                    or key
                    in (
                        "train/tactile_part_map_aux_mae",
                        "train/tactile_part_map_aux_active_mae",
                        "report/tactile_part_map_aux_mae",
                        "report/tactile_part_map_aux_active_mae",
                    )
                ):
                    try:
                        result[key] = float(value)
                    except (TypeError, ValueError):
                        pass
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="outputs/runs_frontier")
    ap.add_argument("--out", default="outputs/results/bct_token_diagnostics.csv")
    args = ap.parse_args()

    rows = []
    for metrics_path in sorted(Path(args.runs_dir).glob("*/metrics.jsonl")):
        metrics = _last_metrics(metrics_path)
        if not metrics:
            continue
        row = {"run_dir": str(metrics_path.parent)}
        row.update(metrics)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
