#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import ruamel.yaml as yaml
except ModuleNotFoundError:
    yaml = None


def _read_last_metric(metrics_path: Path, key: str) -> Optional[float]:
    if not metrics_path.exists():
        return None
    last = None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if key in obj:
                last = obj[key]
    if last is None:
        return None
    try:
        return float(last)
    except Exception:
        return None


def _find_checkpoint(run_dir: Path) -> Path:
    cand = run_dir / "checkpoint.ckpt"
    if cand.exists():
        return cand

    cands: List[Path] = []
    cands += sorted(run_dir.glob("**/*.ckpt"))
    cands += sorted(run_dir.glob("**/checkpoint*"))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        raise FileNotFoundError(f"No checkpoint found under: {run_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed_before = path.exists()
    rows = []
    fieldnames = list(row.keys())
    rewrite = False
    existing_fieldnames = []

    if existed_before:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            rows = list(reader)
        if existing_fieldnames:
            missing = [name for name in fieldnames if name not in existing_fieldnames]
            if missing:
                fieldnames = [*existing_fieldnames, *missing]
                rewrite = True
            else:
                fieldnames = existing_fieldnames

    mode = "w" if rewrite else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w" or not existed_before or not existing_fieldnames:
            writer.writeheader()
        if rewrite:
            for old_row in rows:
                writer.writerow(old_row)
        writer.writerow(row)


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    if yaml is None:
        return {}
    parser = yaml.YAML(typ="safe")
    data = parser.load(config_path.read_text(encoding="utf-8"))
    return data or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--tactile_dropout", type=float, default=0.0)
    ap.add_argument("--mass_scale", type=float, default=1.0)
    ap.add_argument("--friction_scale", type=float, default=1.0)
    ap.add_argument("--results_csv", default="outputs/results/results.csv")
    ap.add_argument("--env", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    run_config = _load_run_config(run_dir)
    env = args.env
    seed = args.seed
    variant = "unknown"
    task = run_config.get("task")
    if env is None and isinstance(task, str) and task.startswith("humanoid_"):
        env = task[len("humanoid_") :]
    if seed is None and "seed" in run_config:
        seed = int(run_config["seed"])
    if run_config.get("tactile_aux_weight", 0.0) > 0:
        variant = "aux"

    m = re.match(r"(.+?)_([A-Za-z0-9_]+)_s(\d+)$", run_dir.name)
    if m:
        if env is None:
            env = m.group(1)
        if variant == "unknown":
            variant = m.group(2)
        if seed is None:
            seed = int(m.group(3))

    if env is None:
        raise ValueError("Could not infer env from run_dir name. Pass --env explicitly.")
    if seed is None:
        seed = 0

    try:
        ckpt = _find_checkpoint(run_dir)
    except FileNotFoundError:
        if not args.dry_run:
            raise
        ckpt = run_dir / "checkpoint.ckpt"
    eval_dir = run_dir / "eval" / (
        f"noise{args.noise}_drop{args.tactile_dropout}"
        f"_mass{args.mass_scale}_fric{args.friction_scale}"
    )

    env_vars = os.environ.copy()
    repo_root = Path(__file__).resolve().parent.parent
    pythonpath_parts = [str(repo_root)]
    if env_vars.get("PYTHONPATH"):
        pythonpath_parts.append(env_vars["PYTHONPATH"])
    env_vars["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if sys.platform.startswith("linux"):
        env_vars.setdefault("MUJOCO_GL", "egl")
        env_vars.setdefault("PYOPENGL_PLATFORM", "egl")
    elif sys.platform == "win32":
        env_vars.setdefault("MUJOCO_GL", "glfw")

    humanoid_cfg = run_config.get("env", {}).get("humanoid", {})
    obs_key = str(humanoid_cfg.get("obs_key", "dict"))
    obs_wrapper = str(humanoid_cfg.get("obs_wrapper", True))
    sensors = str(humanoid_cfg.get("sensors", "tactile"))
    tactile_flat = str(humanoid_cfg.get("tactile_flat", True))
    tactile_concat = str(humanoid_cfg.get("tactile_concat", True))
    jax_platform = str(run_config.get("jax", {}).get("platform", "gpu"))

    cmd = [
        sys.executable, "-m", "embodied.agents.dreamerv3.train",
        "--configs", "humanoid_benchmark",
        "--task", f"humanoid_{env}",
        "--logdir", str(eval_dir),
        "--seed", str(seed),
        "--run.steps", str(args.steps),
        "--run.script", "eval_only",
        "--run.num_envs", "1",
        "--jax.platform", jax_platform,
        "--env.humanoid.obs_key", obs_key,
        "--env.humanoid.obs_wrapper", obs_wrapper,
        "--env.humanoid.sensors", sensors,
        "--env.humanoid.tactile_flat", tactile_flat,
        "--env.humanoid.tactile_concat", tactile_concat,
        "--env.humanoid.proprio_noise", str(args.noise),
        "--env.humanoid.tactile_dropout", str(args.tactile_dropout),
        "--env.humanoid.mass_scale", str(args.mass_scale),
        "--env.humanoid.friction_scale", str(args.friction_scale),
        "--run.from_checkpoint", str(ckpt),
    ]

    tactile_aux_weight = float(run_config.get("tactile_aux_weight", 0.0))
    if tactile_aux_weight > 0:
        cmd += ["--tactile_aux_weight", str(tactile_aux_weight)]

    print("[EVAL CMD]", " ".join(cmd))
    print("[EVAL CKPT]", ckpt)

    if args.dry_run:
        return

    eval_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, env=env_vars)

    metrics_path = eval_dir / "metrics.jsonl"
    success = _read_last_metric(metrics_path, "epstats/success")
    score = _read_last_metric(metrics_path, "epstats/score")
    length = _read_last_metric(metrics_path, "epstats/length")

    row: Dict[str, Any] = {
        "env": env,
        "variant": variant,
        "seed": seed,
        "eval_steps": args.steps,
        "proprio_noise": args.noise,
        "tactile_dropout": args.tactile_dropout,
        "mass_scale": args.mass_scale,
        "friction_scale": args.friction_scale,
        "success": success if success is not None else "",
        "score": score if score is not None else "",
        "ep_length": length if length is not None else "",
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "metrics_path": str(metrics_path),
    }

    _append_csv(Path(args.results_csv), row)
    print(f"wrote/updated: {args.results_csv}")


if __name__ == "__main__":
    main()
