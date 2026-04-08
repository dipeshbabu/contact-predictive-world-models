#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np


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


def make_env(
    env_id: str,
    noise: float,
    drop: float,
    mass_scale: float,
    friction_scale: float,
):
    import gymnasium as gym
    import humanoid_bench  # noqa: F401

    return gym.make(
        env_id,
        render_mode="rgb_array",
        obs_wrapper=True,
        sensors="",
        proprio_noise=noise,
        tactile_dropout=drop,
        mass_scale=mass_scale,
        friction_scale=friction_scale,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--tactile_dropout", type=float, default=0.0)
    ap.add_argument("--mass_scale", type=float, default=1.0)
    ap.add_argument("--friction_scale", type=float, default=1.0)
    ap.add_argument("--results_csv", default="outputs/results/results.csv")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    model_path = run_dir / "ppo_model.zip"
    if not model_path.exists():
        model_path = run_dir / "ppo_model"

    if args.dry_run:
        print(
            "[PPO EVAL DRY RUN]",
            run_dir,
            args.env,
            args.seed,
            args.episodes,
            args.noise,
            args.tactile_dropout,
            args.mass_scale,
            args.friction_scale,
        )
        return

    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))

    returns = []
    successes = []
    lengths = []

    for _ in range(args.episodes):
        env = make_env(
            args.env,
            args.noise,
            args.tactile_dropout,
            args.mass_scale,
            args.friction_scale,
        )
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0
        ep_success = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            ep_success = ep_success or bool(info.get("success", False))

        env.close()
        returns.append(ep_ret)
        successes.append(float(ep_success))
        lengths.append(ep_len)

    row = {
        "env": args.env,
        "variant": "ppo_proprio",
        "seed": args.seed,
        "eval_steps": "",
        "proprio_noise": args.noise,
        "tactile_dropout": args.tactile_dropout,
        "mass_scale": args.mass_scale,
        "friction_scale": args.friction_scale,
        "success": float(np.mean(successes)),
        "score": float(np.mean(returns)),
        "ep_length": float(np.mean(lengths)),
        "run_dir": str(run_dir),
        "eval_dir": "",
        "metrics_path": "",
    }
    _append_csv(Path(args.results_csv), row)
    print(f"wrote/updated: {args.results_csv}")


if __name__ == "__main__":
    main()
