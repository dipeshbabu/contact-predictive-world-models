#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict


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


def make_env_fn(
    env_id: str,
    noise: float,
    drop: float,
    mass_scale: float,
    friction_scale: float,
    sensors: str,
    env_kwargs: dict,
):
    def thunk():
        import gymnasium as gym
        import humanoid_bench  # noqa: F401

        return gym.make(
            env_id,
            render_mode="rgb_array",
            obs_wrapper=True,
            sensors=sensors,
            tactile_flat=True,
            tactile_concat=True,
            proprio_noise=noise,
            tactile_dropout=drop,
            mass_scale=mass_scale,
            friction_scale=friction_scale,
            **env_kwargs,
        )

    return thunk


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
    ap.add_argument("--sensors", default="", help="Use 'tactile' for PPO tactile baseline")
    ap.add_argument("--variant", default="")
    ap.add_argument("--results_csv", default="outputs/results/results.csv")
    ap.add_argument("--door_success_stand_threshold", type=float, default=0.8)
    ap.add_argument("--door_success_door_threshold", type=float, default=0.8)
    ap.add_argument("--door_success_hatch_threshold", type=float, default=0.8)
    ap.add_argument("--door_success_passage_threshold", type=float, default=0.8)
    ap.add_argument("--insert_success_stand_threshold", type=float, default=0.8)
    ap.add_argument("--insert_success_cube_threshold", type=float, default=0.9)
    ap.add_argument("--insert_success_peg_height_threshold", type=float, default=0.9)
    ap.add_argument("--contact_label_overrides", default="")
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
            args.sensors,
        )
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import numpy as np

    returns = []
    successes = []
    lengths = []
    env_kwargs = {
        "door_success_stand_threshold": args.door_success_stand_threshold,
        "door_success_door_threshold": args.door_success_door_threshold,
        "door_success_hatch_threshold": args.door_success_hatch_threshold,
        "door_success_passage_threshold": args.door_success_passage_threshold,
        "insert_success_stand_threshold": args.insert_success_stand_threshold,
        "insert_success_cube_threshold": args.insert_success_cube_threshold,
        "insert_success_peg_height_threshold": args.insert_success_peg_height_threshold,
        "contact_label_overrides": args.contact_label_overrides,
    }

    for _ in range(args.episodes):
        env = DummyVecEnv(
            [
                make_env_fn(
                    args.env,
                    args.noise,
                    args.tactile_dropout,
                    args.mass_scale,
                    args.friction_scale,
                    args.sensors,
                    env_kwargs,
                )
            ]
        )
        norm_path = run_dir / "vecnormalize.pkl"
        if norm_path.exists():
            env = VecNormalize.load(str(norm_path), env)
            env.training = False
            env.norm_reward = False
        model = PPO.load(str(model_path), env=env)
        obs = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            info = infos[0]
            ep_ret += float(reward[0])
            ep_len += 1
            ep_success = ep_success or bool(info.get("success", False))

        env.close()
        returns.append(ep_ret)
        successes.append(float(ep_success))
        lengths.append(ep_len)

    row = {
        "env": args.env,
        "variant": args.variant
        or ("ppo_tactile" if "tactile" in args.sensors.split(",") else "ppo_proprio"),
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
