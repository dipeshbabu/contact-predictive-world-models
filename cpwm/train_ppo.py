#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

def make_env(
    env_id: str,
    noise: float,
    drop: float,
    mass_scale: float,
    friction_scale: float,
):
    import gymnasium as gym
    import humanoid_bench  # noqa: F401
    from stable_baselines3.common.monitor import Monitor

    env = gym.make(
        env_id,
        render_mode="rgb_array",
        obs_wrapper=True,
        sensors="",
        proprio_noise=noise,
        tactile_dropout=drop,
        mass_scale=mass_scale,
        friction_scale=friction_scale,
    )
    return Monitor(env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--tactile_dropout", type=float, default=0.0)
    ap.add_argument("--mass_scale", type=float, default=1.0)
    ap.add_argument("--friction_scale", type=float, default=1.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            "[PPO TRAIN DRY RUN]",
            args.env,
            args.seed,
            args.steps,
            args.noise,
            args.tactile_dropout,
            args.mass_scale,
            args.friction_scale,
        )
        return

    from stable_baselines3 import PPO

    env = make_env(
        args.env,
        args.noise,
        args.tactile_dropout,
        args.mass_scale,
        args.friction_scale,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=args.steps)
    model.save(str(logdir / "ppo_model"))
    env.close()


if __name__ == "__main__":
    main()
