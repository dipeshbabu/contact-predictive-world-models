#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


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
        from stable_baselines3.common.monitor import Monitor

        env = gym.make(
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
        return Monitor(env)

    return thunk


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
    ap.add_argument("--sensors", default="", help="Use 'tactile' for PPO tactile baseline")
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--n_steps", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--net_width", type=int, default=256)
    ap.add_argument("--net_depth", type=int, default=2)
    ap.add_argument("--no_normalize_obs", action="store_true")
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
            args.sensors,
            args.num_envs,
        )
        return

    from stable_baselines3 import PPO
    from gymnasium.spaces import Dict as GymDict
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
            for _ in range(args.num_envs)
        ]
    )
    if not args.no_normalize_obs:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    policy = "MultiInputPolicy" if isinstance(env.observation_space, GymDict) else "MlpPolicy"
    model = PPO(
        policy,
        env,
        verbose=1,
        seed=args.seed,
        policy_kwargs={
            "net_arch": {
                "pi": [args.net_width] * args.net_depth,
                "vf": [args.net_width] * args.net_depth,
            }
        },
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        tensorboard_log=None,
        device=args.device,
    )
    model.learn(total_timesteps=args.steps)
    model.save(str(logdir / "ppo_model"))
    if isinstance(env, VecNormalize):
        env.save(str(logdir / "vecnormalize.pkl"))
    env.close()


if __name__ == "__main__":
    main()
