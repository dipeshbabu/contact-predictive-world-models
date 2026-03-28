#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="HumanoidBench env id, e.g. h1touch-door-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--tactile_aux_weight", type=float, default=0.0)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--jax_platform", default="gpu")
    ap.add_argument("--use_rgb", action="store_true", help="Optional RGB mode if your env supports it")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    task = f"humanoid_{args.env}"
    sensors = "tactile,image" if args.use_rgb else "tactile"
    repo_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable, "-m", "embodied.agents.dreamerv3.train",
        "--configs", "humanoid_benchmark",
        "--seed", str(args.seed),
        "--task", task,
        "--logdir", args.logdir,
        "--run.steps", str(args.steps),
        "--run.num_envs", str(args.num_envs),
        "--jax.platform", args.jax_platform,
        "--env.humanoid.obs_key", "dict",
        "--env.humanoid.obs_wrapper", "True",
        "--env.humanoid.sensors", sensors,
        "--env.humanoid.tactile_flat", "True",
        "--env.humanoid.tactile_concat", "True",
    ]

    if args.tactile_aux_weight > 0:
        cmd += ["--tactile_aux_weight", str(args.tactile_aux_weight)]

    print("[TRAIN CMD]", " ".join(cmd))
    if args.dry_run:
        return

    env_vars = os.environ.copy()
    pythonpath_parts = [str(repo_root)]
    if env_vars.get("PYTHONPATH"):
        pythonpath_parts.append(env_vars["PYTHONPATH"])
    env_vars["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if sys.platform.startswith("linux"):
        env_vars.setdefault("MUJOCO_GL", "egl")
        env_vars.setdefault("PYOPENGL_PLATFORM", "egl")
    elif sys.platform == "win32":
        env_vars.setdefault("MUJOCO_GL", "glfw")

    subprocess.run(cmd, check=True, env=env_vars)


if __name__ == "__main__":
    main()
