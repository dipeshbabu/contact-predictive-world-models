#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--env", required=True, help="HumanoidBench env id, e.g. h1touch-door-v0"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--tactile_aux_weight", type=float, default=0.0)
    ap.add_argument(
        "--tactile_aux_mode",
        choices=("future", "current", "masked"),
        default="future",
        help=(
            "future predicts tau_{t+1:t+H}; current reconstructs tactile; "
            "masked reconstructs tactile from masked tactile observations"
        ),
    )
    ap.add_argument("--tactile_aux_horizon", type=int, default=1)
    ap.add_argument("--tactile_mask_prob", type=float, default=0.25)
    ap.add_argument("--tactile_aux_groups", type=int, default=8)
    ap.add_argument("--tactile_group_aux_weight", type=float, default=0.0)
    ap.add_argument(
        "--no_tactile_aux_action",
        action="store_true",
        help="Disable action conditioning in the tactile auxiliary head",
    )
    ap.add_argument("--contact_aux_weight", type=float, default=0.0)
    ap.add_argument(
        "--contact_aux_mode",
        choices=("future", "current", "onset", "change"),
        default="future",
        help=(
            "future predicts future contact labels; current reconstructs labels; "
            "onset predicts new contacts; change predicts contact state changes"
        ),
    )
    ap.add_argument("--contact_aux_horizon", type=int, default=1)
    ap.add_argument(
        "--no_contact_aux_action",
        action="store_true",
        help="Disable action conditioning in the contact auxiliary head",
    )
    ap.add_argument("--contact_aux_balanced", action="store_true")
    ap.add_argument("--contact_aux_rich_labels", action="store_true")
    ap.add_argument("--contact_aux_ensemble", type=int, default=1)
    ap.add_argument("--contact_imagine_weight", type=float, default=0.0)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--jax_platform", default="gpu")
    ap.add_argument(
        "--sensors",
        default=None,
        help="Comma-separated humanoid sensors. Use an empty string for proprio-only.",
    )
    ap.add_argument("--door_success_stand_threshold", type=float, default=0.8)
    ap.add_argument("--door_success_door_threshold", type=float, default=0.8)
    ap.add_argument("--door_success_hatch_threshold", type=float, default=0.8)
    ap.add_argument("--door_success_passage_threshold", type=float, default=0.8)
    ap.add_argument("--insert_success_stand_threshold", type=float, default=0.8)
    ap.add_argument("--insert_success_cube_threshold", type=float, default=0.9)
    ap.add_argument("--insert_success_peg_height_threshold", type=float, default=0.9)
    ap.add_argument(
        "--contact_label_overrides",
        default="",
        help="Optional CSV with hand-reviewed contact labels for geom/body pairs.",
    )
    ap.add_argument(
        "--use_rgb", action="store_true", help="Optional RGB mode if your env supports it"
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    task = f"humanoid_{args.env}"
    if args.sensors is not None:
        sensors = args.sensors
    else:
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
        "--env.humanoid.door_success_stand_threshold",
        str(args.door_success_stand_threshold),
        "--env.humanoid.door_success_door_threshold",
        str(args.door_success_door_threshold),
        "--env.humanoid.door_success_hatch_threshold",
        str(args.door_success_hatch_threshold),
        "--env.humanoid.door_success_passage_threshold",
        str(args.door_success_passage_threshold),
        "--env.humanoid.insert_success_stand_threshold",
        str(args.insert_success_stand_threshold),
        "--env.humanoid.insert_success_cube_threshold",
        str(args.insert_success_cube_threshold),
        "--env.humanoid.insert_success_peg_height_threshold",
        str(args.insert_success_peg_height_threshold),
        "--env.humanoid.contact_label_overrides",
        args.contact_label_overrides,
    ]

    if "image" in {part.strip() for part in sensors.split(",") if part.strip()}:
        cmd += [
            "--encoder.cnn_keys", "image",
            "--encoder.mlp_keys", "^(?!image$).*",
            "--decoder.cnn_keys", "image",
            "--decoder.mlp_keys", "^(?!image$).*",
        ]

    if args.tactile_aux_weight > 0 or args.tactile_group_aux_weight > 0:
        cmd += [
            "--tactile_aux_mode", args.tactile_aux_mode,
            "--tactile_aux_horizon", str(args.tactile_aux_horizon),
            "--tactile_aux_action", str(not args.no_tactile_aux_action),
            "--tactile_mask_prob", str(args.tactile_mask_prob),
            "--tactile_aux_groups", str(args.tactile_aux_groups),
        ]
    if args.tactile_aux_weight > 0:
        cmd += ["--tactile_aux_weight", str(args.tactile_aux_weight)]
    if args.tactile_group_aux_weight > 0:
        cmd += [
            "--tactile_group_aux_weight", str(args.tactile_group_aux_weight),
            "--tactile_aux_groups", str(args.tactile_aux_groups),
        ]
    if args.contact_aux_weight > 0:
        cmd += [
            "--contact_aux_weight", str(args.contact_aux_weight),
            "--contact_aux_mode", args.contact_aux_mode,
            "--contact_aux_horizon", str(args.contact_aux_horizon),
            "--contact_aux_action", str(not args.no_contact_aux_action),
            "--contact_aux_balanced", str(args.contact_aux_balanced),
            "--contact_aux_rich_labels", str(args.contact_aux_rich_labels),
            "--contact_aux_ensemble", str(args.contact_aux_ensemble),
            "--contact_imagine_weight", str(args.contact_imagine_weight),
        ]

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
