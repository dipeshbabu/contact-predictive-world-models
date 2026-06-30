#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import mujoco


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensors", default="tactile")
    ap.add_argument("--out", default="outputs/results/contact_label_audit.csv")
    args = ap.parse_args()

    import gymnasium as gym
    import humanoid_bench  # noqa: F401
    from humanoid_bench.env import HumanoidEnv

    env = gym.make(
        args.env,
        render_mode="rgb_array",
        obs_wrapper=True,
        sensors=args.sensors,
        tactile_flat=True,
        tactile_concat=True,
    )
    obs, _ = env.reset(seed=args.seed)
    base = env.unwrapped
    if not isinstance(base, HumanoidEnv):
        raise TypeError(f"Expected HumanoidEnv, got {type(base)}")

    counts = Counter()
    examples = {}
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        for i in range(int(base.data.ncon)):
            contact = base.data.contact[i]
            geom_names = []
            body_names = []
            for geom_id in (int(contact.geom1), int(contact.geom2)):
                geom_names.append(
                    mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    or ""
                )
                body_id = int(base.model.geom_bodyid[geom_id])
                body_names.append(
                    mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    or ""
                )
            key = tuple(sorted(geom_names))
            counts[key] += 1
            examples.setdefault(key, (tuple(geom_names), tuple(body_names)))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "geom1",
        "geom2",
        "body1",
        "body2",
        "count",
        "contact_hand",
        "contact_foot",
        "contact_torso",
        "contact_object",
        "contact_robot_object",
        "reviewed",
        "notes",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, count in counts.most_common():
            geom_names, body_names = examples[key]
            labels = base._contact_rule_labels(geom_names, body_names)
            writer.writerow(
                {
                    "geom1": geom_names[0],
                    "geom2": geom_names[1],
                    "body1": body_names[0],
                    "body2": body_names[1],
                    "count": count,
                    "contact_hand": labels["contact_hand"],
                    "contact_foot": labels["contact_foot"],
                    "contact_torso": labels["contact_torso"],
                    "contact_object": labels["contact_object"],
                    "contact_robot_object": labels["contact_robot_object"],
                    "reviewed": 0,
                    "notes": "",
                }
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
