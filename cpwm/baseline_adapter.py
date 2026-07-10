#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def _format_template(template: str, **values: object) -> list[str]:
    command = template.format(**values)
    return shlex.split(command)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("dreamer", "ppo", "external"), required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--logdir", required=True)
    ap.add_argument(
        "--command_template",
        default="",
        help=(
            "External command with {env}, {variant}, {seed}, {steps}, and {logdir} "
            "placeholders. Required for --backend external."
        ),
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.backend == "dreamer":
        cmd = [
            "python",
            "cpwm/train_dreamer.py",
            "--env",
            args.env,
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
            "--logdir",
            args.logdir,
        ]
    elif args.backend == "ppo":
        cmd = [
            "python",
            "cpwm/train_ppo.py",
            "--env",
            args.env,
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
            "--logdir",
            args.logdir,
        ]
    else:
        if not args.command_template:
            raise ValueError("--command_template is required for external baselines")
        cmd = _format_template(
            args.command_template,
            env=args.env,
            variant=args.variant,
            seed=args.seed,
            steps=args.steps,
            logdir=args.logdir,
        )

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    print("[BASELINE CMD]", " ".join(cmd))
    if args.dry_run:
        return
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
