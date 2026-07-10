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


KNOWN_VARIANTS = (
    "future1_noact",
    "proprio_only",
    "ppo_proprio",
    "ppo_tactile",
    "aux_contact",
    "future1",
    "future3",
    "future5",
    "masked",
    "tactile_group",
    "part_tokens",
    "contact_frontier",
    "frontier",
    "frontier_bct",
    "frontier_bct_flat",
    "frontier_bct_native",
    "frontier_bct_spatial",
    "frontier_bct_no_contact",
    "frontier_bct_no_rgb",
    "frontier_rgb",
    "frontier_rgb_bct",
    "frontier_rgb_bct_spatial",
    "contact_onset",
    "contact_change",
    "both_onset",
    "both_change",
    "aux_contact_onset",
    "aux_contact_change",
    "contact1",
    "contact3",
    "current",
    "proprio",
    "contact",
    "noact",
    "recon",
    "both",
    "base",
    "aux",
)


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


def _infer_run_name(name: str) -> Optional[tuple[str, str, int]]:
    match = re.match(r"(.+)_s(\d+)$", name)
    if not match:
        return None
    stem, seed = match.group(1), int(match.group(2))
    for variant in sorted(KNOWN_VARIANTS, key=len, reverse=True):
        suffix = f"_{variant}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], variant, seed
    env, variant = stem.rsplit("_", 1)
    return env, variant, seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--configs", default="humanoid_benchmark")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--tactile_dropout", type=float, default=0.0)
    ap.add_argument("--mass_scale", type=float, default=1.0)
    ap.add_argument("--friction_scale", type=float, default=1.0)
    ap.add_argument("--results_csv", default="outputs/results/results.csv")
    ap.add_argument("--env", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--door_success_stand_threshold", type=float, default=None)
    ap.add_argument("--door_success_door_threshold", type=float, default=None)
    ap.add_argument("--door_success_hatch_threshold", type=float, default=None)
    ap.add_argument("--door_success_passage_threshold", type=float, default=None)
    ap.add_argument("--insert_success_stand_threshold", type=float, default=None)
    ap.add_argument("--insert_success_cube_threshold", type=float, default=None)
    ap.add_argument("--insert_success_peg_height_threshold", type=float, default=None)
    ap.add_argument("--contact_label_overrides", default=None)
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
    parsed = _infer_run_name(run_dir.name)
    if parsed:
        parsed_env, parsed_variant, parsed_seed = parsed
        if env is None:
            env = parsed_env
        variant = parsed_variant
        if seed is None:
            seed = parsed_seed

    if variant == "unknown" and run_config.get("tactile_aux_weight", 0.0) > 0:
        variant = "aux"

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
    if variant in ("proprio", "proprio_only"):
        default_sensors = ""
    elif variant in ("frontier_rgb", "frontier_rgb_bct", "frontier_rgb_bct_spatial"):
        default_sensors = "tactile,image"
    else:
        default_sensors = "tactile"
    sensors = str(humanoid_cfg.get("sensors", default_sensors))
    tactile_flat = str(humanoid_cfg.get("tactile_flat", True))
    tactile_concat = str(humanoid_cfg.get("tactile_concat", True))
    jax_platform = str(run_config.get("jax", {}).get("platform", "gpu"))
    success_contact_args = [
        "--env.humanoid.door_success_stand_threshold",
        str(
            args.door_success_stand_threshold
            if args.door_success_stand_threshold is not None
            else humanoid_cfg.get("door_success_stand_threshold", 0.8)
        ),
        "--env.humanoid.door_success_door_threshold",
        str(
            args.door_success_door_threshold
            if args.door_success_door_threshold is not None
            else humanoid_cfg.get("door_success_door_threshold", 0.8)
        ),
        "--env.humanoid.door_success_hatch_threshold",
        str(
            args.door_success_hatch_threshold
            if args.door_success_hatch_threshold is not None
            else humanoid_cfg.get("door_success_hatch_threshold", 0.8)
        ),
        "--env.humanoid.door_success_passage_threshold",
        str(
            args.door_success_passage_threshold
            if args.door_success_passage_threshold is not None
            else humanoid_cfg.get("door_success_passage_threshold", 0.8)
        ),
        "--env.humanoid.insert_success_stand_threshold",
        str(
            args.insert_success_stand_threshold
            if args.insert_success_stand_threshold is not None
            else humanoid_cfg.get("insert_success_stand_threshold", 0.8)
        ),
        "--env.humanoid.insert_success_cube_threshold",
        str(
            args.insert_success_cube_threshold
            if args.insert_success_cube_threshold is not None
            else humanoid_cfg.get("insert_success_cube_threshold", 0.9)
        ),
        "--env.humanoid.insert_success_peg_height_threshold",
        str(
            args.insert_success_peg_height_threshold
            if args.insert_success_peg_height_threshold is not None
            else humanoid_cfg.get("insert_success_peg_height_threshold", 0.9)
        ),
        "--env.humanoid.contact_label_overrides",
        str(
            args.contact_label_overrides
            if args.contact_label_overrides is not None
            else humanoid_cfg.get("contact_label_overrides", "")
        ),
    ]

    cmd = [
        sys.executable, "-m", "embodied.agents.dreamerv3.train",
        "--configs", args.configs,
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
    ] + success_contact_args

    if "image" in {part.strip() for part in sensors.split(",") if part.strip()}:
        cmd += [
            "--encoder.cnn_keys", "image",
            "--encoder.mlp_keys", "^(?!image$).*",
            "--decoder.cnn_keys", "image",
            "--decoder.mlp_keys", "^(?!image$).*",
        ]

    default_tactile_aux = (
        0.1
        if variant
        in (
            "aux",
            "future1",
            "recon",
            "current",
            "future3",
            "future5",
            "masked",
            "tactile_group",
            "frontier",
            "frontier_bct",
            "frontier_bct_flat",
            "frontier_bct_native",
            "frontier_bct_spatial",
            "frontier_bct_no_contact",
            "frontier_bct_no_rgb",
            "frontier_rgb",
            "frontier_rgb_bct",
            "frontier_rgb_bct_spatial",
            "noact",
            "future1_noact",
            "both",
            "aux_contact",
            "both_onset",
            "both_change",
            "aux_contact_onset",
            "aux_contact_change",
        )
        else 0.0
    )
    tactile_aux_weight = float(run_config.get("tactile_aux_weight", default_tactile_aux))
    tactile_group_aux_weight = float(
        run_config.get(
            "tactile_group_aux_weight",
            0.05
            if variant
            in (
                "tactile_group",
                "frontier",
                "frontier_bct",
                "frontier_bct_flat",
                "frontier_bct_native",
                "frontier_bct_spatial",
                "frontier_bct_no_contact",
                "frontier_bct_no_rgb",
                "frontier_rgb",
                "frontier_rgb_bct",
                "frontier_rgb_bct_spatial",
            )
            else 0.0,
        )
    )
    tactile_part_aux_weight = float(
        run_config.get(
            "tactile_part_aux_weight",
            0.05
            if variant
            in (
                "part_tokens",
                "frontier_bct",
                "frontier_bct_flat",
                "frontier_bct_native",
                "frontier_bct_spatial",
                "frontier_bct_no_contact",
                "frontier_bct_no_rgb",
                "frontier_rgb_bct",
                "frontier_rgb_bct_spatial",
            )
            else 0.0,
        )
    )
    default_part_source = (
        "flat"
        if variant in ("part_tokens", "frontier_bct_flat")
        else "native"
        if variant
        in (
            "frontier_bct",
            "frontier_bct_native",
            "frontier_bct_spatial",
            "frontier_bct_no_contact",
            "frontier_bct_no_rgb",
            "frontier_rgb_bct",
            "frontier_rgb_bct_spatial",
        )
        else "flat"
    )
    tactile_part_aux_source = str(
        run_config.get("tactile_part_aux_source", default_part_source)
    )
    if tactile_part_aux_weight > 0 and tactile_part_aux_source == "native":
        cmd += [
            "--env.humanoid.tactile_part_tokens", "True",
            "--env.humanoid.tactile_part_threshold",
            str(run_config.get("tactile_part_aux_threshold", 1e-4)),
        ]
    tactile_part_map_aux_weight = float(
        run_config.get(
            "tactile_part_map_aux_weight",
            0.03
            if variant in ("frontier_bct_spatial", "frontier_rgb_bct_spatial")
            else 0.0,
        )
    )
    if tactile_part_map_aux_weight > 0:
        cmd += [
            "--env.humanoid.tactile_part_maps", "True",
            "--env.humanoid.tactile_part_threshold",
            str(run_config.get("tactile_part_aux_threshold", 1e-4)),
        ]
    if (
        tactile_aux_weight > 0
        or tactile_group_aux_weight > 0
        or tactile_part_aux_weight > 0
        or tactile_part_map_aux_weight > 0
    ):
        default_tactile_mode = (
            "current"
            if variant in ("recon", "current")
            else "masked"
            if variant
            in (
                "masked",
                "part_tokens",
                "frontier",
                "frontier_bct",
                "frontier_bct_flat",
                "frontier_bct_native",
                "frontier_bct_spatial",
                "frontier_bct_no_contact",
                "frontier_bct_no_rgb",
                "frontier_rgb",
                "frontier_rgb_bct",
                "frontier_rgb_bct_spatial",
            )
            else "future"
        )
        default_tactile_horizon = 3 if variant == "future3" else 5 if variant == "future5" else 1
        default_tactile_action = variant not in ("noact", "future1_noact")
        cmd += [
            "--tactile_aux_mode",
            str(run_config.get("tactile_aux_mode", default_tactile_mode)),
            "--tactile_aux_horizon",
            str(run_config.get("tactile_aux_horizon", default_tactile_horizon)),
            "--tactile_aux_action",
            str(run_config.get("tactile_aux_action", default_tactile_action)),
            "--tactile_mask_prob", str(run_config.get("tactile_mask_prob", 0.25)),
            "--tactile_aux_groups", str(run_config.get("tactile_aux_groups", 8)),
        ]
    if tactile_aux_weight > 0:
        cmd += ["--tactile_aux_weight", str(tactile_aux_weight)]
    if tactile_group_aux_weight > 0:
        cmd += [
            "--tactile_group_aux_weight", str(tactile_group_aux_weight),
            "--tactile_aux_groups", str(run_config.get("tactile_aux_groups", 8)),
        ]
    if tactile_part_aux_weight > 0:
        cmd += [
            "--tactile_part_aux_weight", str(tactile_part_aux_weight),
            "--tactile_part_aux_parts", str(run_config.get("tactile_part_aux_parts", 8)),
            "--tactile_part_aux_threshold",
            str(run_config.get("tactile_part_aux_threshold", 1e-4)),
            "--tactile_part_aux_source", tactile_part_aux_source,
        ]
    if tactile_part_map_aux_weight > 0:
        cmd += [
            "--tactile_part_map_aux_weight", str(tactile_part_map_aux_weight),
        ]
    default_contact_aux = (
        0.1
        if variant
        in (
            "contact",
            "contact1",
            "contact3",
            "contact_frontier",
            "contact_onset",
            "contact_change",
            "frontier",
            "frontier_bct",
            "frontier_bct_flat",
            "frontier_bct_native",
            "frontier_bct_spatial",
            "frontier_bct_no_rgb",
            "frontier_rgb",
            "frontier_rgb_bct",
            "frontier_rgb_bct_spatial",
            "both",
            "aux_contact",
            "both_onset",
            "both_change",
            "aux_contact_onset",
            "aux_contact_change",
        )
        else 0.0
    )
    contact_aux_weight = float(run_config.get("contact_aux_weight", default_contact_aux))
    if contact_aux_weight > 0:
        default_contact_mode = "future"
        if variant in ("contact_onset", "both_onset", "aux_contact_onset"):
            default_contact_mode = "onset"
        elif variant in (
            "contact_frontier",
            "frontier",
            "frontier_bct",
            "frontier_bct_flat",
            "frontier_bct_native",
            "frontier_bct_spatial",
            "frontier_bct_no_rgb",
            "frontier_rgb",
            "frontier_rgb_bct",
            "frontier_rgb_bct_spatial",
        ):
            default_contact_mode = "onset"
        elif variant in ("contact_change", "both_change", "aux_contact_change"):
            default_contact_mode = "change"
        cmd += [
            "--contact_aux_weight", str(contact_aux_weight),
            "--contact_aux_mode",
            str(run_config.get("contact_aux_mode", default_contact_mode)),
            "--contact_aux_horizon",
            str(run_config.get("contact_aux_horizon", 3 if variant == "contact3" else 1)),
            "--contact_aux_action", str(run_config.get("contact_aux_action", True)),
            "--contact_aux_balanced",
            str(
                run_config.get(
                    "contact_aux_balanced",
                    variant
                    in (
                        "contact_frontier",
                        "frontier",
                        "frontier_bct",
                        "frontier_bct_flat",
                        "frontier_bct_native",
                        "frontier_bct_spatial",
                        "frontier_bct_no_rgb",
                        "frontier_rgb",
                        "frontier_rgb_bct",
                        "frontier_rgb_bct_spatial",
                    ),
                )
            ),
            "--contact_aux_rich_labels",
            str(
                run_config.get(
                    "contact_aux_rich_labels",
                    variant
                    in (
                        "contact_frontier",
                        "frontier",
                        "frontier_bct",
                        "frontier_bct_flat",
                        "frontier_bct_native",
                        "frontier_bct_spatial",
                        "frontier_bct_no_rgb",
                        "frontier_rgb",
                        "frontier_rgb_bct",
                        "frontier_rgb_bct_spatial",
                    ),
                )
            ),
            "--contact_aux_ensemble",
            str(
                run_config.get(
                    "contact_aux_ensemble",
                    4
                    if variant
                    in (
                        "contact_frontier",
                        "frontier",
                        "frontier_bct",
                        "frontier_bct_flat",
                        "frontier_bct_native",
                        "frontier_bct_spatial",
                        "frontier_bct_no_rgb",
                        "frontier_rgb",
                        "frontier_rgb_bct",
                        "frontier_rgb_bct_spatial",
                    )
                    else 1,
                )
            ),
            "--contact_imagine_weight",
            str(
                run_config.get(
                    "contact_imagine_weight",
                    0.05
                    if variant
                    in (
                        "contact_frontier",
                        "frontier",
                        "frontier_bct",
                        "frontier_bct_flat",
                        "frontier_bct_native",
                        "frontier_bct_spatial",
                        "frontier_bct_no_rgb",
                        "frontier_rgb",
                        "frontier_rgb_bct",
                        "frontier_rgb_bct_spatial",
                    )
                    else 0.0,
                )
            ),
        ]

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
