import csv
import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import humanoid_bench.dmc_deps.dmc_index as index
import collections
NamedIndexStructs = collections.namedtuple(
    'NamedIndexStructs', ['model', 'data'])

from dm_control.utils import rewards

from humanoid_bench.dmc_deps.dmc_wrapper import MjDataWrapper, MjModelWrapper

from .wrappers import (
    SingleReachWrapper,
    DoubleReachAbsoluteWrapper,
    DoubleReachRelativeWrapper,
    BlockedHandsLocoWrapper,
    ObservationWrapper,
)

from .envs.door import Door
from .envs.push import Push
from .envs.cabinet import Cabinet
from .envs.insert import Insert
from .envs.locomotion import Walk, Run
from .robots import H1Touch

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}
DEFAULT_RANDOMNESS = 0.01

ROBOTS = {"h1touch": H1Touch}
TASKS = {
    "walk": Walk,
    "run": Run,
    "door": Door,
    "push": Push,
    "cabinet": Cabinet,
    "insert_small": Insert,  # This is not an error
}


class HumanoidEnv(MujocoEnv, gym.utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        robot=None,
        control=None,
        task=None,
        render_mode="rgb_array",
        width=256,
        height=256,
        randomness=DEFAULT_RANDOMNESS,
        **kwargs,
    ):
        assert robot and control and task, f"{robot} {control} {task}"
        gym.utils.EzPickle.__init__(self, metadata=self.metadata)

        asset_path = os.path.join(os.path.dirname(__file__), "assets")

        if "model_path" in kwargs:
            model_path = kwargs["model_path"]
        else:
            model_path = f"envs/{robot}_{control}_{task}.xml"
        
        model_path = os.path.join(asset_path, model_path)
        self.robot_name = robot

        self.robot = ROBOTS[robot](self)
        if isinstance(task, str):
            task_info = TASKS[task](self.robot, None, **kwargs)
        else:
            task_info = task(self.robot, None, **kwargs)

        self.obs_wrapper = kwargs.get("obs_wrapper", False)
        if not isinstance(self.obs_wrapper, bool):
            self.obs_wrapper = str(self.obs_wrapper).lower() == "true"

        self.blocked_hands = kwargs.get("blocked_hands", False)
        if not isinstance(self.blocked_hands, bool):
            self.blocked_hands = str(self.blocked_hands).lower() == "true"

        self.small_obs = kwargs.get("small_obs", False)
        if not isinstance(self.small_obs, bool):
            self.small_obs = str(self.small_obs).lower() == "true"

        self.mass_scale = float(kwargs.get("mass_scale", 1.0) or 1.0)
        self.friction_scale = float(kwargs.get("friction_scale", 1.0) or 1.0)
        self.contact_label_overrides_path = str(
            kwargs.get("contact_label_overrides", "") or ""
        )

        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=task_info.frame_skip,
            observation_space=task_info.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=task_info.camera_name,
        )

        self._default_body_mass = self.model.body_mass.copy()
        self._default_geom_friction = self.model.geom_friction.copy()
        if self.mass_scale != 1.0:
            self.model.body_mass[:] = self._default_body_mass * self.mass_scale
        if self.friction_scale != 1.0:
            self.model.geom_friction[:] = self._default_geom_friction * self.friction_scale

        self.action_high = self.action_space.high
        self.action_low = self.action_space.low
        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )

        if isinstance(task, str):
            self.task = TASKS[task](self.robot, self, **kwargs)
        else:
            self.task = task(self.robot, self, **kwargs)

        if self.blocked_hands:
            self.task = BlockedHandsLocoWrapper(self.task, **kwargs)

        # Wrap for hierarchical control
        if (
            "policy_type" in kwargs
            and kwargs["policy_type"]
            and kwargs["policy_type"] is not None
            and kwargs["policy_type"] != "flat"
        ):
            if kwargs["policy_type"] == "reach_single":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = SingleReachWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_absolute":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachAbsoluteWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_relative":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachRelativeWrapper(self.task, **kwargs)
            else:
                raise ValueError(f"Unknown policy_type: {kwargs['policy_type']}")
        

        if self.obs_wrapper:
            # Note that observation wrapper is not compatible with hierarchical policy
            self.task = ObservationWrapper(self.task, **kwargs)
            self.observation_space = self.task.observation_space

        # Keyframe
        self.keyframe = (
            self.model.key(kwargs["keyframe"]).id
            if "keyframe" in kwargs
            else (0 if self.model.nkey else None)
        )

        self.randomness = randomness
        # Set up named indexing.
        data = MjDataWrapper(self.data)
        model = MjModelWrapper(self.model)
        axis_indexers = index.make_axis_indexers(model)
        self.named = NamedIndexStructs(
            model=index.struct_indexer(model, "mjmodel", axis_indexers),
            data=index.struct_indexer(data, "mjdata", axis_indexers),
        )

        assert self.robot.dof + self.task.dof == len(data.qpos), (
            self.robot.dof,
            self.task.dof,
            len(data.qpos),
        )
        self._contact_label_overrides = self._load_contact_label_overrides(
            self.contact_label_overrides_path
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.task.step(action)
        info = {**info, **self._contact_info()}
        return obs, reward, terminated, truncated, info

    def _contact_info(self):
        count = int(self.data.ncon)
        labels = {
            "contact_any": float(count > 0),
            "contact_count": float(count),
            "contact_hand": 0.0,
            "contact_left_hand": 0.0,
            "contact_right_hand": 0.0,
            "contact_foot": 0.0,
            "contact_left_foot": 0.0,
            "contact_right_foot": 0.0,
            "contact_torso": 0.0,
            "contact_object": 0.0,
            "contact_robot_object": 0.0,
            "contact_hand_object": 0.0,
            "contact_foot_floor": 0.0,
            "contact_robot_floor": 0.0,
            "contact_object_table": 0.0,
            "contact_label_override_used": 0.0,
            "contact_label_unknown_count": 0.0,
        }

        for i in range(count):
            contact = self.data.contact[i]
            geom_names, body_names = self._contact_pair_names(contact)
            pair_key = self._contact_pair_key(geom_names)
            pair_labels = self._contact_label_overrides.get(pair_key)
            if pair_labels is not None:
                labels["contact_label_override_used"] = 1.0
                pair_labels = {
                    **self._contact_rule_labels(geom_names, body_names),
                    **pair_labels,
                }
            else:
                pair_labels = self._contact_rule_labels(geom_names, body_names)
            for key in (
                "contact_hand",
                "contact_left_hand",
                "contact_right_hand",
                "contact_foot",
                "contact_left_foot",
                "contact_right_foot",
                "contact_torso",
                "contact_object",
                "contact_robot_object",
                "contact_hand_object",
                "contact_foot_floor",
                "contact_robot_floor",
                "contact_object_table",
            ):
                labels[key] = max(labels[key], float(pair_labels.get(key, 0.0)))
            labels["contact_label_unknown_count"] += float(
                pair_labels.get("contact_unknown", 0.0)
            )
        return labels

    def _load_contact_label_overrides(self, path):
        if not path:
            return {}
        if not os.path.exists(path):
            raise FileNotFoundError(f"contact_label_overrides not found: {path}")
        overrides = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"geom1", "geom2"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"contact_label_overrides missing columns: {sorted(missing)}"
                )
            for row in reader:
                key = self._contact_pair_key((row.get("geom1", ""), row.get("geom2", "")))
                overrides[key] = {}
                for name in (
                    "contact_hand",
                    "contact_left_hand",
                    "contact_right_hand",
                    "contact_foot",
                    "contact_left_foot",
                    "contact_right_foot",
                    "contact_torso",
                    "contact_object",
                    "contact_robot_object",
                    "contact_hand_object",
                    "contact_foot_floor",
                    "contact_robot_floor",
                    "contact_object_table",
                ):
                    if name in row and row.get(name, "") != "":
                        overrides[key][name] = float(row.get(name, 0.0) or 0.0)
        return overrides

    def _contact_pair_names(self, contact):
        geom_names = []
        body_names = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            geom_names.append(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            )
            body_id = int(self.model.geom_bodyid[geom_id])
            body_names.append(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
        return tuple(geom_names), tuple(body_names)

    @staticmethod
    def _contact_pair_key(geom_names):
        return tuple(sorted(str(name or "") for name in geom_names))

    @staticmethod
    def _contact_rule_labels(geom_names, body_names):
        names = [str(name or "").lower() for name in (*geom_names, *body_names)]

        def any_token(tokens):
            return any(any(token in name for token in tokens) for name in names)

        object_tokens = (
            "object",
            "box",
            "cube",
            "block",
            "peg",
            "door",
            "hatch",
            "cabinet",
            "drawer",
            "handle",
            "target",
            "table",
        )
        hand_tokens = (
            "hand",
            "palm",
            "finger",
            "thumb",
            "wrist",
            "lh_",
            "rh_",
            "left_f",
            "right_f",
            "left_th",
            "right_th",
        )
        foot_tokens = ("foot", "ankle")
        torso_tokens = ("torso", "pelvis", "waist", "hip")
        left_tokens = ("left", "lh_")
        right_tokens = ("right", "rh_")
        floor_tokens = ("floor", "ground")
        table_tokens = ("table",)

        hand = any_token(hand_tokens)
        foot = any_token(foot_tokens)
        torso = any_token(torso_tokens)
        obj = any_token(object_tokens)
        floor = any_token(floor_tokens)
        table = any_token(table_tokens)
        left = any_token(left_tokens)
        right = any_token(right_tokens)
        robot = hand or foot or torso
        unknown = not (hand or foot or torso or obj or floor)
        return {
            "contact_hand": float(hand),
            "contact_left_hand": float(hand and left),
            "contact_right_hand": float(hand and right),
            "contact_foot": float(foot),
            "contact_left_foot": float(foot and left),
            "contact_right_foot": float(foot and right),
            "contact_torso": float(torso),
            "contact_object": float(obj),
            "contact_robot_object": float(obj and robot),
            "contact_hand_object": float(hand and obj),
            "contact_foot_floor": float(foot and floor),
            "contact_robot_floor": float(robot and floor),
            "contact_object_table": float(obj and table),
            "contact_unknown": float(unknown),
        }

    def reset_model(self):
        if self.keyframe is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe)
            mujoco.mj_forward(self.model, self.data)
        else:
            init_qpos = self.model.qpos0.copy()
            qpos0_robot = getattr(self.task, "qpos0_robot", {}).get(self.robot_name)
            if qpos0_robot:
                robot_qpos = np.fromstring(qpos0_robot, sep=" ")
                if robot_qpos.size == self.model.nq:
                    init_qpos = robot_qpos
            self.set_state(init_qpos, np.zeros(self.model.nv))

        # Add randomness
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        r = self.randomness
        self.set_state(
            init_qpos + self.np_random.uniform(-r, r, size=self.model.nq), init_qvel
        )

        # Task-specific reset and return observations
        return self.task.reset_model()

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        return self.task.render()
