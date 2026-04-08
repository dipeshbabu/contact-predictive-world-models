import numpy as np
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task


_STAND_HEIGHT = 1.65


class BaseLocomotion(Task):
    target_speed = 1.0
    camera_name = "cam_default"
    max_episode_steps = 1000
    dof = 0
    success_bar = 700

    qpos0_robot = {
        "h1touch": """
            0 0 0.98 1 0 0 0
            0 0 -0.4 0.8 -0.4 0
            0 -0.4 0.8 -0.4 0
            0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0
        """,
    }

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1,),
            dtype=np.float64,
        )

    def get_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0.0,
        )
        stand_reward = standing * upright

        x_vel = float(self.robot.center_of_mass_velocity()[0])
        move_reward = rewards.tolerance(
            x_vel,
            bounds=(self.target_speed, float("inf")),
            margin=self.target_speed,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10.0,
            value_at_margin=0.0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4.0 + small_control) / 5.0

        reward = stand_reward * (0.2 * small_control + 0.8 * move_reward)
        success = bool(stand_reward > 0.8 and x_vel > 0.8 * self.target_speed)

        return reward, {
            "stand_reward": stand_reward,
            "move_reward": move_reward,
            "small_control": small_control,
            "x_velocity": x_vel,
            "success": success,
        }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.58, {}


class Walk(BaseLocomotion):
    target_speed = 1.0
    success_bar = 700


class Run(BaseLocomotion):
    target_speed = 1.8
    success_bar = 900
