"""
===========================================================
PioneerP3DX Navigation Assignment (Gymnasium + CoppeliaSim)
===========================================================

Students MUST implement ALL TODO sections.

Required components to implement:
- __init__()
- reset()
- step()
- _get_obs()

Required elements:
- Action space
- Observation space
- Reward function
- Termination logic
- CoppeliaSim connection

This environment:
- Controls PioneerP3DX from PointA to PointB
- Observation = Dict:
    * "image": RGB camera image (H, W, 3) uint8
    * "prox" : 16 ultrasonic distances float32
- Reward = progress toward goal + shaping for safe navigation
- Termination = goal reached OR collision OR timeout
- Uses manual stepping via client.setStepping(True) for RL determinism
- Includes robust vision unpacking (bytes OR packed table via unpackUInt8Table)
- Includes frame skipping to speed up training
- Includes env.close() to stop simulation cleanly at end
"""

import time
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class DroneEnv(gym.Env):
    """
    NOTE: Name kept as DroneEnv to match your provided template.
    This environment actually controls PioneerP3DX.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        """
        =======================================
        TODO: 1. Connect to CoppeliaSim
        - Initialize the Remote API Client
        - Access the 'sim' object
        - Retrieve all required handles (robot, target, camera, sensors)
        =======================================
        """
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")

        # Manual stepping for RL
        self.client.setStepping(True)

        # Object paths (adjust if your scene differs)
        self.robot_path = "/PioneerP3DX"
        self.pointA_path = "/PointA"
        self.pointB_path = "/PointB"
        self.vision_path = "/PioneerP3DX/visionSensor"

        # Handles (cast to python int -> CBOR-safe)
        self.robot = int(self.sim.getObject(self.robot_path))
        self.left_motor = int(self.sim.getObject(self.robot_path + "/leftMotor"))
        self.right_motor = int(self.sim.getObject(self.robot_path + "/rightMotor"))

        self.pointA = int(self.sim.getObject(self.pointA_path))
        self.pointB = int(self.sim.getObject(self.pointB_path))

        self.camera = int(self.sim.getObject(self.vision_path))

        # Ultrasonic sensors (16)
        self.us_handles = []
        for i in range(16):
            h = int(self.sim.getObject(f"{self.robot_path}/ultrasonicSensor[{i}]"))
            self.us_handles.append(h)

        # Front arc groups (typical P3DX)
        self.front_left_idx = [0, 1, 2, 3]
        self.front_right_idx = [4, 5, 6, 7]

        """
        =======================================
        TODO: 2. Define Action Space
        Suggested:100
        - Discrete(N), e.g., 4 or 6 movement actions
        =======================================
        """
        # 0: forward, 1: left, 2: right, 3: backward, 4: stop
        self.action_space = spaces.Discrete(5)

        """
        =======================================
        TODO: 3. Define Observation Space
        Suggested:
        - RGB camera image with shape (H, W, 3) dtype uint8
        PLUS (improvement):
        - Ultrasonic distances (16,) float32 for wall understanding
        =======================================
        """
        self.IMG_H = 84
        self.IMG_W = 84

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(self.IMG_H, self.IMG_W, 3), dtype=np.uint8
                ),
                "prox": spaces.Box(
                    low=0.0, high=2.0, shape=(16,), dtype=np.float32
                ),
            }
        )

        """
        =======================================
        TODO: 4. Simulation parameters
        - Movement step size
        - Episode length limit
        =======================================
        """
        # Motor speeds (deg/s)
        self.base_speed = 180.0
        self.turn_speed = 140.0
        self.back_speed = 120.0

        # Termination thresholds
        self.goal_tol = 0.25          # meters
        self.collision_dist = 0.12    # meters (front min)

        # Episode limit
        self.max_steps = 2000
        self.steps = 0

        # Reward shaping state
        self.prev_dist_to_goal = None
        self.prev_pos = None
        self.stuck_count = 0

        # Performance: action repeat / frame skip
        self.frame_skip = 4  # big speed + stability boost

        # Track sim state for clean resets
        self._sim_started = False


    # -------------------- Internal helpers -------------------- #
    def _deg_to_rad(self, deg_s: float) -> float:
        return deg_s * math.pi / 180.0

    def _set_wheels_deg(self, left_deg_s: float, right_deg_s: float):
        self.sim.setJointTargetVelocity(self.left_motor, self._deg_to_rad(left_deg_s))
        self.sim.setJointTargetVelocity(self.right_motor, self._deg_to_rad(right_deg_s))

    def _stop_robot(self):
        self._set_wheels_deg(0.0, 0.0)

    def _read_ultrasonics(self) -> np.ndarray:
        """
        Returns distances[16] in meters, non-detections clamped to max_range.
        """
        max_range = 2.0
        d = np.zeros(16, dtype=np.float32)
        for i, h in enumerate(self.us_handles):
            state, dist, pt, obj, normal = self.sim.readProximitySensor(int(h))
            if dist == 0.0:
                dist = max_range
            d[i] = dist
        return d

    def _dist_to_goal(self) -> float:
        p = np.array(self.sim.getObjectPosition(self.robot, -1), dtype=np.float32)
        g = np.array(self.sim.getObjectPosition(self.pointB, -1), dtype=np.float32)
        return float(np.linalg.norm((g - p)[:2]))

    def _robot_pos(self) -> np.ndarray:
        return np.array(self.sim.getObjectPosition(self.robot, -1), dtype=np.float32)

    def _wait_for_stop(self, max_iter=600):
        # Stop sim and step until it is stopped
        try:
            self.sim.stopSimulation()
        except Exception:
            return
        for _ in range(max_iter):
            state = self.sim.getSimulationState()
            if state == self.sim.simulation_stopped:
                self._sim_started = False
                return
            try:
                self.sim.step()
            except Exception:
                break

    def _wait_for_start(self, max_iter=600):
        try:
            self.sim.startSimulation()
        except Exception:
            return
        for _ in range(max_iter):
            state = self.sim.getSimulationState()
            if state == self.sim.simulation_advancing:
                self._sim_started = True
                return
            try:
                self.sim.step()
            except Exception:
                break


    # ============================================================
    # TODO: OBSERVATION FUNCTION
    # Must return an RGB image + ultrasonic distances.
    # ============================================================
    def _get_obs(self):
        """
        TODO:
        - Retrieve raw image from the CoppeliaSim vision sensor
        - Convert to a uint8 numpy array
        - Resize to (IMG_H, IMG_W)
        - Return a dict observation: {"image": img, "prox": distances}
        """
        data, resolution = self.sim.getVisionSensorImg(self.camera)
        w, h = int(resolution[0]), int(resolution[1])

        # Robust decoding: bytes OR packed table
        if isinstance(data, (bytes, bytearray, memoryview)):
            img = np.frombuffer(data, dtype=np.uint8)
        else:
            img = self.sim.unpackUInt8Table(data)
            img = np.array(img, dtype=np.uint8)

        # Coppelia image is typically "upside down" -> flip vertically
        img = img.reshape((h, w, 3))
        img = np.flipud(img)

        # Resize for RL
        img = cv2.resize(img, (self.IMG_W, self.IMG_H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.uint8)

        prox = self._read_ultrasonics().astype(np.float32)

        return {"image": img, "prox": prox}


    # ============================================================
    # TODO: RESET FUNCTION
    # ============================================================
    def reset(self, seed=None, options=None):
        """
        TODO:
        1. Stop the CoppeliaSim simulation (blocking loop)
        2. Restart simulation cleanly
        3. Reset robot position to PointA
        4. Reset target position (PointB fixed here)
        5. Reset counters/state
        6. Return initial observation
        """
        super().reset(seed=seed)

        # Clean restart
        self._wait_for_stop()
        self._wait_for_start()

        # Reset pose to PointA
        posA = self.sim.getObjectPosition(self.pointA, -1)
        oriA = self.sim.getObjectOrientation(self.pointA, -1)
        self.sim.setObjectPosition(self.robot, -1, posA)
        self.sim.setObjectOrientation(self.robot, -1, oriA)

        self._stop_robot()

        # Settle a few steps
        for _ in range(5):
            self.sim.step()

        self.steps = 0
        self.prev_dist_to_goal = self._dist_to_goal()
        self.prev_pos = self._robot_pos()
        self.stuck_count = 0

        obs = self._get_obs()
        info = {"dist_to_goal": self.prev_dist_to_goal}
        return obs, info


    # ============================================================
    # TODO: STEP FUNCTION
    # ============================================================
    def step(self, action):
        """
        TODO:
        1. Apply movement based on 'action'
        2. Step the simulation forward (frame_skip times)
        3. Collect observation via _get_obs()
        4. Compute reward
        5. Check termination (goal/collision/timeout)
        6. Return: (obs, reward, terminated, truncated, info)
        """

        # 1) Apply action -> wheel commands
        if action == 0:      # forward
            self._set_wheels_deg(self.base_speed, self.base_speed)
        elif action == 1:    # turn left
            self._set_wheels_deg(self.base_speed - self.turn_speed,
                                 self.base_speed + self.turn_speed)
        elif action == 2:    # turn right
            self._set_wheels_deg(self.base_speed + self.turn_speed,
                                 self.base_speed - self.turn_speed)
        elif action == 3:    # backward
            self._set_wheels_deg(-self.back_speed, -self.back_speed)
        elif action == 4:    # stop
            self._stop_robot()
        else:
            self._stop_robot()

        # 2) Step sim forward (frame skip)
        for _ in range(self.frame_skip):
            self.sim.step()

        # 3) Obs
        obs = self._get_obs()

        # 4) Reward shaping
        self.steps += 1

        dist = self._dist_to_goal()
        prev = self.prev_dist_to_goal if self.prev_dist_to_goal is not None else dist
        progress = float(prev - dist)
        self.prev_dist_to_goal = dist

        prox = obs["prox"]
        fl = float(np.min(prox[self.front_left_idx]))
        fr = float(np.min(prox[self.front_right_idx]))
        fmin = min(fl, fr)

        # Base: reward progress, small time penalty
        reward = 4.0 * progress - 0.001

        # Encourage forward to break "dither near wall" behavior
        if action == 0:
            reward += 0.002

        # Penalize proximity to obstacles smoothly (agent also observes prox, so this is consistent)
        if fmin < 0.4:
            reward -= float(0.4 - fmin) * 0.4

        # Penalize "stuck" (no movement)
        pos = self._robot_pos()
        moved = float(np.linalg.norm((pos - self.prev_pos)[:2]))
        self.prev_pos = pos

        if moved < 0.002:  # ~2mm (tune)
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count > 40:
            reward -= 0.2  # nudge it out of deadlock
            self.stuck_count = 0

        # 5) Termination logic
        terminated = False
        truncated = False

        # Goal reached
        if dist <= self.goal_tol:
            terminated = True
            reward += 10.0
            self._stop_robot()

        # Collision
        if fmin <= self.collision_dist:
            terminated = True
            reward -= 10.0
            self._stop_robot()

        # Timeout
        if self.steps >= self.max_steps:
            truncated = True
            self._stop_robot()

        info = {
            "dist_to_goal": dist,
            "front_min": fmin,
            "front_left": fl,
            "front_right": fr,
            "steps": self.steps,
        }

        return obs, reward, terminated, truncated, info


    # ============================================================
    # Clean shutdown (stops sim when training ends)
    # ============================================================
    def close(self):
        try:
            self._stop_robot()
            self.sim.stopSimulation()
            # In stepping mode, give it some steps to transition to stopped
            for _ in range(50):
                self.sim.step()
        except Exception:
            pass
        super().close()


# ============================================================
# Optional: main block (training with PPO)
# ============================================================
if __name__ == "__main__":
    """
    Example PPO training:

    pip install stable-baselines3[extra]

    IMPORTANT:
    - Dict observation => use "MultiInputPolicy"
    - No VecTransposeImage needed for MultiInputPolicy
      (SB3 handles image internally for dict obs)
    """

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(DroneEnv())
    check_env(env, warn=True)

    venv = DummyVecEnv([lambda: env])

    model = PPO(
        "MultiInputPolicy",
        venv,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    try:
        model.learn(total_timesteps=100_000)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        model.save("ppo_pioneer_multiinput2")
        print("Model saved!")
        venv.close()
