import os
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
    NOTE: Name kept as DroneEnv to match template.
    This environment controls the PioneerP3DX robot.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # -----------------------------
        # 1) Connect to CoppeliaSim
        # -----------------------------
        host = os.getenv("COPPELIA_HOST", "127.0.0.1")
        port = int(os.getenv("COPPELIA_PORT", "23000"))

        print(f"ENV: connecting to CoppeliaSim at {host}:{port}", flush=True)
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require("sim")
        print("ENV: connected and got sim", flush=True)


        # Manual stepping for RL determinism
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

        # Front arc groups (typical P3DX layout)
        self.front_left_idx = [0, 1, 2, 3]
        self.front_right_idx = [4, 5, 6, 7]

        # -----------------------------
        # 2) Action space
        # -----------------------------
        # 0: forward, 1: left, 2: right, 3: backward, 4: stop
        self.action_space = spaces.Discrete(5)

        # -----------------------------
        # 3) Observation space (Dict)
        # IMPORTANT: image is channel-first (C,H,W)
        # -----------------------------
        self.IMG_H = 84
        self.IMG_W = 84

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.IMG_H, self.IMG_W),  # CHW
                    dtype=np.uint8,
                ),
                "prox": spaces.Box(
                    low=0.0,
                    high=2.0,
                    shape=(16,),
                    dtype=np.float32,
                ),
            }
        )

        # -----------------------------
        # 4) Simulation parameters
        # -----------------------------
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
        self.frame_skip = 4  # speeds up training + smoother motion

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
    # OBSERVATION FUNCTION
    # Returns {"image": CHW uint8, "prox": float32(16)}
    # ============================================================
    def _get_obs(self):
        data, resolution = self.sim.getVisionSensorImg(self.camera)
        w, h = int(resolution[0]), int(resolution[1])

        # Robust decoding: bytes OR packed table
        if isinstance(data, (bytes, bytearray, memoryview)):
            img = np.frombuffer(data, dtype=np.uint8)
        else:
            img = self.sim.unpackUInt8Table(data)
            img = np.array(img, dtype=np.uint8)

        # Coppelia images are commonly vertically flipped
        img = img.reshape((h, w, 3))
        img = np.flipud(img)

        # Resize to (H, W)
        img = cv2.resize(img, (self.IMG_W, self.IMG_H), interpolation=cv2.INTER_AREA)

        # IMPORTANT: convert to channel-first (C,H,W)
        #img = np.transpose(img, (2, 0, 1)).astype(np.uint8)
        img = np.transpose(img, (2, 0, 1)).astype(np.uint8)
        prox = self._read_ultrasonics().astype(np.float32)

        return {"image": img, "prox": prox}

    # ============================================================
    # RESET FUNCTION
    # ============================================================
    def reset(self, seed=None, options=None):
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
    # STEP FUNCTION
    # ============================================================
    def step(self, action):
        # 1) Apply action -> wheel commands
        if action == 0:      # forward
            self._set_wheels_deg(self.base_speed, self.base_speed)
        elif action == 1:    # turn left
            self._set_wheels_deg(
                self.base_speed - self.turn_speed,
                self.base_speed + self.turn_speed
            )
        elif action == 2:    # turn right
            self._set_wheels_deg(
                self.base_speed + self.turn_speed,
                self.base_speed - self.turn_speed
            )
        elif action == 3:    # backward
            self._set_wheels_deg(-self.back_speed, -self.back_speed)
        else:                # stop
            self._stop_robot()

        # 2) Step sim forward (frame skip)
        for _ in range(self.frame_skip):
            self.sim.step()

        # 3) Observation
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

        # Encourage forward to break "dithering near wall"
        if action == 0:
            reward += 0.002

        # Penalize proximity to obstacles smoothly
        if fmin < 0.4:
            reward -= float(0.4 - fmin) * 0.4

        # Penalize being stuck (no movement)
        pos = self._robot_pos()
        moved = float(np.linalg.norm((pos - self.prev_pos)[:2]))
        self.prev_pos = pos

        if moved < 0.002:  # ~2mm
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count > 40:
            reward -= 0.2
            self.stuck_count = 0

        # 5) Termination logic
        terminated = False
        truncated = False

        if dist <= self.goal_tol:
            terminated = True
            reward += 10.0
            self._stop_robot()

        if fmin <= self.collision_dist:
            terminated = True
            reward -= 10.0
            self._stop_robot()

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
    # Clean shutdown (stops sim when training/inference ends)
    # ============================================================
    def close(self):
        try:
            self._stop_robot()
            self.sim.stopSimulation()
            for _ in range(50):
                self.sim.step()
        except Exception:
            pass
        super().close()
