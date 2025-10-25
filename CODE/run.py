

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import math
import csv
import threading
from typing import List, Union

import numpy as np

# Gym / SB3
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise



# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# AirSim
import airsim

# VX_MIN, VX_MAX = 0.6, 3.5
# VY_MAX = 1.5
# VZ_MAX_UP = 1.0
# YAW_RATE_CMD_MAX_DEGPS = 145.0
# CHECK_HZ = 15
# CMD_DT = 1.0 / CHECK_HZ
# W_PROGRESS = 12.0
# W_CLEAR = 0.6
# W_CENTER = 0.1
# W_ALT = 0.5
# W_SMOOTH = 0.02
# CLEAR_SAFE_M = 2.8
# ALT_GOAL_TOL_M = 0.6

# ======================= Global tunables =======================
# Vehicle/LiDAR default names (instances can override)
VEHICLE_NAME_DEFAULT = "Drone1"
LIDAR_NAME_DEFAULT   = "Lidar1"     # must exist on the json; H-FOV 360; DataFrame=SensorLocalFrame

# Increasing/Descresing this number determines Conservstive vs Agressive
CHECK_HZ = 6
CMD_DT   = 1.0 / CHECK_HZ

# Action limits / mappings
YAW_RATE_CMD_MAX_DEGPS = 145.0      # deg/s for sharper turns
VX_MIN, VX_MAX = 1.2, 7.0        # m/s forward
VY_MAX = 1.5
VZ_MAX_UP = 1.0

# LiDAR ring
BINS             = 360
RANGE_CLIP_MIN   = 0.20
RANGE_CLIP_MAX   = 20.0
USE_HEIGHT_BAND  = (-0.7, 1.8)

# Observation sizing
OBS_BINS = 60                         # 360 -> 60 buckets for the lidar ring

# Goal handling
GOAL_OBJECT_NAME = "landing_zone"
GOAL_RADIUS_M    = 1.5
ALT_GOAL_TOL_M   = 0.6
GOAL_NORM_M      = 25.0

# Collisions to ignore 
IGNORE_COLLISION_TOKENS = {"takeoff_zone"}

# Altitude target (above start); fights slow drift toward ground
ALT_TARGET_M  = 2.0
ALT_NORM_M    = 3.0

# Rewards (shaping)
R_SUCCESS     = 100.0
R_COLLISION   = 100.0
W_PROGRESS    = 30.0
W_CLEAR       = 0.6
W_CENTER      = 0.1
W_ALT         = 0.5
W_SMOOTH      = 0.02

# Forward clearance window
FOV_FRONTAL_DEG = 85
CLEAR_SAFE_M    = 2.8

# Auto-land behavior
AUTO_LAND_ON_GOAL = True
GOAL_APPROACH_RADIUS_M = 2.5
LAND_HOVER_SEC = 0.3

COLLISION_GRACE_SEC = 0.8


# ======================= Utility functions =======================
def clamp(v, lo, hi): return max(lo, min(hi, v))
def wrap_deg(d): return (d + 180.0) % 360.0 - 180.0

def quat_to_yaw(q):
    # AirSim quaternion (w,x,y,z) -> yaw (rad)
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def world_to_body_xy(yaw, vx_w, vy_w):
    # rotate world -> body XY (0 yaw: body aligns with world X)
    c, s = math.cos(yaw), math.sin(yaw)
    vx_b =  c*vx_w + s*vy_w
    vy_b = -s*vx_w + c*vy_w
    return vx_b, vy_b

def rotate_world_to_body_xy(yaw, dx_w, dy_w):
    c, s = math.cos(yaw), math.sin(yaw)
    x_b =  c*dx_w + s*dy_w
    y_b = -s*dx_w + c*dy_w
    return x_b, y_b

def scan_to_ring(pts):
    
    #360-bin ring where index 0 (0°) is straight ahead (body +x), angles increase to the left, wrap at 360.
 
    if pts.shape[0] == 0:
        return np.full(BINS, RANGE_CLIP_MAX, dtype=np.float32)
    angles = np.degrees(np.arctan2(pts[:,1], pts[:,0]))   # [-180,180]
    ranges = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    ranges = np.clip(ranges, RANGE_CLIP_MIN, RANGE_CLIP_MAX)
    bins = np.floor((angles + 360.0) % 360.0).astype(int)
    ring = np.full(BINS, RANGE_CLIP_MAX, dtype=np.float32)
    np.minimum.at(ring, bins, ranges)
    for _ in range(2):
        ring = np.minimum(ring, np.roll(ring, 1))
        ring = np.minimum(ring, np.roll(ring, -1))
    return ring

def downsample_ring(ring, obs_bins=OBS_BINS):
    stride = BINS // obs_bins
    ring_blocks = ring[:obs_bins*stride].reshape(obs_bins, stride)
    return ring_blocks.min(axis=1).astype(np.float32)

def forward_sector_indices(fov_deg):
    half = int(fov_deg // 2)
    return np.r_[range(360 - half, 360), range(0, half + 1)]

FWD_IDX = forward_sector_indices(FOV_FRONTAL_DEG)

def forward_min_clearance(ring):
    return float(np.min(ring[FWD_IDX]))

def left_right_mean_clearance(ring):
    half = int(FOV_FRONTAL_DEG // 2)
    left_idx  = np.arange(1, half+1)                    # 1..+half (left)
    right_idx = np.r_[np.arange(360-half, 360), 0]      # -half..0 (right)
    left_mean  = float(np.mean(ring[left_idx]))
    right_mean = float(np.mean(ring[right_idx]))
    return left_mean, right_mean


# ======================= DroneEnv (vehicle/lidar) =======================
class DroneEnv(gym.Env):
    """
    AirSim multirotor env with LiDAR avoidance + goal seeking.
    Args:
        vehicle_name (str): AirSim vehicle to control (e.g., "Drone1", "Drone2").
        lidar_name   (str): LiDAR sensor name on that vehicle.
        reset_mode   (str): "sim" (global client.reset) OR "vehicle" (no global reset).
                            Use "vehicle" for multi-eval so drones don't reset each other.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 vehicle_name: str = VEHICLE_NAME_DEFAULT,
                 lidar_name:   str = LIDAR_NAME_DEFAULT,
                 reset_mode:   str = "sim"):
        super().__init__()
        self.vehicle_name = vehicle_name
        self.lidar_name   = lidar_name
        self.reset_mode   = reset_mode  # "sim" or "vehicle"

        # Actions: a ∈ [-1,1]^4 - [yaw_rate, vx, vy, vz_up]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observations: [ring_ds_norm (0..1)^OBS_BINS, vx_b, vy_b, vz_up, alt_err, goal_xb, goal_yb, goal_z_up, dist_norm]
        low  = np.concatenate([np.zeros(OBS_BINS, dtype=np.float32),  # ring
                               np.full(3, -1.0, dtype=np.float32),    # vx, vy, vz_up
                               np.full(1, -1.0, dtype=np.float32),    # alt_err
                               np.full(3, -1.0, dtype=np.float32),    # goal vec
                               np.array([0.0], dtype=np.float32)])    # dist_norm (0..1)
        high = np.concatenate([np.ones(OBS_BINS, dtype=np.float32),
                               np.full(3,  1.0, dtype=np.float32),
                               np.full(1,  1.0, dtype=np.float32),
                               np.full(3,  1.0, dtype=np.float32),
                               np.array([1.0], dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # Episode state
        self.step_count = 0
        self.episode_idx = 0
        self.start_z_ned = None
        self.goal_world = None
        self.prev_goal_dist = None
        self.prev_cmd_phys = np.zeros(4, dtype=np.float32)  # [yaw_rate, vx, vy, vz_up]
        self._grace_until = 0.0

    # ------------- Core API -------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # progress print for last episode
        if getattr(self, "step_count", 0) > 0:
            print(f"[{self.vehicle_name}] episode {self.episode_idx} steps={self.step_count}")
        self.episode_idx += 1
        self.step_count = 0

        # Reset world env
        if self.reset_mode == "sim":
            self.client.reset()

        # Ensure control for this vehicle only
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True,  self.vehicle_name)

        # Goal pose
        gpose = self.client.simGetObjectPose(GOAL_OBJECT_NAME)
        if not (np.isfinite(gpose.position.x_val) and np.isfinite(gpose.position.y_val) and np.isfinite(gpose.position.z_val)):
            
            vpose = self.client.simGetVehiclePose(self.vehicle_name)
            self.goal_world = np.array([vpose.position.x_val + 30.0, vpose.position.y_val, vpose.position.z_val], dtype=np.float32)
        else:
            self.goal_world = np.array([gpose.position.x_val, gpose.position.y_val, gpose.position.z_val], dtype=np.float32)

        # Takeoff
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.2)

        # Move to nominal altitude ALT_TARGET_M above start
        vpose = self.client.simGetVehiclePose(self.vehicle_name)
        self.start_z_ned = vpose.position.z_val
        z_target = self.start_z_ned - ALT_TARGET_M   # NED: In AirSim API up means more negative
        self.client.moveToZAsync(z=z_target, velocity=1.0, vehicle_name=self.vehicle_name).join()
        time.sleep(0.2)

        #  random initial yaw to simulate inital noise
        yaw0 = float(np.random.uniform(-15.0, 15.0))
        self.client.rotateToYawAsync(yaw0, 5, vehicle_name=self.vehicle_name).join()
        time.sleep(0.1)

        # Collision grace window (ignore spawn jitter touches)
        self._grace_until = time.time() + COLLISION_GRACE_SEC

        # logging
        self.prev_cmd_phys = np.zeros(4, dtype=np.float32)
        obs = self._get_obs()
        self.prev_goal_dist = self._get_goal_distance()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        a = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)

        # ----- Map action to drone motion commands -----
        yaw_rate = float(a[0] * YAW_RATE_CMD_MAX_DEGPS)                     # deg/s
        vx       = float(VX_MIN + 0.5*(a[1] + 1.0) * (VX_MAX - VX_MIN))     # [VX_MIN, VX_MAX]
        vy       = float(a[2] * VY_MAX)                                     # [-VY_MAX, VY_MAX]
        vz_up    = float(a[3] * VZ_MAX_UP)                                  # [-VZ_MAX_UP, VZ_MAX_UP]
        vz_ned   = -vz_up                                                   # NED: up-positive → negative

        # Always-forward velocities in BODY frame + yaw rate control
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz_ned, CMD_DT, yaw_mode=yaw_mode, vehicle_name=self.vehicle_name)
        time.sleep(CMD_DT)

        # ----- Observation -----
        obs = self._get_obs()

        # ----- Rewards -----
        ring = self._get_ring()
        min_front = forward_min_clearance(ring)
        left_c, right_c = left_right_mean_clearance(ring)

        # Goal progress
        cur_dist = self._get_goal_distance()
        progress = (self.prev_goal_dist - cur_dist)   # >0 when moving closer
        r_progress = W_PROGRESS * progress

        # Clearance encouragement (0..1 around CLEAR_SAFE_M)
        r_clear = W_CLEAR * clamp(min_front / CLEAR_SAFE_M, 0.0, 1.0)

        # Corridor centering (prefer equal L/R in forward window)
        denom = (left_c + right_c + 1e-6)
        r_center = W_CENTER * clamp(1.0 - abs(left_c - right_c) / denom, 0.0, 1.0)

        # Altitude penalty (normalize by ALT_NORM_M)
        alt_err = self._get_alt_error()
        r_alt = - W_ALT * clamp(abs(alt_err) / ALT_NORM_M, 0.0, 1.0)

        # Smoothness penalty (command deltas in physical units)
        cmd_phys = np.array([yaw_rate, vx, vy, vz_up], dtype=np.float32)
        r_smooth = - W_SMOOTH * float(np.linalg.norm(cmd_phys - self.prev_cmd_phys))
        self.prev_cmd_phys = cmd_phys

        # --- Collision / success / auto-land function activation---
        col_info  = self.client.simGetCollisionInfo(self.vehicle_name)
        raw_collided = bool(col_info.has_collided)
        obj_name = (getattr(col_info, "object_name", "") or "")
        name_l = obj_name.lower()

        # True success if drone touches the actual landing object named in Unreal
        landed_on_goal = raw_collided and (obj_name == GOAL_OBJECT_NAME or obj_name == "END_WALL")

        # Airsim API treats initial start as a collision with ground - ignore collisions with takeoff object in Unreal
        ignored_hit = raw_collided and (not landed_on_goal) and any(tok in name_l for tok in IGNORE_COLLISION_TOKENS)

        # Ignore collisions during the grace window after reset
        in_grace = (time.time() < self._grace_until)

       
        collided = raw_collided and (not ignored_hit) and (not in_grace)

        at_goal  = self._goal_reached()
        should_land = AUTO_LAND_ON_GOAL and (cur_dist <= GOAL_APPROACH_RADIUS_M)

        r_done = 0.0
        terminated = False
        term_reason = None

        if landed_on_goal or at_goal or should_land:
            if AUTO_LAND_ON_GOAL:
                self._auto_land(goal=self.goal_world)
            r_done += R_SUCCESS
            terminated = True
            term_reason = "success"

        elif collided:
            r_done -= R_COLLISION
            terminated = True
            term_reason = "collision"

        elif ignored_hit:
           
            term_reason = "ignored_collision"

        truncated = False  # no timeouts

        reward = r_progress + r_clear + r_center + r_alt + r_smooth + r_done
        self.prev_goal_dist = cur_dist

        info = {
            "min_front": min_front,
            "left_mean": left_c,
            "right_mean": right_c,
            "progress": progress,
            "alt_err": alt_err,
            "cmd": {"yaw_rate": yaw_rate, "vx": vx, "vy": vy, "vz_up": vz_up},
            "terminated_reason": term_reason
        }
        return obs, reward, terminated, truncated, info

    # ------------- Helpers -------------
    def _lidar_points(self):
      
        data = self.client.getLidarData(lidar_name=self.lidar_name, vehicle_name=self.vehicle_name)
        pts = np.array(data.point_cloud, dtype=np.float32)
        if pts.size == 0:
            return np.empty((0,3), dtype=np.float32)
        pts = pts.reshape(-1,3)
        if USE_HEIGHT_BAND is not None:
            zmin, zmax = USE_HEIGHT_BAND
            mask = (pts[:,2] >= zmin) & (pts[:,2] <= zmax)
            pts = pts[mask]
        return pts

    def _get_ring(self):
        pts = self._lidar_points()
        return scan_to_ring(pts)

    def _get_obs(self):
        ring = self._get_ring()
        ring_ds = downsample_ring(ring)
        ring_norm = np.clip(ring_ds / RANGE_CLIP_MAX, 0.0, 1.0)

        # Pose/vel
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        yaw = quat_to_yaw(state.kinematics_estimated.orientation)
        vel = state.kinematics_estimated.linear_velocity
        pos = state.kinematics_estimated.position  # sometimes zeros; fallback to sim pose
        if not np.isfinite(pos.x_val):
            vpose = self.client.simGetVehiclePose(self.vehicle_name)
            pos = vpose.position
            yaw = quat_to_yaw(vpose.orientation)

        # Body velocities
        vx_b, vy_b = world_to_body_xy(yaw, vel.x_val, vel.y_val)
        vz_up = -vel.z_val  # NED to up-positive

        # Normalize velocities
        vx_b_n = clamp(vx_b / VX_MAX, -1.0, 1.0)
        vy_b_n = clamp(vy_b / VY_MAX, -1.0, 1.0)
        vz_up_n= clamp(vz_up / VZ_MAX_UP, -1.0, 1.0)

        # Altitude error (target above start)
        alt_err = self._get_alt_error()
        alt_err_n = clamp(alt_err / ALT_NORM_M, -1.0, 1.0)

        # Goal vector in BODY frame + distance
        dx_w, dy_w, dz_up = self._goal_delta_world_up()
        gx_b, gy_b = rotate_world_to_body_xy(yaw, dx_w, dy_w)

        gx_n = clamp(gx_b / GOAL_NORM_M, -1.0, 1.0)
        gy_n = clamp(gy_b / GOAL_NORM_M, -1.0, 1.0)
        gz_n = clamp(dz_up / GOAL_NORM_M, -1.0, 1.0)
        dist = math.sqrt(dx_w*dx_w + dy_w*dy_w + dz_up*dz_up)
        dist_n = clamp(dist / GOAL_NORM_M, 0.0, 1.0)

        extras = np.array([vx_b_n, vy_b_n, vz_up_n, alt_err_n, gx_n, gy_n, gz_n, dist_n], dtype=np.float32)
        return np.concatenate([ring_norm, extras]).astype(np.float32)

    def _get_goal_distance(self):
        dx_w, dy_w, dz_up = self._goal_delta_world_up()
        return math.sqrt(dx_w*dx_w + dy_w*dy_w + dz_up*dz_up)

    def _auto_land(self, goal=None):
        try:
            if goal is not None:
                vpose = self.client.simGetVehiclePose(self.vehicle_name)
                self.client.moveToPositionAsync(
                    goal[0], goal[1], vpose.position.z_val, 1.0,
                    vehicle_name=self.vehicle_name
                ).join()
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            time.sleep(LAND_HOVER_SEC)
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass

    def _goal_reached(self):
        d = self._get_goal_distance()
        alt_err = abs(self._get_alt_error())
        return (d <= GOAL_RADIUS_M) and (alt_err <= ALT_GOAL_TOL_M)

    def _goal_delta_world_up(self):
        vpose = self.client.simGetVehiclePose(self.vehicle_name)
        cx, cy, cz = vpose.position.x_val, vpose.position.y_val, vpose.position.z_val
        gx, gy, gz = self.goal_world[0], self.goal_world[1], self.goal_world[2]
        dx_w = gx - cx
        dy_w = gy - cy
        dz_up = -(gz - cz)   # NED → up-positive
        return dx_w, dy_w, dz_up

    def _get_alt_error(self):
        vpose = self.client.simGetVehiclePose(self.vehicle_name)
        z_ned = vpose.position.z_val
        alt_above_start = (self.start_z_ned - z_ned)  # up-positive meters
        return (ALT_TARGET_M - alt_above_start)

    def close(self):
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        self.client.enableApiControl(False, self.vehicle_name)


# ======================= Train logger (TB + CSV; per-episode and optional per-step) =======================
class TrainLogger(BaseCallback):
    """
    TensorBoard + CSV logging for train/cont runs.
    log_every:
      - "episode"  -> log once per episode (default)
      - "step"     -> log every step
      - int N>0    -> log every N steps
    """
    def __init__(self, folder, tag="train", log_every="episode"):
        super().__init__()
        self.folder = folder
        self.tag = tag

      
        if isinstance(log_every, str):
            v = log_every.lower()
            if v == "episode":
                self.step_stride = None
            elif v == "step":
                self.step_stride = 1
            else:
                raise ValueError("log_every must be 'episode', 'step', or int>0")
        elif isinstance(log_every, int) and log_every > 0:
            self.step_stride = int(log_every)
        else:
            raise ValueError("log_every must be 'episode', 'step', or int>0")

        
        self.tb_dir    = os.path.join(self.folder, "tensors")
        self.ep_csvdir = os.path.join(self.folder, "train_logs")
        self.step_dir  = os.path.join(self.folder, "step_logs") if self.step_stride is not None else None
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.ep_csvdir, exist_ok=True)
        if self.step_dir: os.makedirs(self.step_dir, exist_ok=True)

        # episode accumulators
        self.ep_return = 0.0
        self.ep_steps  = 0
        self.ep_index  = 0
        self.global_step = 0  # TB x-axis for step logs

        # file handles
        self._ep_csv_path = os.path.join(self.ep_csvdir, f"{self.tag}_episodes.csv")
        self._ep_csv_file = None
        self._ep_csv_writer = None
        self._step_csv_file = None
        self._step_csv_writer = None

        self.writer = None

    def _on_training_start(self) -> None:
        # TB writer
        self.writer = SummaryWriter(self.tb_dir)
        # episode CSV file
        is_new = not os.path.exists(self._ep_csv_path)
        self._ep_csv_file = open(self._ep_csv_path, "a", newline="")
        self._ep_csv_writer = csv.writer(self._ep_csv_file)
        if is_new:
            self._ep_csv_writer.writerow(["episode", "steps", "return", "result"])

    def _start_step_csv(self):
        if self.step_stride is None: return
        step_csv = os.path.join(self.step_dir, f"{self.tag}_steps_ep{self.ep_index}.csv")
        self._step_csv_file = open(step_csv, "w", newline="")
        self._step_csv_writer = csv.writer(self._step_csv_file)
        self._step_csv_writer.writerow([
            "t", "reward", "dist_to_goal", "min_front",
            "yaw_rate", "vx", "vy", "vz_up", "alt_err", "terminated_reason"
        ])

    def _close_step_csv(self):
        if self._step_csv_file:
            self._step_csv_file.close()
            self._step_csv_file = None
            self._step_csv_writer = None

    def _on_step(self) -> bool:
       
        infos   = self.locals.get("infos", [])
        dones   = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        info = infos[0] if infos else {}
        done = bool(dones[0]) if len(dones) > 0 else False
        r = float(rewards[0]) if len(rewards) > 0 else 0.0

        # on new episode start, open step CSV 
        if self.ep_steps == 0 and self.step_stride is not None and self._step_csv_file is None:
            self._start_step_csv()

        # accumulate
        self.ep_steps  += 1
        self.ep_return += r

        # step-level logging
        if self.step_stride is not None and (self.ep_steps % self.step_stride == 0):
           
            env = self.training_env.envs[0].unwrapped
            dist_to_goal = getattr(env, "_get_goal_distance", lambda: float("nan"))()
            min_front = float(info.get("min_front", np.nan))
            cmd = info.get("cmd", {})
            yaw_rate = float(cmd.get("yaw_rate", np.nan))
            vx = float(cmd.get("vx", np.nan))
            vy = float(cmd.get("vy", np.nan))
            vz_up = float(cmd.get("vz_up", np.nan))
            alt_err = float(info.get("alt_err", np.nan))
            term_reason = info.get("terminated_reason", "")

            # CSV
            if self._step_csv_writer:
                self._step_csv_writer.writerow([
                    self.ep_steps, r, dist_to_goal, min_front,
                    yaw_rate, vx, vy, vz_up, alt_err, term_reason
                ])

            # TB
            self.writer.add_scalar(f"{self.tag}_step/reward",       r,               self.global_step)
            self.writer.add_scalar(f"{self.tag}_step/dist_to_goal", dist_to_goal,    self.global_step)
            self.writer.add_scalar(f"{self.tag}_step/min_front",    min_front,       self.global_step)
            self.writer.add_scalar(f"{self.tag}_step/yaw_rate",     yaw_rate,        self.global_step)
            self.writer.add_scalar(f"{self.tag}_step/vx",           vx,              self.global_step)
            self.writer.add_scalar(f"{self.tag}_step/vy",           vy,              self.global_step)
            self.writer.add_scalar(f"{self.tag}_step/vz_up",        vz_up,           self.global_step)
            self.writer.add_scalar(f"{self.tag}_step/alt_err",      alt_err,         self.global_step)
            self.global_step += 1

       
        if done:
            reason = info.get("terminated_reason") or "done"
            success = 1 if reason == "success" else 0

            # TB per-episode
            self.writer.add_scalar(f"{self.tag}/episode_return", self.ep_return, self.ep_index)
            self.writer.add_scalar(f"{self.tag}/episode_length", self.ep_steps,  self.ep_index)
            self.writer.add_scalar(f"{self.tag}/success",        success,        self.ep_index)

            # CSV per-episode
            self._ep_csv_writer.writerow([self.ep_index, self.ep_steps, self.ep_return, reason])
            self._ep_csv_file.flush()

            # reset counters for next episode
            self.ep_index  += 1
            self.ep_steps   = 0
            self.ep_return  = 0.0
            self._close_step_csv()

        return True

    def _on_training_end(self) -> None:
        self._close_step_csv()
        if self.writer:
            self.writer.flush()
            self.writer.close()
        if self._ep_csv_file:
            self._ep_csv_file.close()


# ======================= PPO Trainer =======================
class PPOTrainer:
    def __init__(self, log_every="episode"):
        self.env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        self.eval_env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        self.folder = "PPO/static_run_advanced/"
        self.tensorboard_log_dir = os.path.join(self.folder, "tensors")
        self.best_model_dir = os.path.join(self.folder, "best_model")
        self.checkpoint_dir = os.path.join(self.folder, "checkpoints")
        self.eval_log_dir = os.path.join(self.folder, "eval_logs")

        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)

        self.checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoint_dir,
            name_prefix="checkpoint"
        )
        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.eval_log_dir,
            eval_freq=3000,
            deterministic=True,
            render=False
        )
        self.train_logger = TrainLogger(folder=self.folder, tag="train", log_every=log_every)
        self.callback = CallbackList([self.checkpoint_callback, self.eval_callback, self.train_logger])

    def train(self, steps=100_000):
        model = PPO(
            "MlpPolicy",
            env=self.env,
            verbose=1,
            tensorboard_log=self.tensorboard_log_dir,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.95,
            device="cpu"
        )

        model.learn(
            total_timesteps=steps,
            callback=self.callback,
            log_interval=1
        )

        model.save(os.path.join(self.folder, "final_model"))
        self.env.close()
        self.eval_env.close()
        print("✅ Training complete. Final model and logs saved.")


# ======================= PPO Continue Training =======================
class PPOCont:
    def __init__(self, model_path, steps=100_000, out_folder="cont_runs", log_every="episode"):
        self.model_path = model_path
        self.steps = steps
        self.out_folder = out_folder

        self.tensorboard_log_dir = os.path.join(self.out_folder, "tensors")
        self.best_model_dir = os.path.join(self.out_folder, "best_model")
        self.checkpoint_dir = os.path.join(self.out_folder, "checkpoints")
        self.eval_log_dir = os.path.join(self.out_folder, "eval_logs")

        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)

        self.log_every = log_every

    def run(self):
        env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        eval_env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")

        model = PPO.load(self.model_path, env=env, device="cpu")

        checkpoint_cb = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoint_dir,
            name_prefix="checkpoint"
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.eval_log_dir,
            eval_freq=3000,
            deterministic=True,
            render=False
        )
        cont_logger = TrainLogger(folder=self.out_folder, tag="cont", log_every=self.log_every)
        cb = CallbackList([checkpoint_cb, eval_cb, cont_logger])

        model.learn(
            total_timesteps=self.steps,
            callback=cb,
            log_interval=1,
            reset_num_timesteps=False   # continue from previous step count
        )

        final_path = os.path.join(self.out_folder, "final_model")
        model.save(final_path)
        env.close()
        eval_env.close()
        print(f"✅ Continued training complete. Saved to {final_path}")


# ======================= PPO Evaluation (configurable cadence) =======================
class PPOEval:
    """
    Evaluate a saved PPO model with configurable logging frequency.

    log_every:
      - "episode"  -> log once per episode (default)
      - "step"     -> log every step
      - int N>0    -> log every N steps
    """
    def __init__(self, model_path, episodes=10, out_dir="", log_every="episode", tag="eval",
                 vehicle_name: str = VEHICLE_NAME_DEFAULT, lidar_name: str = LIDAR_NAME_DEFAULT):
        self.model_path = model_path
        self.episodes = episodes
        self.folder = out_dir
        self.tag = tag
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name

        # Resolve stride from log_every
        if isinstance(log_every, str):
            v = log_every.lower()
            if v == "episode":
                self.step_stride = None
            elif v == "step":
                self.step_stride = 1
            else:
                raise ValueError("log_every must be 'episode', 'step', or an integer > 0")
        elif isinstance(log_every, int) and log_every > 0:
            self.step_stride = int(log_every)
        else:
            raise ValueError("log_every must be 'episode', 'step', or an integer > 0")

        # Folders
        self.tensorboard_log_dir = os.path.join(self.folder, "tensors")
        self.eval_log_dir = os.path.join(self.folder, "eval_logs")
        self.step_log_dir = os.path.join(self.folder, "step_logs")
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)
        if self.step_stride is not None:
            os.makedirs(self.step_log_dir, exist_ok=True)

    def run(self):
        env = DroneEnv(vehicle_name=self.vehicle_name, lidar_name=self.lidar_name, reset_mode="sim")
        model = PPO.load(self.model_path, env=env, device="cuda")
        writer = SummaryWriter(self.tensorboard_log_dir)

        # Per-episode CSV
        csv_path = os.path.join(self.eval_log_dir, "eval_summary.csv")
        with open(csv_path, "w", newline="") as fsum:
            wsum = csv.writer(fsum)
            wsum.writerow(["episode", "steps", "return", "result"])

            successes = 0
            global_step = 0  # TB step index across all episodes

            for ep in range(self.episodes):
                obs, _ = env.reset()
                done = False
                ep_rew = 0.0
                steps = 0
                term_reason = None

                # Per-step CSV 
                if self.step_stride is not None:
                    step_csv = os.path.join(self.step_log_dir, f"steps_ep{ep}.csv")
                    fstep = open(step_csv, "w", newline="")
                    wstep = csv.writer(fstep)
                    wstep.writerow([
                        "t", "reward", "dist_to_goal", "min_front",
                        "yaw_rate", "vx", "vy", "vz_up", "alt_err", "terminated_reason"
                    ])
                else:
                    fstep = None
                    wstep = None

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    ep_rew += float(reward)

                    if self.step_stride is not None and (steps % self.step_stride == 0):
                        dist_to_goal = env._get_goal_distance()
                        min_front = float(info.get("min_front", np.nan))
                        cmd = info.get("cmd", {})
                        yaw_rate   = float(cmd.get("yaw_rate", np.nan))
                        vx         = float(cmd.get("vx", np.nan))
                        vy         = float(cmd.get("vy", np.nan))
                        vz_up      = float(cmd.get("vz_up", np.nan))
                        alt_err    = float(info.get("alt_err", np.nan))
                        term_reason = info.get("terminated_reason")

                        # CSV
                        wstep.writerow([
                            steps, float(reward), dist_to_goal, min_front,
                            yaw_rate, vx, vy, vz_up, alt_err, term_reason or ""
                        ])

                        # TB
                        writer.add_scalar(f"{self.tag}_step/reward",        float(reward), global_step)
                        writer.add_scalar(f"{self.tag}_step/dist_to_goal",  dist_to_goal,  global_step)
                        writer.add_scalar(f"{self.tag}_step/min_front",     min_front,     global_step)
                        writer.add_scalar(f"{self.tag}_step/yaw_rate",      yaw_rate,      global_step)
                        writer.add_scalar(f"{self.tag}_step/vx",            vx,            global_step)
                        writer.add_scalar(f"{self.tag}_step/vy",            vy,            global_step)
                        writer.add_scalar(f"{self.tag}_step/vz_up",         vz_up,         global_step)
                        writer.add_scalar(f"{self.tag}_step/alt_err",       alt_err,       global_step)
                        global_step += 1

                    if terminated or truncated:
                        term_reason = info.get("terminated_reason") or ("truncated" if truncated else "done")
                        done = True

                if fstep is not None:
                    fstep.close()

                # Episode-level logging
                success = 1 if term_reason == "success" else 0
                successes += success
                wsum.writerow([ep, steps, ep_rew, term_reason])

                writer.add_scalar(f"{self.tag}/episode_return", ep_rew, ep)
                writer.add_scalar(f"{self.tag}/episode_length", steps, ep)
                writer.add_scalar(f"{self.tag}/success", success, ep)

                print(f"[eval] episode={ep} steps={steps} reward={ep_rew:.2f} result={term_reason}")

            # Final success rate
            sr = successes / max(1, self.episodes)
            writer.add_scalar(f"{self.tag}/success_rate", sr, self.episodes)
            writer.flush()
            writer.close()

        print(f"✅ Eval done: episodes={self.episodes} success_rate={sr:.2%}")
        print(f"    TensorBoard: {self.tensorboard_log_dir}")
        print(f"    Episode CSV: {csv_path}")
        if self.step_stride is not None:
            print(f"    Step CSVs:   {self.step_log_dir}")
        env.close()


# ======================= Multi-drone concurrent evaluation =======================
class MultiEvalPPO:
    
    # Concurrent multi-drone evaluation of ONE trained PPO model.

    # Modes:
    #   - Per-drone reset (default): only the colliding drone resets itself.
    #   - Global reset on any collision: first collision forces a full sim reset;
    #     all drones stop their current episode and restart (optionally staggered).

    def __init__(self, model_path: str, vehicle_names: List[str],
                 lidar_name: str = LIDAR_NAME_DEFAULT,
                 episodes: int = 5,
                 out_dir: str = "multi_eval_runs",
                 log_every: Union[str, int] = "episode",
                 tag: str = "multi_eval",
                 stagger_sec: float = 0.0,
                 inter_episode_delay_sec: float = 0.0,
                 global_reset_on_any_collision: bool = False,
                 stagger_on_global_reset: bool = True):
        self.model_path = model_path
        self.vehicle_names = list(vehicle_names)
        self.lidar_name = lidar_name
        self.episodes = int(episodes)
        self.out_dir = out_dir
        self.tag = tag

        # Resolve stride from log_every
        if isinstance(log_every, str):
            v = log_every.lower()
            if v == "episode":
                self.step_stride = None
            elif v == "step":
                self.step_stride = 1
            else:
                raise ValueError("log_every must be 'episode', 'step', or an integer > 0")
        elif isinstance(log_every, int) and log_every > 0:
            self.step_stride = int(log_every)
        else:
            raise ValueError("log_every must be 'episode', 'step', or an integer > 0")

        # Timing
        self.stagger_sec = float(stagger_sec)
        self.inter_episode_delay_sec = float(inter_episode_delay_sec)

        # Reset policy
        self.global_reset_on_any_collision = bool(global_reset_on_any_collision)
        self.stagger_on_global_reset = bool(stagger_on_global_reset)

        os.makedirs(self.out_dir, exist_ok=True)

        # --- Shared sync objects for global reset mode ---
        self._want_global_reset = threading.Event()
        self._reset_barrier = threading.Barrier(len(self.vehicle_names))
        self._print_lock = threading.Lock()  # console prints

    def run(self):
        threads = []
        for idx, vname in enumerate(self.vehicle_names):
            start_delay = idx * self.stagger_sec
            t = threading.Thread(target=self._worker, args=(idx, vname, start_delay), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        with self._print_lock:
            print("✅ Multi-eval finished for:", ", ".join(self.vehicle_names))

    def _worker(self, idx: int, vehicle_name: str, start_delay_sec: float):
        # Optional stagger before constructing env/model
        if start_delay_sec > 0.0:
            with self._print_lock:
                print(f"[{vehicle_name}] delayed start: +{start_delay_sec:.2f}s")
            time.sleep(start_delay_sec)

        # Per-vehicle folders
        v_root = os.path.join(self.out_dir, vehicle_name)
        tb_dir     = os.path.join(v_root, "tensors")
        eval_dir   = os.path.join(v_root, "eval_logs")
        step_dir   = os.path.join(v_root, "step_logs")
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        if self.step_stride is not None:
            os.makedirs(step_dir, exist_ok=True)

        # Dedicated env & model; do NOT global-reset on per-episode resets
        env = DroneEnv(vehicle_name=vehicle_name, lidar_name=self.lidar_name, reset_mode="vehicle")
        model = PPO.load(self.model_path, env=env, device="cpu")

        writer = SummaryWriter(tb_dir)
        summary_csv = os.path.join(eval_dir, "eval_summary.csv")
        new_file = not os.path.exists(summary_csv)
        fsum = open(summary_csv, "a", newline="")
        wsum = csv.writer(fsum)
        if new_file:
            wsum.writerow(["episode", "steps", "return", "result"])

        successes = 0
        global_step = 0  # TB step index for this vehicle

        ep = 0
        while ep < self.episodes:
            # Optional per-vehicle delay between episodes
            if ep > 0 and self.inter_episode_delay_sec > 0.0:
                time.sleep(self.inter_episode_delay_sec)

            obs, _ = env.reset()
            done = False
            ep_rew = 0.0
            steps = 0
            term_reason = None

            # Per-step CSV (optional)
            if self.step_stride is not None:
                step_csv = os.path.join(step_dir, f"steps_ep{ep}.csv")
                fstep = open(step_csv, "w", newline="")
                wstep = csv.writer(fstep)
                wstep.writerow([
                    "t", "reward", "dist_to_goal", "min_front",
                    "yaw_rate", "vx", "vy", "vz_up", "alt_err", "terminated_reason"
                ])
            else:
                fstep = None; wstep = None

            while not done:
                # If another drone requested a global reset, cut this episode short
                if self.global_reset_on_any_collision and self._want_global_reset.is_set():
                    term_reason = term_reason or "global_reset_cut"
                    done = True
                    break

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                ep_rew += float(reward)

                # Step logs
                if self.step_stride is not None and (steps % self.step_stride == 0):
                    dist_to_goal = env._get_goal_distance()
                    min_front = float(info.get("min_front", np.nan))
                    cmd = info.get("cmd", {})
                    yaw_rate = float(cmd.get("yaw_rate", np.nan))
                    vx = float(cmd.get("vx", np.nan))
                    vy = float(cmd.get("vy", np.nan))
                    vz_up = float(cmd.get("vz_up", np.nan))
                    alt_err = float(info.get("alt_err", np.nan))
                    t_reason = info.get("terminated_reason")

                    wstep.writerow([
                        steps, float(reward), dist_to_goal, min_front,
                        yaw_rate, vx, vy, vz_up, alt_err, t_reason or ""
                    ])
                    writer.add_scalar(f"{self.tag}_step/reward",       float(reward), global_step)
                    writer.add_scalar(f"{self.tag}_step/dist_to_goal", dist_to_goal,  global_step)
                    writer.add_scalar(f"{self.tag}_step/min_front",    min_front,     global_step)
                    writer.add_scalar(f"{self.tag}_step/yaw_rate",     yaw_rate,      global_step)
                    writer.add_scalar(f"{self.tag}_step/vx",           vx,            global_step)
                    writer.add_scalar(f"{self.tag}_step/vy",           vy,            global_step)
                    writer.add_scalar(f"{self.tag}_step/vz_up",        vz_up,         global_step)
                    writer.add_scalar(f"{self.tag}_step/alt_err",      alt_err,       global_step)
                    global_step += 1

                if terminated or truncated:
                    term_reason = info.get("terminated_reason") or ("truncated" if truncated else "done")
                    done = True

            if fstep is not None:
                fstep.close()

            # Episode-level TB/CSV
            success = 1 if term_reason == "success" else 0
            successes += success
            wsum.writerow([ep, steps, ep_rew, term_reason])
            writer.add_scalar(f"{self.tag}/episode_return", ep_rew, ep)
            writer.add_scalar(f"{self.tag}/episode_length", steps,  ep)
            writer.add_scalar(f"{self.tag}/success",        success, ep)
            with self._print_lock:
                print(f"[{vehicle_name}] ep={ep} steps={steps} return={ep_rew:.2f} result={term_reason}")

            # --- Global reset coordination  ---
            if self.global_reset_on_any_collision:
                # If THIS drone collided, request a global reset
                if term_reason == "collision":
                    self._want_global_reset.set()

                
                self._reset_barrier.wait()

                # Leader performs the sim reset if needed
                if idx == 0 and self._want_global_reset.is_set():
                    with self._print_lock:
                        print("[GLOBAL] Resetting AirSim (collision detected).")
                    leader_client = airsim.MultirotorClient()
                    leader_client.confirmConnection()
                    leader_client.reset()
                    time.sleep(0.25)
                    self._want_global_reset.clear()

                # Wait for reset to complete
                self._reset_barrier.wait()

                # Optionally re-stagger before next episodes begin
                if self.stagger_on_global_reset and self.stagger_sec > 0.0:
                    delay = idx * self.stagger_sec
                    time.sleep(delay)

            # Advance to next episode
            ep += 1

        # Final success rate
        sr = successes / max(1, self.episodes)
        writer.add_scalar(f"{self.tag}/success_rate", sr, self.episodes)
        writer.flush(); writer.close()
        fsum.flush(); fsum.close()
        env.close()
        with self._print_lock:
            print(f"✅ {vehicle_name}: success_rate={sr:.2%}  logs={v_root}")



# ======================= Multi-drone concurrent training =======================
class MultiTrainPPO:
    """
    Train ONE PPO model PER VEHICLE concurrently (thread-per-drone),
    using DroneEnv(reset_mode="vehicle") so episodes don't global-reset the sim.

    Each drone writes to: out_dir/<VehicleName>/{tensors, train_logs, step_logs, best_model, checkpoints}

    Args:
      model_path_init: optional path to a pre-trained model to warm-start each drone's training
      vehicle_names:   list of AirSim vehicle names ["Drone1", "Drone2", ...]
      lidar_name:      LiDAR sensor name present on each vehicle
      total_timesteps: training steps PER DRONE
      out_dir:         parent output directory
      log_every:       "episode", "step", or int N (every N steps) → passed to TrainLogger
      stagger_sec:     uniform start offset (Drone[i] waits i*stagger_sec before starting)
      save_freq:       checkpoint save frequency (env steps)
      eval_freq:       EvalCallback frequency (env steps)
      seed:            optional random seed per drone (internally shifted by index)
      ppo_kwargs:      dict with PPO hyperparams to override defaults (n_steps, batch_size, etc.)
    """
    def __init__(self,
                 vehicle_names,
                 lidar_name="Lidar1",
                 total_timesteps=100_000,
                 out_dir="multi_train_runs",
                 log_every="episode",
                 stagger_sec=0.0,
                 save_freq=250,
                 eval_freq=250,
                 model_path_init=None,
                 seed=None,
                 ppo_kwargs=None):
        self.vehicle_names = list(vehicle_names)
        self.lidar_name = lidar_name
        self.total_timesteps = int(total_timesteps)
        self.out_dir = out_dir
        self.log_every = log_every
        self.stagger_sec = float(stagger_sec)
        self.save_freq = int(save_freq)
        self.eval_freq = int(eval_freq)
        self.model_path_init = model_path_init
        self.seed = seed
        self.ppo_kwargs = ppo_kwargs or {}

        os.makedirs(self.out_dir, exist_ok=True)
        self._print_lock = threading.Lock()

    def run(self):
        threads = []
        for idx, vname in enumerate(self.vehicle_names):
            delay = idx * self.stagger_sec
            t = threading.Thread(target=self._worker, args=(idx, vname, delay), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        with self._print_lock:
            print("✅ Multi-train finished for:", ", ".join(self.vehicle_names))

    def _worker(self, idx: int, vehicle_name: str, start_delay_sec: float):
        # Optional stagger before starting
        if start_delay_sec > 0:
            with self._print_lock:
                print(f"[{vehicle_name}] training delayed start: +{start_delay_sec:.2f}s")
            time.sleep(start_delay_sec)

        # Per-vehicle folders
        vroot = os.path.join(self.out_dir, vehicle_name)
        tb_dir        = os.path.join(vroot, "tensors")
        best_model_dir= os.path.join(vroot, "best_model")
        ckpt_dir      = os.path.join(vroot, "checkpoints")
        eval_log_dir  = os.path.join(vroot, "eval_logs")
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(eval_log_dir, exist_ok=True)

        # Build envs — crucial: reset_mode="vehicle" so this drone's episode resets won't
        # global reset the world (which would kick other drones).
        env = DroneEnv(vehicle_name=vehicle_name, lidar_name=self.lidar_name, reset_mode="vehicle")
        eval_env = DroneEnv(vehicle_name=vehicle_name, lidar_name=self.lidar_name, reset_mode="vehicle")

        # Callbacks (mirrors your single-drone trainer)
        checkpoint_cb = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=ckpt_dir,
            name_prefix="checkpoint"
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=eval_log_dir,
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False
        )
        train_logger = TrainLogger(folder=vroot, tag="train", log_every=self.log_every)
        callbacks = CallbackList([checkpoint_cb, eval_cb, train_logger])

        # PPO hyperparams (defaults match your trainer; can be overridden via ppo_kwargs)
        hp = dict(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=tb_dir,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.95,
            device="cpu",
            seed=(None if self.seed is None else int(self.seed) + idx),
        )
        hp.update(self.ppo_kwargs)

        # Create or load model
        if self.model_path_init:
            model = PPO.load(self.model_path_init, env=env, device=hp.get("device", "cpu"))
            # When loading, SB3 ignores many kwargs; set what we can afterwards
            model.set_logger(model.logger)  # keep TB path
        else:
            model = PPO(**hp)

        # Train
        with self._print_lock:
            print(f"[{vehicle_name}] starting training for {self.total_timesteps} steps")
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            log_interval=1,
            reset_num_timesteps=(self.model_path_init is None)
        )

        # Save final model
        final_path = os.path.join(vroot, "final_model")
        model.save(final_path)

        # Cleanup
        env.close()
        eval_env.close()
        with self._print_lock:
            print(f"✅ [{vehicle_name}] training done. Saved: {final_path}")


# ======================= Multi-drone concurrent CONTINUATION =======================
class MultiContPPO:
    """
    Continue training ONE saved PPO model PER VEHICLE concurrently (thread-per-drone).

    Each drone:
      - loads the SAME base model (model_path_base)
      - keeps its own env (reset_mode='vehicle') so episode resets don't affect others
      - trains for `steps` timesteps
      - logs TB + CSV in out_dir/<VehicleName>/
      - saves checkpoints, best model (EvalCallback), and final_model

    Args:
      model_path_base: path to a previously trained single-drone PPO (e.g., "test_runs/final_model")
      vehicle_names:   ["Drone1","Drone2",...]
      lidar_name:      LiDAR sensor name present on each vehicle
      steps:           continuation timesteps PER DRONE
      out_dir:         parent output directory
      log_every:       "episode", "step", or int N (every N steps) for TrainLogger
      stagger_sec:     uniform thread start offset (Drone[i] waits i*stagger_sec before starting)
      save_freq:       checkpoint save frequency (env steps)
      eval_freq:       evaluation frequency (env steps)
      seed:            optional base seed; will be shifted per drone (seed+idx)
      ppo_overrides:   dict to override PPO hyperparams after loading (e.g., learning_rate)
    """
    def __init__(self,
                 model_path_base: str,
                 vehicle_names: list,
                 lidar_name: str = "Lidar1",
                 steps: int = 100_000,
                 out_dir: str = "multi_cont_runs",
                 log_every="episode",
                 stagger_sec: float = 0.0,
                 save_freq: int = 25000,
                 eval_freq: int = 3000,
                 seed: int | None = None,
                 ppo_overrides: dict | None = None):
        self.model_path_base = model_path_base
        self.vehicle_names   = list(vehicle_names)
        self.lidar_name      = lidar_name
        self.steps           = int(steps)
        self.out_dir         = out_dir
        self.log_every       = log_every
        self.stagger_sec     = float(stagger_sec)
        self.save_freq       = int(save_freq)
        self.eval_freq       = int(eval_freq)
        self.seed            = seed
        self.ppo_overrides   = ppo_overrides or {}

        os.makedirs(self.out_dir, exist_ok=True)
        self._print_lock = threading.Lock()

    def run(self):
        threads = []
        for idx, vname in enumerate(self.vehicle_names):
            delay = idx * self.stagger_sec
            t = threading.Thread(target=self._worker, args=(idx, vname, delay), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        with self._print_lock:
            print("✅ Multi-continue finished for:", ", ".join(self.vehicle_names))

    def _worker(self, idx: int, vehicle_name: str, start_delay_sec: float):
        # Optional stagger before constructing env/model
        if start_delay_sec > 0.0:
            with self._print_lock:
                print(f"[{vehicle_name}] delayed start: +{start_delay_sec:.2f}s")
            time.sleep(start_delay_sec)

        # Per-vehicle folders
        v_root = os.path.join(self.out_dir, vehicle_name)
        tb_dir     = os.path.join(v_root, "tensors")
        eval_dir   = os.path.join(v_root, "eval_logs")
        step_dir   = os.path.join(v_root, "step_logs")
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        if self.step_stride is not None:
            os.makedirs(step_dir, exist_ok=True)

        # Dedicated env & model
        env = DroneEnv(vehicle_name=vehicle_name, lidar_name=self.lidar_name, reset_mode="vehicle")
        model = PPO.load(self.model_path, env=env, device="cpu")

        writer = SummaryWriter(tb_dir)
        summary_csv = os.path.join(eval_dir, "eval_summary.csv")
        new_file = not os.path.exists(summary_csv)
        fsum = open(summary_csv, "a", newline="")
        wsum = csv.writer(fsum)
        if new_file:
            wsum.writerow(["episode", "steps", "return", "result"])

        successes = 0
        global_step = 0

        ep = 0
        while ep < self.episodes:
            if ep > 0 and self.inter_episode_delay_sec > 0.0:
                time.sleep(self.inter_episode_delay_sec)

            obs, _ = env.reset()
            done = False
            ep_rew = 0.0
            steps = 0
            term_reason = None
            last_info = {}

            if self.step_stride is not None:
                step_csv = os.path.join(step_dir, f"steps_ep{ep}.csv")
                fstep = open(step_csv, "w", newline="")
                wstep = csv.writer(fstep)
                wstep.writerow([
                    "t", "reward", "dist_to_goal", "min_front",
                    "yaw_rate", "vx", "vy", "vz_up", "alt_err", "terminated_reason"
                ])
            else:
                fstep = None; wstep = None

            while not done:
                if self.global_reset_on_any_collision and self._want_global_reset.is_set():
                    term_reason = term_reason or "global_reset_cut"
                    done = True
                    break

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                ep_rew += float(reward)
                last_info = info  ### store last info for post-episode checks

                if self.step_stride is not None and (steps % self.step_stride == 0):
                    dist_to_goal = env._get_goal_distance()
                    min_front = float(info.get("min_front", np.nan))
                    cmd = info.get("cmd", {})
                    yaw_rate = float(cmd.get("yaw_rate", np.nan))
                    vx = float(cmd.get("vx", np.nan))
                    vy = float(cmd.get("vy", np.nan))
                    vz_up = float(cmd.get("vz_up", np.nan))
                    alt_err = float(info.get("alt_err", np.nan))
                    t_reason = info.get("terminated_reason")

                    wstep.writerow([
                        steps, float(reward), dist_to_goal, min_front,
                        yaw_rate, vx, vy, vz_up, alt_err, t_reason or ""
                    ])
                    writer.add_scalar(f"{self.tag}_step/reward",       float(reward), global_step)
                    writer.add_scalar(f"{self.tag}_step/dist_to_goal", dist_to_goal,  global_step)
                    writer.add_scalar(f"{self.tag}_step/min_front",    min_front,     global_step)
                    writer.add_scalar(f"{self.tag}_step/yaw_rate",     yaw_rate,      global_step)
                    writer.add_scalar(f"{self.tag}_step/vx",           vx,            global_step)
                    writer.add_scalar(f"{self.tag}_step/vy",           vy,            global_step)
                    writer.add_scalar(f"{self.tag}_step/vz_up",        vz_up,         global_step)
                    writer.add_scalar(f"{self.tag}_step/alt_err",      alt_err,       global_step)
                    global_step += 1

                if terminated or truncated:
                    term_reason = info.get("terminated_reason") or ("truncated" if truncated else "done")
                    done = True

            if fstep is not None:
                fstep.close()

            # ------------------------------
            # Episode-level TB/CSV
            # ------------------------------
            ### CHANGED: treat "end_wall" collision as success
            obj_name = last_info.get("obj_name", "").upper()
            if term_reason == "success" or obj_name == "END_WALL":
                success = 1
            else:
                success = 0

            successes += success
            wsum.writerow([ep, steps, ep_rew, term_reason])
            writer.add_scalar(f"{self.tag}/episode_return", ep_rew, ep)
            writer.add_scalar(f"{self.tag}/episode_length", steps,  ep)
            writer.add_scalar(f"{self.tag}/success",        success, ep)
            with self._print_lock:
                print(f"[{vehicle_name}] ep={ep} steps={steps} return={ep_rew:.2f} result={term_reason} success={success}")

            # --- Global reset coordination (unchanged) ---
            if self.global_reset_on_any_collision:
                if term_reason == "collision":
                    self._want_global_reset.set()
                self._reset_barrier.wait()
                if idx == 0 and self._want_global_reset.is_set():
                    with self._print_lock:
                        print("[GLOBAL] Resetting AirSim (collision detected).")
                    leader_client = airsim.MultirotorClient()
                    leader_client.confirmConnection()
                    leader_client.reset()
                    time.sleep(0.25)
                    self._want_global_reset.clear()
                self._reset_barrier.wait()
                if self.stagger_on_global_reset and self.stagger_sec > 0.0:
                    delay = idx * self.stagger_sec
                    time.sleep(delay)

            ep += 1

        # Final success rate
        sr = successes / max(1, self.episodes)
        writer.add_scalar(f"{self.tag}/success_rate", sr, self.episodes)
        writer.flush(); writer.close()
        fsum.flush(); fsum.close()
        env.close()
        with self._print_lock:
            print(f"✅ {vehicle_name}: success_rate={sr:.2%}  logs={v_root}")



# ======================= DDPG Trainer =======================
class DDPGTrainer:
    def __init__(self, log_every="episode",
                 action_noise_type="ou",   # 'ou' or 'normal' or None
                 noise_sigma=0.2, noise_theta=0.15, noise_dt=1.0):
        self.env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        self.eval_env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        self.folder = "DDPG_/static_run_advanced"
        self.tensorboard_log_dir = os.path.join(self.folder, "tensors")
        self.best_model_dir = os.path.join(self.folder, "best_model")
        self.checkpoint_dir = os.path.join(self.folder, "checkpoints")
        self.eval_log_dir = os.path.join(self.folder, "eval_logs")
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)

        self.checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoint_dir,
            name_prefix="checkpoint"
        )
        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.eval_log_dir,
            eval_freq=3000,
            deterministic=True,
            render=False
        )
        self.train_logger = TrainLogger(folder=self.folder, tag="train", log_every=log_every)
        self.callback = CallbackList([self.checkpoint_callback, self.eval_callback, self.train_logger])

        # action noise config
        self.action_noise_type = action_noise_type
        self.noise_sigma = float(noise_sigma)
        self.noise_theta = float(noise_theta)
        self.noise_dt = float(noise_dt)

    def _make_action_noise(self, env):
        if self.action_noise_type is None:
            return None
        n_actions = env.action_space.shape[0]
        sigma = self.noise_sigma * np.ones(n_actions, dtype=np.float32)
        if self.action_noise_type.lower() == "normal":
            return NormalActionNoise(mean=np.zeros(n_actions, dtype=np.float32), sigma=sigma)
        elif self.action_noise_type.lower() == "ou":
            return OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions, dtype=np.float32),
                sigma=sigma, theta=self.noise_theta, dt=self.noise_dt
            )
        else:
            return None

    def train(self, steps=100_000):
        noise = self._make_action_noise(self.env)

        model = DDPG(
            "MlpPolicy",
            env=self.env,
            verbose=1,
            tensorboard_log=self.tensorboard_log_dir,
            # Typical DDPG defaults/tweaks
            learning_rate=1e-3,
            buffer_size=1_000_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=(100, "step"),
            gradient_steps=100,
            learning_starts=10_000,
            action_noise=noise,
            device="auto",
        )

        model.learn(
            total_timesteps=steps,
            callback=self.callback,
            log_interval=1
        )

        model.save(os.path.join(self.folder, "final_model"))
        self.env.close()
        self.eval_env.close()
        print("✅ DDPG training complete. Final model and logs saved.")


# ======================= DDPG Continue Training =======================
class DDPGCont:
    def __init__(self, model_path, steps=100_000, out_folder="ddpg_cont_runs",
                 log_every="episode",
                 action_noise_type=None, noise_sigma=0.2, noise_theta=0.15, noise_dt=1.0):
        self.model_path = model_path
        self.steps = steps
        self.out_folder = out_folder

        self.tensorboard_log_dir = os.path.join(self.out_folder, "tensors")
        self.best_model_dir = os.path.join(self.out_folder, "best_model")
        self.checkpoint_dir = os.path.join(self.out_folder, "checkpoints")
        self.eval_log_dir = os.path.join(self.out_folder, "eval_logs")
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)

        self.log_every = log_every
        self.action_noise_type = action_noise_type
        self.noise_sigma = float(noise_sigma)
        self.noise_theta = float(noise_theta)
        self.noise_dt = float(noise_dt)

    def _make_action_noise(self, env):
        if self.action_noise_type is None:
            return None
        n_actions = env.action_space.shape[0]
        sigma = self.noise_sigma * np.ones(n_actions, dtype=np.float32)
        if self.action_noise_type.lower() == "normal":
            return NormalActionNoise(mean=np.zeros(n_actions, dtype=np.float32), sigma=sigma)
        elif self.action_noise_type.lower() == "ou":
            return OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions, dtype=np.float32),
                sigma=sigma, theta=self.noise_theta, dt=self.noise_dt
            )
        else:
            return None

    def run(self):
        env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        eval_env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")

        model = DDPG.load(self.model_path, env=env, device="auto")

        # Optional: plug action noise for continued exploration
        model.action_noise = self._make_action_noise(env)

        checkpoint_cb = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoint_dir,
            name_prefix="checkpoint"
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.eval_log_dir,
            eval_freq=3000,
            deterministic=True,
            render=False
        )
        cont_logger = TrainLogger(folder=self.out_folder, tag="cont", log_every=self.log_every)
        cb = CallbackList([checkpoint_cb, eval_cb, cont_logger])

        model.learn(
            total_timesteps=self.steps,
            callback=cb,
            log_interval=1,
            reset_num_timesteps=False
        )

        final_path = os.path.join(self.out_folder, "final_model")
        model.save(final_path)
        env.close()
        eval_env.close()
        print(f"✅ DDPG continued training complete. Saved to {final_path}")


# ======================= DDPG Evaluation =======================
class DDPGEval:
    """
    Evaluate a saved DDPG model with configurable logging frequency.

    log_every:
      - "episode"  -> log once per episode
      - "step"     -> log every step
      - int N>0    -> log every N steps
    """
    def __init__(self, model_path, episodes=10, out_dir="", log_every="episode", tag="eval",
                 vehicle_name: str = VEHICLE_NAME_DEFAULT, lidar_name: str = LIDAR_NAME_DEFAULT):
        self.model_path = model_path
        self.episodes = episodes
        self.folder = out_dir
        self.tag = tag
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name

        # resolve stride
        if isinstance(log_every, str):
            v = log_every.lower()
            if v == "episode":
                self.step_stride = None
            elif v == "step":
                self.step_stride = 1
            else:
                raise ValueError("log_every must be 'episode', 'step', or int>0")
        elif isinstance(log_every, int) and log_every > 0:
            self.step_stride = int(log_every)
        else:
            raise ValueError("log_every must be 'episode', 'step', or int>0")

        self.tensorboard_log_dir = os.path.join(self.folder, "tensors")
        self.eval_log_dir = os.path.join(self.folder, "eval_logs")
        self.step_log_dir = os.path.join(self.folder, "step_logs")
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)
        if self.step_stride is not None:
            os.makedirs(self.step_log_dir, exist_ok=True)

    def run(self):
        env = DroneEnv(vehicle_name=self.vehicle_name, lidar_name=self.lidar_name, reset_mode="sim")
        model = DDPG.load(self.model_path, env=env, device="auto")
        writer = SummaryWriter(self.tensorboard_log_dir)

        csv_path = os.path.join(self.eval_log_dir, "eval_summary.csv")
        with open(csv_path, "w", newline="") as fsum:
            wsum = csv.writer(fsum)
            wsum.writerow(["episode", "steps", "return", "result"])

            successes = 0
            global_step = 0

            for ep in range(self.episodes):
                obs, _ = env.reset()
                done = False
                ep_rew = 0.0
                steps = 0
                term_reason = None

                if self.step_stride is not None:
                    step_csv = os.path.join(self.step_log_dir, f"steps_ep{ep}.csv")
                    fstep = open(step_csv, "w", newline="")
                    wstep = csv.writer(fstep)
                    wstep.writerow([
                        "t", "reward", "dist_to_goal", "min_front",
                        "yaw_rate", "vx", "vy", "vz_up", "alt_err", "terminated_reason"
                    ])
                else:
                    fstep = None; wstep = None

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    ep_rew += float(reward)

                    if self.step_stride is not None and (steps % self.step_stride == 0):
                        dist_to_goal = env._get_goal_distance()
                        min_front = float(info.get("min_front", np.nan))
                        cmd = info.get("cmd", {})
                        yaw_rate = float(cmd.get("yaw_rate", np.nan))
                        vx = float(cmd.get("vx", np.nan))
                        vy = float(cmd.get("vy", np.nan))
                        vz_up = float(cmd.get("vz_up", np.nan))
                        alt_err = float(info.get("alt_err", np.nan))
                        term_reason = info.get("terminated_reason")

                        wstep.writerow([
                            steps, float(reward), dist_to_goal, min_front,
                            yaw_rate, vx, vy, vz_up, alt_err, term_reason or ""
                        ])
                        writer.add_scalar(f"{self.tag}_step/reward",       float(reward), global_step)
                        writer.add_scalar(f"{self.tag}_step/dist_to_goal", dist_to_goal,  global_step)
                        writer.add_scalar(f"{self.tag}_step/min_front",    min_front,     global_step)
                        writer.add_scalar(f"{self.tag}_step/yaw_rate",     yaw_rate,      global_step)
                        writer.add_scalar(f"{self.tag}_step/vx",           vx,            global_step)
                        writer.add_scalar(f"{self.tag}_step/vy",           vy,            global_step)
                        writer.add_scalar(f"{self.tag}_step/vz_up",        vz_up,         global_step)
                        writer.add_scalar(f"{self.tag}_step/alt_err",      alt_err,       global_step)
                        global_step += 1

                    if terminated or truncated:
                        term_reason = info.get("terminated_reason") or ("truncated" if truncated else "done")
                        done = True

                if fstep is not None:
                    fstep.close()

                success = 1 if term_reason == "success" else 0
                successes += success
                wsum.writerow([ep, steps, ep_rew, term_reason])
                writer.add_scalar(f"{self.tag}/episode_return", ep_rew, ep)
                writer.add_scalar(f"{self.tag}/episode_length", steps, ep)
                writer.add_scalar(f"{self.tag}/success", success, ep)
                print(f"[ddpg eval] episode={ep} steps={steps} reward={ep_rew:.2f} result={term_reason}")

            sr = successes / max(1, self.episodes)
            writer.add_scalar(f"{self.tag}/success_rate", sr, self.episodes)
            writer.flush(); writer.close()

        print(f"✅ DDPG eval done: episodes={self.episodes} success_rate={sr:.2%}")
        print(f"    TensorBoard: {self.tensorboard_log_dir}")
        print(f"    Episode CSV: {csv_path}")
        if self.step_stride is not None:
            print(f"    Step CSVs:   {self.step_log_dir}")
        env.close()





from stable_baselines3 import DQN

# ======================= DQN Trainer =======================
class DQNTrainer:
    def __init__(self, log_every="episode"):
        self.env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        self.eval_env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        self.folder = "dqn/static_runs/200k"
        self.tensorboard_log_dir = os.path.join(self.folder, "tensors")
        self.best_model_dir = os.path.join(self.folder, "best_model")
        self.checkpoint_dir = os.path.join(self.folder, "checkpoints")
        self.eval_log_dir = os.path.join(self.folder, "eval_logs")
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)

        self.checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoint_dir,
            name_prefix="checkpoint"
        )
        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.eval_log_dir,
            eval_freq=3000,
            deterministic=True,
            render=False
        )
        self.train_logger = TrainLogger(folder=self.folder, tag="train", log_every=log_every)
        self.callback = CallbackList([self.checkpoint_callback, self.eval_callback, self.train_logger])

    def train(self, steps=100_000):
        model = DQN(
            "MlpPolicy",
            env=self.env,
            verbose=1,
            tensorboard_log=self.tensorboard_log_dir,
            learning_rate=1e-4,
            buffer_size=100_000,
            batch_size=512,
            learning_starts=10_000,
            gamma=0.99,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            device="auto"
        )

        model.learn(total_timesteps=steps, callback=self.callback, log_interval=1)
        model.save(os.path.join(self.folder, "final_model"))
        self.env.close(); self.eval_env.close()
        print("✅ DQN training complete. Final model and logs saved.")


# ======================= DQN Continue Training =======================
class DQNCont:
    def __init__(self, model_path, steps=100_000, out_folder="dqn_cont_runs", log_every="episode"):
        self.model_path = model_path
        self.steps = steps
        self.out_folder = out_folder
        self.tensorboard_log_dir = os.path.join(self.out_folder, "tensors")
        self.best_model_dir = os.path.join(self.out_folder, "best_model")
        self.checkpoint_dir = os.path.join(self.out_folder, "checkpoints")
        self.eval_log_dir = os.path.join(self.out_folder, "eval_logs")
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)
        self.log_every = log_every

    def run(self):
        env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        eval_env = DroneEnv(vehicle_name=VEHICLE_NAME_DEFAULT, lidar_name=LIDAR_NAME_DEFAULT, reset_mode="sim")
        model = DQN.load(self.model_path, env=env, device="auto")

        checkpoint_cb = CheckpointCallback(save_freq=10000, save_path=self.checkpoint_dir, name_prefix="checkpoint")
        eval_cb = EvalCallback(eval_env, best_model_save_path=self.best_model_dir,
                               log_path=self.eval_log_dir, eval_freq=3000, deterministic=True, render=False)
        cont_logger = TrainLogger(folder=self.out_folder, tag="cont", log_every=self.log_every)
        cb = CallbackList([checkpoint_cb, eval_cb, cont_logger])

        model.learn(total_timesteps=self.steps, callback=cb, log_interval=1, reset_num_timesteps=False)
        final_path = os.path.join(self.out_folder, "final_model")
        model.save(final_path)
        env.close(); eval_env.close()
        print(f"✅ DQN continued training complete. Saved to {final_path}")


# ======================= DQN Evaluation =======================
class DQNEval:
    def __init__(self, model_path, episodes=10, out_dir="", log_every="episode", tag="eval",
                 vehicle_name: str = VEHICLE_NAME_DEFAULT, lidar_name: str = LIDAR_NAME_DEFAULT):
        self.model_path = model_path
        self.episodes = episodes
        self.folder = out_dir
        self.tag = tag
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name

        if isinstance(log_every, str):
            v = log_every.lower()
            self.step_stride = None if v == "episode" else (1 if v == "step" else None)
        elif isinstance(log_every, int) and log_every > 0:
            self.step_stride = int(log_every)
        else:
            raise ValueError("log_every must be 'episode', 'step', or int>0")

        self.tensorboard_log_dir = os.path.join(self.folder, "tensors")
        self.eval_log_dir = os.path.join(self.folder, "eval_logs")
        self.step_log_dir = os.path.join(self.folder, "step_logs")
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)
        if self.step_stride is not None:
            os.makedirs(self.step_log_dir, exist_ok=True)

    def run(self):
        env = DroneEnv(vehicle_name=self.vehicle_name, lidar_name=self.lidar_name, reset_mode="sim")
        model = DQN.load(self.model_path, env=env, device="auto")
        writer = SummaryWriter(self.tensorboard_log_dir)

        csv_path = os.path.join(self.eval_log_dir, "eval_summary.csv")
        with open(csv_path, "w", newline="") as fsum:
            wsum = csv.writer(fsum)
            wsum.writerow(["episode", "steps", "return", "result"])

            successes = 0
            global_step = 0
            for ep in range(self.episodes):
                obs, _ = env.reset()
                done, ep_rew, steps, term_reason = False, 0.0, 0, None
                if self.step_stride is not None:
                    step_csv = os.path.join(self.step_log_dir, f"steps_ep{ep}.csv")
                    fstep = open(step_csv, "w", newline=""); wstep = csv.writer(fstep)
                    wstep.writerow(["t", "reward", "dist_to_goal", "min_front",
                                    "yaw_rate", "vx", "vy", "vz_up", "alt_err", "terminated_reason"])
                else:
                    fstep = None; wstep = None

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1; ep_rew += float(reward)

                    if self.step_stride and steps % self.step_stride == 0:
                        dist_to_goal = env._get_goal_distance()
                        min_front = float(info.get("min_front", np.nan))
                        cmd = info.get("cmd", {})
                        yaw_rate, vx, vy, vz_up = [float(cmd.get(k, np.nan)) for k in ["yaw_rate", "vx", "vy", "vz_up"]]
                        alt_err = float(info.get("alt_err", np.nan))
                        term_reason = info.get("terminated_reason")

                        wstep.writerow([steps, float(reward), dist_to_goal, min_front,
                                        yaw_rate, vx, vy, vz_up, alt_err, term_reason or ""])
                        writer.add_scalar(f"{self.tag}_step/reward", float(reward), global_step)
                        writer.add_scalar(f"{self.tag}_step/dist_to_goal", dist_to_goal, global_step)
                        global_step += 1

                    if terminated or truncated:
                        term_reason = info.get("terminated_reason") or ("truncated" if truncated else "done")
                        done = True

                if fstep: fstep.close()
                success = 1 if term_reason == "success" else 0
                successes += success
                wsum.writerow([ep, steps, ep_rew, term_reason])
                writer.add_scalar(f"{self.tag}/episode_return", ep_rew, ep)
                writer.add_scalar(f"{self.tag}/episode_length", steps, ep)
                writer.add_scalar(f"{self.tag}/success", success, ep)
                print(f"[dqn eval] episode={ep} steps={steps} reward={ep_rew:.2f} result={term_reason}")

            sr = successes / max(1, self.episodes)
            writer.add_scalar(f"{self.tag}/success_rate", sr, self.episodes)
            writer.flush(); writer.close()

        print(f"✅ DQN eval done: episodes={self.episodes} success_rate={sr:.2%}")
        env.close()





# ======================= Entrypoint examples =======================
if __name__ == "__main__":
    # --- Choose one of the flows below and uncomment ---

    # # 1) Train with per-episode logs:
    # trainer = PPOTrainer(log_every="episode")
    # trainer.train(steps=150_000)

    # 2) Continue training for 200k steps with per-step logs:
    #PPOCont(model_path="400K/final_model", steps=200_000, out_folder="static_runs/600K", log_every="episode").run()

    # 3) Single-drone evaluation, log every 5 steps:
    PPOEval(model_path="C:\\THESIS\\PPO\\Aggressive\\150k\\best_model\\best_model", episodes=1, out_dir="C:\\THESIS\\PPO\\Aggressive\\150k\\evaluations\\1_episode", log_every="episode",
            vehicle_name="Drone1", lidar_name="Lidar1").run()

    # 4) Multi-drone concurrent eval (Drone1 & Drone2), per-episode logs:
    # MultiEvalPPO(
    #     model_path="static_runs/600K/final_model",
    #     vehicle_names=["Drone1", "Drone2"],   
    #     lidar_name="Lidar1",
    #     episodes=1,
    #     out_dir="static_runs/600K/evaluations/two/final",
    #     log_every="episode",
    #     stagger_sec=10,
    #     global_reset_on_any_collision=True   # <-- per-drone reset mode
    # ).run()


    # MultiTrainPPO(
    #     vehicle_names=["Drone1", "Drone2", "Drone3"],
    #     lidar_name="Lidar1",
    #     total_timesteps=200_000,   # steps PER DRONE
    #     out_dir="multi_train_runs",
    #     log_every="episode",       # or "step", or an int N
    #     stagger_sec=3.5,           # Drone1 @0s, Drone2 @3.5s, Drone3 @7.0s
    #     save_freq=25000,
    #     eval_freq=3000,
    #     # model_path_init="test_runs/final_model",  # uncomment to warm-start each from a base model
    #     seed=123,                  # optional; will shift per drone (123,124,125)
    #     ppo_kwargs=dict(           # optional overrides
    #         n_steps=2048,
    #         batch_size=512,
    #         learning_rate=3e-4,
    #         gamma=0.995,
    #     )
    # ).run()

    # MultiContPPO(
    #     model_path_base="400K/final_model",       # or "test_runs/best_model/best_model"
    #     vehicle_names=["Drone1", "Drone2", "Drone3"],
    #     lidar_name="Lidar1",
    #     steps=50_000,                  # per drone
    #     out_dir="450K/three",
    #     log_every="episode",           # or "step" or N
    #     stagger_sec=3.5,
    #     save_freq=25000,
    #     eval_freq=3000,
    #     seed=123,                      # optional; becomes 123,124,125 per drone
    #     ppo_overrides=dict(learning_rate=2.5e-4)  # optional
    # ).run()




    # --- DDPG usage examples ---

    # 1) Train DDPG:
    # ddpg_tr = DDPGTrainer(log_every="episode", action_noise_type="ou", noise_sigma=0.2, noise_theta=0.15)
    # ddpg_tr.train(steps=200_000)

    # 2) Continue DDPG training:
    # DDPGCont(model_path="C:\\THESIS\\ddpg\\static_runs\\600k\\continues\\continues\\checkpoints\\checkpoint_400000_steps", steps=200_000,
    #          out_folder="C:\\THESIS\\ddpg\\static_runs\\600k\\cont_600k\\400k", log_every="episode",
    #          action_noise_type="normal", noise_sigma=0.1).run()

    # # 3) Evaluate DDPG:
    # DDPGEval(model_path="DDPG_\\static_run_advanced\\best_model\\best_model",
    #          episodes=1, out_dir="DDPG_\\static_run_advanced\\evaluations\\best\\drone1", log_every="episode",
    #          vehicle_name="Drone1", lidar_name="Lidar1").run()







    # --- DQN usage examples ---


    # # Train a brand new DQN agent
    # trainer = DQNTrainer(log_every="episode")   # or "step" if you want step-level logs
    # trainer.train(steps=200_000)                # adjust steps as needed


    # # Continue training from a previous run
    # cont = DQNCont(model_path="dqn/static_runs/250k/cont_model",
    #             steps=100_000,
    #             out_folder="dqn/static_runs/350k")
    # cont.run()

    # # Evaluate for 10 episodes
    # evaluator = DQNEval(model_path="C:\\THESIS\\dqn\\static_runs\\350k\\continues\\30k\\cont\\final_model",
    #                     episodes=10,
    #                     out_dir="C:\\THESIS\\dqn\\static_runs\\350k\\continues\\30k\\cont\\evaluations\\final",
    #                     log_every="episode")   # or "step" for detailed logs
    # evaluator.run()


    pass

