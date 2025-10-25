# core_env.py
# Unified environments (continuous + discrete), utilities, and training logger
# Used by PPO/DDPG (continuous control) and DQN (discrete control)

import os
import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback


# ================== Global Tunables ==================
GOAL_RADIUS_M    = 3.0
ALT_GOAL_TOL_M   = 9.0

CHECK_HZ = 15
CMD_DT   = 1.0 / CHECK_HZ
YAW_RATE_CMD_MAX_DEGPS = 145.0
VX_MIN, VX_MAX = 0.6, 3.5
VY_MAX = 1.5
VZ_MAX_UP = 1.0

BINS = 360
OBS_BINS = 60
RANGE_CLIP_MAX = 20.0
RANGE_CLIP_MIN = 0.2
USE_HEIGHT_BAND  = (-0.7, 1.8)
IGNORE_COLLISION_TOKENS = {"takeoff_zone"}

GOAL_OBJECT_NAME = "landing_zone"

GOAL_NORM_M      = 25.0
ALT_TARGET_M     = 2.0
ALT_NORM_M       = 3.0

R_SUCCESS     = 100.0
R_COLLISION   = 100.0
W_PROGRESS    = 12.0
W_CLEAR       = 0.6
W_CENTER      = 0.1
W_ALT         = 0.5
W_SMOOTH      = 0.02

FOV_FRONTAL_DEG = 85
CLEAR_SAFE_M    = 2.8

AUTO_LAND_ON_GOAL = True
GOAL_APPROACH_RADIUS_M = 2.5
LAND_HOVER_SEC = 0.3

COLLISION_GRACE_SEC = 0.8


# =============== Utilities ===============
def clamp(v, lo, hi): return max(lo, min(hi, v))

def wrap_deg(d): return (d + 180.0) % 360.0 - 180.0

def quat_to_yaw(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def world_to_body_xy(yaw, vx_w, vy_w):
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
    if pts.shape[0] == 0:
        return np.full(BINS, RANGE_CLIP_MAX, dtype=np.float32)
    angles = np.degrees(np.arctan2(pts[:,1], pts[:,0]))
    ranges = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    ranges = np.clip(ranges, RANGE_CLIP_MIN, RANGE_CLIP_MAX)
    bins = np.floor((angles + 360.0) % 360.0).astype(int)
    ring = np.full(BINS, RANGE_CLIP_MAX, dtype=np.float32)
    np.minimum.at(ring, bins, ranges)
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
    """Split forward FOV into left/right halves and return mean clearance."""
    half = len(FWD_IDX) // 2
    left_idx = FWD_IDX[:half]
    right_idx = FWD_IDX[half:]
    return float(np.mean(ring[left_idx])), float(np.mean(ring[right_idx]))


# ======================= Continuous Env (PPO/DDPG) =======================
class DroneEnvContinuous(gym.Env):
    """
    Env for continuous-control algorithms (PPO, DDPG).
    Action space: Box(4,) → [yaw_rate, vx, vy, vz_up]
    Observation: lidar ring + velocity + alt_err + goal vec + dist
    """
    metadata = {"render_modes": []}

    def __init__(self, vehicle_name="Drone1", lidar_name="Lidar1", reset_mode="sim"):
        super().__init__()
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name
        self.reset_mode = reset_mode

        self.step_count = 0
        self.episode_idx = 0
        self.prev_cmd_phys = np.zeros(4, dtype=np.float32)

        # Continuous actions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Obs: lidar ring + extras
        low  = np.concatenate([np.zeros(OBS_BINS, dtype=np.float32),
                               np.full(3, -1.0, dtype=np.float32),    # vx, vy, vz
                               np.full(1, -1.0, dtype=np.float32),    # alt_err
                               np.full(3, -1.0, dtype=np.float32),    # goal vec
                               np.array([0.0], dtype=np.float32)])    # dist
        high = np.concatenate([np.ones(OBS_BINS, dtype=np.float32),
                               np.full(3,  1.0, dtype=np.float32),
                               np.full(1,  1.0, dtype=np.float32),
                               np.full(3,  1.0, dtype=np.float32),
                               np.array([1.0], dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # simple progress print for last episode
        if getattr(self, "step_count", 0) > 0:
            print(f"[{self.vehicle_name}] episode {self.episode_idx} steps={self.step_count}")
        self.episode_idx += 1
        self.step_count = 0

        # Reset world or just this vehicle
        if self.reset_mode == "sim":
            self.client.reset()

        # Ensure control for this vehicle only
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True,  self.vehicle_name)

        # Goal pose
        gpose = self.client.simGetObjectPose(GOAL_OBJECT_NAME)
        if not (np.isfinite(gpose.position.x_val) and np.isfinite(gpose.position.y_val) and np.isfinite(gpose.position.z_val)):
            # fallback: virtual goal 30m ahead in world-X
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
        z_target = self.start_z_ned - ALT_TARGET_M   # NED: up means more negative
        self.client.moveToZAsync(z=z_target, velocity=1.0, vehicle_name=self.vehicle_name).join()
        time.sleep(0.2)

        # Small random initial yaw to diversify
        yaw0 = float(np.random.uniform(-15.0, 15.0))
        self.client.rotateToYawAsync(yaw0, 5, vehicle_name=self.vehicle_name).join()
        time.sleep(0.1)

        # Collision grace window (ignore spawn jitter touches)
        self._grace_until = time.time() + COLLISION_GRACE_SEC

        # Bookkeeping
        self.prev_cmd_phys = np.zeros(4, dtype=np.float32)
        obs = self._get_obs()
        self.prev_goal_dist = self._get_goal_distance()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        a = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)

        # ----- Map action to physical commands -----
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

        # --- Collision / success / auto-land ---
        col_info  = self.client.simGetCollisionInfo(self.vehicle_name)
        raw_collided = bool(col_info.has_collided)
        obj_name = (getattr(col_info, "object_name", "") or "")
        name_l = obj_name.lower()

        # True success if we touch the actual landing object
        landed_on_goal = raw_collided and (obj_name == GOAL_OBJECT_NAME or obj_name == "END_WALL")

        # Ignore benign hits (e.g., takeoff pad), unless it's the goal
        ignored_hit = raw_collided and (not landed_on_goal) and any(tok in name_l for tok in IGNORE_COLLISION_TOKENS)

        # Ignore collisions during the grace window after reset
        in_grace = (time.time() < self._grace_until)

        # Effective collision used for termination
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
            # optional: tiny shaping penalty to discourage scraping the pad
            # r_done -= 0.05
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
        """Read LiDAR for THIS vehicle/sensor only."""
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



# ======================= Discrete Env (DQN) =======================
class DroneEnvDiscrete(gym.Env):
    """
    Env for discrete-action algorithms (DQN).
    Action space: Discrete(7) → forward, left, right, up, down, yawL, yawR
    Observation: lidar ring + extras
    """
    metadata = {"render_modes": []}

    def __init__(self, vehicle_name="Drone1", lidar_name="Lidar1", debug=False):
        super().__init__()
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name
        self.debug = debug

        self.prev_cmd_phys = np.zeros(4, dtype=np.float32)

        # Discrete actions
        self.action_space = spaces.Discrete(7)

        # Obs: lidar ring + extras
        low  = np.concatenate([np.zeros(OBS_BINS, dtype=np.float32),
                               np.full(8, -1.0, dtype=np.float32)])
        high = np.concatenate([np.ones(OBS_BINS, dtype=np.float32),
                               np.full(8,  1.0, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def reset(self, **kwargs):
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        gpose = self.client.simGetObjectPose(GOAL_OBJECT_NAME)
        self.goal_world = np.array([gpose.position.x_val,
                                    gpose.position.y_val,
                                    gpose.position.z_val], dtype=np.float32)

        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.2)
        vpose = self.client.simGetVehiclePose(self.vehicle_name)
        self.start_z_ned = vpose.position.z_val
        z_target = self.start_z_ned - ALT_TARGET_M
        self.client.moveToZAsync(z=z_target, velocity=1.0,
                                 vehicle_name=self.vehicle_name).join()

        self._grace_until = time.time() + COLLISION_GRACE_SEC
        obs = self._get_obs()
        self.prev_goal_dist = self._get_goal_distance()
        return obs, {}





    def step(self, action):
        yaw_rate, vx, vy, vz_up = self._map_action(action)
        vz_ned = -vz_up
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz_ned, CMD_DT,
                                                 yaw_mode=yaw_mode,
                                                 vehicle_name=self.vehicle_name)
        time.sleep(CMD_DT)

        obs = self._get_obs()
        ring = self._get_ring()
        min_front = forward_min_clearance(ring)

        cur_dist = self._get_goal_distance()
        progress = (self.prev_goal_dist - cur_dist)
        r_progress = W_PROGRESS * progress
        r_clear = W_CLEAR * clamp(min_front / CLEAR_SAFE_M, 0.0, 1.0)
        alt_err = self._get_alt_error()
        r_alt = - W_ALT * clamp(abs(alt_err) / ALT_NORM_M, 0.0, 1.0)
        cmd_phys = np.array([yaw_rate, vx, vy, vz_up], dtype=np.float32)
        r_smooth = - W_SMOOTH * float(np.linalg.norm(cmd_phys - self.prev_cmd_phys))
        self.prev_cmd_phys = cmd_phys

        reward, terminated = 0.0, False
        col_info = self.client.simGetCollisionInfo(self.vehicle_name)
        collided = bool(col_info.has_collided)
        if collided and time.time() > self._grace_until:
            reward -= R_COLLISION
            terminated = True
        if cur_dist <= GOAL_RADIUS_M and abs(alt_err) <= ALT_GOAL_TOL_M:
            reward += R_SUCCESS
            terminated = True

        reward += r_progress + r_clear + r_alt + r_smooth
        self.prev_goal_dist = cur_dist
        info = {
            "min_front": min_front,
            "progress": progress,
            "alt_err": alt_err,
            "cmd": {"yaw_rate": yaw_rate, "vx": vx, "vy": vy, "vz_up": vz_up},
            
        }
        return obs, reward, terminated, False, info

    # ---- Helpers ----
    def _map_action(self, a):
        if a == 0:   return 0.0, VX_MAX, 0.0, 0.0   # forward
        if a == 1:   return 0.0, VX_MIN,  VY_MAX, 0.0  # left
        if a == 2:   return 0.0, VX_MIN, -VY_MAX, 0.0  # right
        if a == 3:   return 0.0, VX_MIN, 0.0,  VZ_MAX_UP # up
        if a == 4:   return 0.0, VX_MIN, 0.0, -VZ_MAX_UP # down
        if a == 5:   return  YAW_RATE_CMD_MAX_DEGPS, VX_MIN, 0.0, 0.0
        if a == 6:   return -YAW_RATE_CMD_MAX_DEGPS, VX_MIN, 0.0, 0.0
        return 0.0, VX_MIN, 0.0, 0.0

    def _lidar_points(self):
        data = self.client.getLidarData(lidar_name=self.lidar_name,
                                        vehicle_name=self.vehicle_name)
        pts = np.array(data.point_cloud, dtype=np.float32)
        return pts.reshape(-1,3) if pts.size > 0 else np.empty((0,3), dtype=np.float32)

    def _get_ring(self): return scan_to_ring(self._lidar_points())

    def _get_obs(self):
        ring = self._get_ring()
        ring_ds = downsample_ring(ring)
        ring_norm = np.clip(ring_ds / RANGE_CLIP_MAX, 0.0, 1.0)
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        yaw = quat_to_yaw(state.kinematics_estimated.orientation)
        vel = state.kinematics_estimated.linear_velocity
        pos = state.kinematics_estimated.position
        vx_b, vy_b = world_to_body_xy(yaw, vel.x_val, vel.y_val)
        vz_up = -vel.z_val
        vx_b_n = clamp(vx_b / VX_MAX, -1.0, 1.0)
        vy_b_n = clamp(vy_b / VY_MAX, -1.0, 1.0)
        vz_up_n= clamp(vz_up / VZ_MAX_UP, -1.0, 1.0)
        alt_err_n = clamp(self._get_alt_error() / ALT_NORM_M, -1.0, 1.0)
        dx_w, dy_w, dz_up = self._goal_delta_world_up()
        gx_b, gy_b = rotate_world_to_body_xy(yaw, dx_w, dy_w)
        gx_n = clamp(gx_b / GOAL_NORM_M, -1.0, 1.0)
        gy_n = clamp(gy_b / GOAL_NORM_M, -1.0, 1.0)
        gz_n = clamp(dz_up / GOAL_NORM_M, -1.0, 1.0)
        dist = math.sqrt(dx_w*dx_w + dy_w*dy_w + dz_up*dz_up)
        dist_n = clamp(dist / GOAL_NORM_M, 0.0, 1.0)
        extras = np.array([vx_b_n, vy_b_n, vz_up_n, alt_err_n,
                           gx_n, gy_n, gz_n, dist_n], dtype=np.float32)
        return np.concatenate([ring_norm, extras]).astype(np.float32)

    def _get_goal_distance(self):
        dx, dy, dz = self._goal_delta_world_up()
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _goal_delta_world_up(self):
        vpose = self.client.simGetVehiclePose(self.vehicle_name)
        cx, cy, cz = vpose.position.x_val, vpose.position.y_val, vpose.position.z_val
        gx, gy, gz = self.goal_world
        dx_w = gx - cx
        dy_w = gy - cy
        dz_up = -(gz - cz)
        return dx_w, dy_w, dz_up

    def _get_alt_error(self):
        vpose = self.client.simGetVehiclePose(self.vehicle_name)
        z_ned = vpose.position.z_val
        alt_above_start = (self.start_z_ned - z_ned)
        return (ALT_TARGET_M - alt_above_start)

# ======================= Factory =======================
def make_env(algo: str, vehicle_name="Drone1", lidar_name="Lidar1", **kwargs):
    """
    Factory to select environment depending on algorithm.
    """
    algo = algo.lower()
    if algo in ["ppo", "ddpg"]:
        return DroneEnvContinuous(vehicle_name=vehicle_name, lidar_name=lidar_name, **kwargs)
    elif algo == "dqn":
        return DroneEnvDiscrete(vehicle_name=vehicle_name, lidar_name=lidar_name, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm '{algo}'")


# =============== Training Logger ===============
class TrainLogger(BaseCallback):
    def __init__(self, folder, tag="train"):
        super().__init__()
        self.writer = SummaryWriter(os.path.join(folder, "tensors"))
        self.ep_return, self.ep_steps, self.ep_index = 0.0, 0, 0

    def _on_step(self) -> bool:
        r = float(self.locals.get("rewards", [0])[0])
        done = bool(self.locals.get("dones", [False])[0])
        self.ep_return += r
        self.ep_steps += 1
        if done:
            self.writer.add_scalar("episode/return", self.ep_return, self.ep_index)
            self.writer.add_scalar("episode/length", self.ep_steps, self.ep_index)
            self.ep_return, self.ep_steps = 0.0, 0
            self.ep_index += 1
        return True
