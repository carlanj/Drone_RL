# Machine Learning-Based Drone Navigation and Sensor Mapping in Simulated 3D Environments

## Overview

This repository contains the code, simulation pipeline, training logs, evaluation results, and media associated with:

The project trains autonomous quadrotor agents in a high-fidelity Unreal Engine / Microsoft AirSim environment to:
- avoid obstacles,
- navigate toward a goal,
- and land safely near a designated target.

Three deep reinforcement learning (DRL) algorithms are implemented and compared under consistent sensing, reward, and environment conditions:
- **Deep Q-Network (DQN)** – off-policy, discrete action space
- **Proximal Policy Optimization (PPO)** – on-policy, continuous action space
- **Deep Deterministic Policy Gradient (DDPG)** – off-policy, continuous action space

The agent relies primarily on LiDAR range data for perception, rather than relying on GPS or vision alone, to better handle cluttered and GPS-denied navigation scenarios

---

## Goals of the System

- Learn safe, goal-directed navigation in a cluttered 3D obstacle course without hand-coded waypoints or maps
- Study the tradeoffs between on-policy and off-policy learning for UAV control
- Compare discrete “motion primitive” control vs fully continuous velocity/yaw-rate control
- Generate policies that can be transferred between different environment configurations inside simulation

---

## System Architecture

The project is organized around three core modules:

### 1. Environment Interface
Custom Gymnasium-style environment (see `core_env.py`) that talks directly to AirSim.
- Handles drone spawn, arming, takeoff, and stabilization at target altitude
- Gathers LiDAR data + drone state each step
- Issues velocity / yaw commands through AirSim’s multirotor control API
- Computes reward, termination, and success/failure reasons

Supports:
- **Discrete control mode** (for DQN)
- **Continuous control mode** (for PPO and DDPG)

This environment exposes the standard `reset()` / `step(action)` loop that DRL algorithms expect

### 2. Training and Evaluation
Scripts in `CODE/` handle:
- running training loops for each algorithm,
- checkpointing intermediate models,
- saving the “best model,”
- and periodically running evaluation rollouts.

During evaluation, the agent is flown deterministically and will trigger an autonomous landing routine when it reaches the goal region. After touchdown, the drone disarms and logs the episode result

### 3. Logging and Analytics
Training and evaluation both log:
- Episode reward
- Episode length / steps survived
- Success or collision reason
- Goal distance over time
- Minimum forward clearance from LiDAR
- Altitude tracking error

These are written to:
- TensorBoard event logs
- Per-episode CSVs
- Optional per-step CSVs for deep debugging

There’s also a utility (`MAKE_TENSORS.py`) that converts spreadsheet logs from evaluations  into TensorBoard event files so different algorithms / runs can be compared visually

---

## Observation Space

Each observation given to the agent is a single fused vector made of:

1. **LiDAR perception**  
   - A simulated 360° LiDAR ring is sampled around the drone.
   - Distances are clipped, then downsampled into a fixed number of angular bins.
   - Values are normalized to `[0, 1]`.  
   This captures local free space and obstacle proximity without sending the full raw point cloud to the network

2. **Ego-state and goal features**  
   - Body-frame linear velocities (vx, vy, vz_up)  
   - Altitude error relative to the desired flight band  
   - Relative goal direction (goal vector in the drone’s body frame)  
   - Normalized distance-to-goal  

Together, this gives the policy awareness of:
- “What’s around me?”
- “How am I currently moving?”
- “Where should I go next?”

---

## Action Space

Two modes are supported:

### Discrete control (used by DQN)
The policy picks from a finite set of motion primitives, e.g.:
- move forward,
- yaw left / yaw right,
- strafe left / strafe right,
- ascend / descend.

This is well-suited for DQN because Q-learning over a fixed action set is stable and sample-efficient

### Continuous control (used by PPO / DDPG)
The policy outputs a 4D continuous vector:
`[yaw_rate, vx, vy, vz_up]`

- `yaw_rate`: commanded yaw rate (deg/s)  
- `vx, vy`: body-frame horizontal velocities  
- `vz_up`: vertical velocity in the drone’s up direction

Each component is clipped to physically reasonable limits (forward speed caps, max climb rate, etc.), producing smooth, physically consistent motion in AirSim

---

## Reward Design

The reward signal is shaped to balance safety, efficiency, and goal completion:

- **Progress toward goal**  
  Positive reward when the drone reduces its distance to the target.

- **Obstacle clearance**  
  Encourages keeping a safe forward gap using LiDAR distance in the frontal sector.

- **Corridor centering**  
  Rewards staying centered between obstacles instead of scraping walls.

- **Altitude stability**  
  Penalizes drifting too far from the allowed flight band.

- **Smoothness / control discipline**  
  Penalizes large jumps between successive control commands to reduce jitter.

- **Terminal events**  
  + Large positive reward for reaching and landing at/near the goal object  
  – Large negative penalty on collision

This structure teaches agents not just “don’t crash,” but “get to the goal efficiently, safely, and cleanly”

---

## Training Workflow

1. **Episode start**
   - Reset AirSim or just the vehicle (depending on mode).
   - Take off, stabilize at target altitude.
   - Sample / assign a navigation goal in the world (e.g. a landing zone marker).

2. **Control loop**
   - Read LiDAR + drone kinematics.
   - Policy chooses an action (discrete or continuous).
   - Action is applied via `moveByVelocityBodyFrameAsync(...)` for a short duration.
   - Reward is computed.
   - Episode continues until success, collision, or termination condition.

3. **Logging**
   - Per-step and per-episode metrics are written to CSV and TensorBoard.
   - “Best model” snapshots and rolling checkpoints are saved.

4. **Evaluation**
   - Run the trained agent in deterministic mode.
   - If it reaches the goal radius, trigger an automated landing and disarm procedure.
   - Record final return and outcome (success/collision/etc.)

---

## Algorithms

### Deep Q-Network (DQN)
- Off-policy, value-based
- Learns Q(s,a) over a discrete action set
- Uses replay buffer and a target network for stability
- Good for higher-level motion primitives

### Proximal Policy Optimization (PPO)
- On-policy, policy-gradient
- Uses a clipped objective to prevent unstable policy updates
- Works well with continuous actions and tends to produce smooth trajectories

### Deep Deterministic Policy Gradient (DDPG)
- Off-policy, actor–critic
- Learns a deterministic continuous control policy
- Allows fine-grained velocity/yaw control
- More sensitive to tuning and exploration noise

All three are trained and evaluated in the same LiDAR-based environment with the same reward model so the comparison is fair across policy families (on-policy vs off-policy) and action spaces (discrete vs continuous)

---

## Results (High-Level)

- **PPO** produced stable, smooth navigation and reliable goal completion once converged.
- **DQN** learned to reach goals and avoid collisions using only discrete motion primitives. It converged quickly in structured layouts.
- **DDPG** showed precise control in continuous space but demanded more care in tuning noise, stability, and update cadence.

Both conservative (slower, smoother control loops) and aggressive (faster, higher-speed, tighter response) settings were tested. Agents in both regimes were able to reach the goal region and perform controlled landings in simulation

---

## Repository Structure

```text
.
├── CODE/
│   ├── Common/                         # Shared env helpers, logging utilities, callbacks
│   ├── core_env.py                     # Core AirSim <-> Gym environment (discrete + continuous)
│   ├── run_dqn.py                      # Train DQN agent
│   ├── run.py                          # Train PPO / DDPG agent(s)
│   ├── eval_dqn.py                     # Evaluate DQN policy + auto-landing routine
│   ├── eval_ppo.py                     # Evaluate PPO policy + landing and logging
│   ├── eval_ddpg.py                    # Evaluate DDPG policy
│   └── MAKE_TENSORS.py                 # Convert spreadsheet logs → TensorBoard events
│
├── CONFIG/
│   └── SETTINGS/                       # JSON configs / scenario settings for runs
│
├── DATA/
│   ├── aggressive/                     # Logs and outputs for aggressive-control runs
│   └── normaldev/                      # Baseline / conservative-control experiment data
│
├── DOCUMENTATION/
│   ├── FINAL_DRAFT.docx                # Full thesis draft (Word)
│   ├── FINAL_DRAFT.pdf                 # Final thesis PDF
│   └── PROPOSAL.doc                    # Original research proposal
│
├── MODELS/
│   ├── DQN/
│   │   ├── best_model/                 # Best-performing checkpoint
│   │   ├── checkpoints/                # Periodic training checkpoints
│   │   ├── eval_logs/                  # Evaluation summaries / CSV
│   │   ├── tensors/                    # TensorBoard scalar logs
│   │   ├── train_logs/                 # Episode/step CSV logs
│   │   └── {10k,25k,50k,75k,100k,...}/ # Snapshots grouped by total timesteps trained
│   └── PPO/
│       ├── best_model/
│       ├── checkpoints/
│       ├── eval_logs/
│       ├── tensors/
│       
│
├── TENSORBOARDS/ # event.out.tfevents.* files live here
│  
│          

