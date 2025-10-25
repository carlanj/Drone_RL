Machine Learning-Based Drone Navigation and Sensor Mapping in Simulated 3D Environments
Project Overview

This repository accompanies the Master’s thesis research by Carlan Jackson (Alabama A&M University, 2025) on developing an autonomous UAV navigation framework leveraging Reinforcement Learning (RL) and LiDAR-based perception in Microsoft AirSim.

The project implements and compares Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Deep Deterministic Policy Gradient (DDPG) algorithms under identical environmental and sensor conditions. The goal is to evaluate their performance in path planning, collision avoidance, and goal-seeking within simulated 3D environments.

Research Context

Autonomous UAV navigation remains a central challenge in robotics, particularly in unstructured or cluttered environments such as forests, urban areas, or disaster zones

FINAL_DRAFT

. Traditional rule-based and map-dependent planners like A* or Dijkstra struggle in these scenarios due to limited adaptability.

This project explores learning-driven navigation, where agents acquire decision-making policies through trial and error using LiDAR-based sensory input. By training within AirSim’s high-fidelity Unreal Engine environment, the framework achieves both safety in simulation and realism in physical modeling — creating a foundation for future real-world UAV deployment.

System Architecture

The project follows a modular three-tier structure, combining perception, learning, and data analytics.

1. Environment Interface

Implements a custom Gymnasium-compatible environment built on AirSim APIs.

reset() — Initializes UAV position, altitude, and LiDAR state.

step(action) — Executes control actions and returns next observation, reward, and termination flags.

Supports both discrete control (DQN) and continuous control (PPO/DDPG).

2. Training Module

Uses Stable-Baselines3 (SB3) for algorithm implementation, handling:

Episode management and checkpointing

Reward and evaluation tracking

TensorBoard logging and replay buffer management

3. Data Logging Module

Captures:

Episode rewards and success rates

Collision and clearance statistics

Time-to-goal metrics

TensorBoard-ready logs (via MAKE_TENSORS.py)

Observation and Action Spaces
Observation Space
Component	Description
LiDAR Scan Vector	360° ring downsampled into fixed bins, representing obstacle distance.
Ego-State	Body-frame velocities (vx, vy, vz), altitude error, and goal vector.
Normalization	All sensor inputs scaled to [0, 1] for generalization.
Action Space
Mode	Control Type	Description
DQN	Discrete	Movement primitives: forward, turn left/right, ascend/descend.
PPO/DDPG	Continuous	4D control vector [yaw_rate, vx, vy, vz] for fine-grained maneuvering.
Reward Shaping
Component	Description	Weight
Progress	Positive reward for reducing goal distance.	12.0
Clearance	Encourages maintaining safe LiDAR distance.	0.6
Centering	Penalizes deviation from mid-path.	0.1
Altitude Stability	Penalizes deviation from desired height.	0.5
Smoothness	Penalizes abrupt motion transitions.	0.02
Success / Collision	±100 terminal rewards for goal reach or impact.	—

This multi-term formulation guides the UAV toward efficient, stable, and collision-free flight

FINAL_DRAFT

.

Algorithms Implemented
Algorithm	Type	Action Space	Key Advantage
DQN	Off-policy	Discrete	Efficient sample use with replay buffers.
PPO	On-policy	Continuous	Stable policy updates with clipped objectives.
DDPG	Off-policy	Continuous	Fine-grained continuous control.

Each algorithm is trained and evaluated under identical simulation and reward conditions to enable fair comparison.

Installation and Setup
Requirements
conda create -n uav_rl python=3.10
conda activate uav_rl
pip install torch gymnasium airsim stable-baselines3 pandas tensorboard

Clone Repository
git clone https://github.com/<your-username>/UAV-RL-Project.git
cd UAV-RL-Project

AirSim Environment

Ensure AirSim is installed and running within Unreal Engine (see AirSim setup guide
).

Running the Code
Train DQN
python run_dqn.py

Train PPO or DDPG
python run.py

Evaluate Agents
python eval_dqn.py
python eval_ppo.py

Convert Excel Logs to TensorBoard
python MAKE_TENSORS.py

Results Summary

Experiments confirmed that:

PPO achieved stable convergence and smooth trajectories.

DQN offered faster learning but with higher oscillation.

DDPG produced continuous, precise motion but required careful hyperparameter tuning.

Under conservative configurations, agents achieved goal completion rates above 90% after convergence, demonstrating effective LiDAR-driven navigation

FINAL_DRAFT

.

Example Outputs

Figure 1: LiDAR point cloud visualization
Figure 2: TensorBoard training curves (episode reward vs timesteps)
Figure 3: Path trajectories for PPO vs DQN under identical environments

(Replace with real images or plots in /docs/images/.)

Future Work

Multi-agent UAV coordination

Real-world transfer using onboard LiDAR hardware

Curriculum-based training and noise randomization

Adaptive reward tuning for dynamic obstacles
