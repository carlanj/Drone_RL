# Machine Learning-Based Drone Navigation and Sensor Mapping in Simulated 3D Environments



1. Environment Setup
conda create -n airsim_rl python=3.10
conda activate airsim_rl
pip install -r requirements.txt

2. Required Packages

If requirements.txt isn’t provided, install these manually:

pip install gymnasium airsim torch stable-baselines3 tensorboard pandas

🧩 Training
DQN

Train a DQN agent:

python run_dqn.py

PPO

Train a PPO agent:

python run.py


All training logs, models, and checkpoints are automatically saved in:

/<algorithm>/<run_name>/

🧪 Evaluation
DQN Evaluation
python eval_dqn.py


Evaluates a trained DQN model for a set number of episodes and performs a controlled landing.

PPO Evaluation
python eval_ppo.py


Runs a PPO-trained agent deterministically and logs performance metrics to TensorBoard.

📊 Visualization
1. TensorBoard

To view performance in TensorBoard:

tensorboard --logdir ./PPO/static_run_advanced/tensors

2. Convert Excel Results

If you’ve logged training data to an Excel file:

python MAKE_TENSORS.py


This creates TensorBoard-compatible logs from .xlsx files for all algorithms.

🧱 Core Architecture
Environment

Implemented in core_env.py, the environment inherits from gym.Env and integrates tightly with AirSim:

Observation: Downsampled 360° LiDAR ring + drone velocity, altitude error, and goal vector

Actions:

DQN → 7 discrete movement actions

PPO/DDPG → continuous 4D action vector [yaw_rate, vx, vy, vz_up]

Rewards:

Progress toward goal

Clearance and centering

Smooth motion and stability

Collision and goal penalties

🪄 Key Components
File	Description
core_env.py	Defines unified environment and physics interface for PPO/DQN
run_dqn.py	Trains and logs a discrete DQN agent
eval_dqn.py	Evaluates trained DQN policies with landing behavior
eval_ppo.py	Evaluates PPO-trained agents and logs results
MAKE_TENSORS.py	Converts Excel logs into TensorBoard-compatible events
airsim_rl_multi.py	Adds support for concurrent multi-drone PPO evaluation
run.py	Main training launcher for PPO / DDPG agents
🧠 Example Reward Components
Term	Meaning	Weight
W_PROGRESS	Reward for moving closer to goal	12.0
W_CLEAR	Reward for maintaining safe LiDAR clearance	0.6
W_ALT	Penalty for altitude deviation	0.5
W_SMOOTH	Penalizes jerky control	0.02
R_SUCCESS	Bonus for reaching goal	+100
R_COLLISION	Penalty for collision	-100
📚 References

Microsoft AirSim

Stable Baselines3

OpenAI Gymnasium

🧑‍💻 Author

Mr. Jackson
Master’s Student in Computer Science (AI / Machine Learning)
GitHub: [your username]
Email: [your contact]

📜 License

This project is released under the MIT License.
You are free to use, modify, and distribute this software with attribution.
