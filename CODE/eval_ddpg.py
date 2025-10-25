# eval_ddpg.py
import os
import time
import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import DDPG
from Common.core_env import make_env   # ✅ your factory, handles DroneEnvContinuous

class DDPGEval:
    def __init__(self, model_path, episodes=5, device="cuda"):
        # Make environment
        self.env = make_env("ddpg")  
        
        # Load model with the same env and device
        self.model = DDPG.load(model_path, env=self.env, device=device)
        
        self.episodes = episodes

    def run(self):
        for ep in range(self.episodes):
            obs, info = self.env.reset()
            done, truncated = False, False
            ep_reward, step_count = 0.0, 0

            print(f"\n[Eval] Starting episode {ep+1}/{self.episodes}")
            while not (done or truncated):
                # Select action from policy
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)

                ep_reward += reward
                step_count += 1

                if "terminated_reason" in info and info["terminated_reason"]:
                    print(f" Terminated reason: {info['terminated_reason']}")

                # Optional: slow down to observe flight
                # time.sleep(0.05)

            print(f"[Eval] Episode {ep+1} finished | Steps={step_count}, Total Reward={ep_reward:.2f}")

        self.env.close()


if __name__ == "__main__":
    model_path = "Aggressive\\150k\\best_model\\best_model"  # update to your saved DDPG path
    evaluator = DDPGEval(model_path=model_path, episodes=3, device="cuda")
    evaluator.run()
