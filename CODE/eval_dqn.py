# eval_dqn.py
# Evaluation script for DQN split from your original run_dqn.py
import sys, os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from Common.core_env import TrainLogger,make_env

class DQNEval:
    def __init__(self, model_path, episodes=5,
                 out_dir="dqn/evaluations/default", log_every="episode",
                 vehicle_name="Drone1", lidar_name="Lidar1"):
        self.model_path = model_path
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name
        self.episodes = episodes
        self.out_dir = out_dir

        env = make_env("dqn", vehicle_name="Drone1", lidar_name="Lidar1")

        self.env = DummyVecEnv([lambda: env])
        self.model = DQN.load(self.model_path, env=self.env)
        self.log_cb = TrainLogger(folder=self.out_dir, tag=log_every)

    def lander(self, env, descent_speed=1.5):
        """
        Lands the drone by commanding a downward velocity until it touches the ground.
        descent_speed: positive number in m/s (controls how fast it descends).
        """
        print(f"[Eval] Goal reached — starting custom landing at {descent_speed:.2f} m/s")

        landed = False
        while not landed:
            # Command downward motion in body frame (Z down in NED)
            env.client.moveByVelocityBodyFrameAsync(
                vx=0.0, vy=0.0, vz=descent_speed, duration=2.5,
                vehicle_name=self.vehicle_name
            ).join()

            # Check if drone has touched the ground
            state = env.client.getMultirotorState(vehicle_name=self.vehicle_name)
            z_ned = state.kinematics_estimated.position.z_val
            landed_state = state.landed_state

            if landed_state == 1:  # 1 = Landed (AirSim API)
                landed = True
                print("[Eval] Drone landed (landed_state=1).")

        # After landing, disarm
        env.client.armDisarm(False, vehicle_name=self.vehicle_name)
        print("[Eval] Custom landing sequence complete.")


    def run(self):
        env = make_env("dqn", vehicle_name="Drone1", lidar_name="Lidar1")

        for ep in range(self.episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated:
                    # Call your custom lander
                    self.lander(env)
                done = terminated or truncated
            print(f"[Eval] Episode {ep+1}: return={total_reward:.2f}")
            self.log_cb.writer.add_scalar("eval/return", total_reward, ep)



if __name__ == "__main__":
    num_episodes = 3
    drone_name = "Drone1"
    lidar_name = "Lidar1"
    dqn_eval = DQNEval(model_path="Aggressive\\150k\\best_model\\best_model",
                       episodes=num_episodes, out_dir=f"Aggressive\\150k\\evaluations\\{num_episodes}_episodes",
                       vehicle_name=drone_name, lidar_name=lidar_name)
    dqn_eval.run()
