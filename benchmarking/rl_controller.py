# rl_controller.py

import numpy as np
from stable_baselines3 import PPO  # Example using PPO from Stable Baselines 3


class RLController:
    def __init__(self, model_path):
        """
        Initializes the RL controller with a trained model.

        Parameters:
            model_path (str): Path to the trained RL model.
        """
        self.model = PPO.load(model_path)
        print(f"RL model loaded from {model_path}")

    def move_to(self, target_position, env):
        """
        Moves the pipette to the target position using the RL model.

        Parameters:
            target_position (np.ndarray): Desired [x, y, z] position.
            env (OT2Env): The simulation environment.

        Returns:
            final_position (np.ndarray): Final [x, y, z] position after movement.
        """
        obs, _ = env.reset()
        env.goal_position = np.array(target_position, dtype=np.float32)
        done = False
        step = 0
        max_steps = 2000  # Define a step limit to prevent infinite loops

        final_position = None

        while not done and step < max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            final_position = obs[:3]
            distance = np.linalg.norm(env.goal_position - final_position)

            if distance < 0.001:  # 1 mm threshold
                done = True
            if terminated or truncated:
                done = True
            step += 1

        return final_position

    def reset(self):
        """Resets the RL controller if necessary."""
        pass  # Implement if your RL model requires resetting internal states
