"""

This module defines the OT2Env class, a custom Gymnasium environment for simulating
a robotic pipette using the PyBullet physics engine. The environment facilitates
interaction with the simulation by translating agent actions into simulation commands
and providing observations about the pipette's state and its goal position.
"""
import math
import os

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_class import Simulation


class OT2Env(gym.Env):

    def __init__(self, seed=None, render=False, max_steps=1000, distance_threshold=0.001):
        """
        Initializes the OT2Env environment.

        Parameters:
            render (bool): If True, enables rendering of the simulation GUI.
                           If False, runs the simulation in DIRECT mode without GUI.
            max_steps (int): The maximum number of steps allowed per episode.
        """
        super(OT2Env, self).__init__()
        self.enable_render = render
        self.max_steps = max_steps
        self.seed = seed

        # Initialize the Simulation with one agent and rendering as specified
        self.sim = Simulation(num_agents=1, render=self.enable_render)

        # Define the action space: 3 continuous actions (x, y, z velocities)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

        # Define the observation space: 6 continuous observations (pipette x, y, z and goal x, y, z)
        self.observation_space = spaces.Box(
            low=-math.inf,
            high=math.inf,
            shape=(6,),
            dtype=np.float32
        )

        # Initialize step counter and robot identifier
        self.steps = 0
        self.robot_Id = None
        self.previous_distance = None
        self.distance_threshold = distance_threshold

    def get_plate_image(self):
        """
        Captures and returns the current image of the plate from the simulation.

        Returns:
            np.ndarray: Image of the plate in RGB format.
        """
        if not self.enable_render:
            raise ValueError("Rendering is disabled. Enable rendering to capture images.")

            # Get the plate image path from the Simulation class
        image_path = self.sim.get_plate_image()

        # Load the image using matplotlib

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Plate image not found at path: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        return image

    @staticmethod
    def get_working_envelope_coords():
        """
        Retrieves the predefined working envelope coordinates.

        Returns:
            tuple: (x_min, x_max, y_min, y_max, z_min, z_max) defining the workspace boundaries.
        """
        working_envelope = {
            'corner_1': [-0.187, -0.171, 0.17],
            'corner_2': [-0.187, -0.171, 0.289],
            'corner_3': [-0.187, 0.22, 0.17],
            'corner_4': [-0.187, 0.22, 0.289],
            'corner_5': [0.253, -0.171, 0.17],
            'corner_6': [0.253, -0.171, 0.289],
            'corner_7': [0.253, 0.22, 0.17],
            'corner_8': [0.253, 0.22, 0.289]

        }

        # Extract min and max values for each axis from the working envelope
        x_min = min(pos[0] for pos in working_envelope.values())
        x_max = max(pos[0] for pos in working_envelope.values())
        y_min = min(pos[1] for pos in working_envelope.values())
        y_max = max(pos[1] for pos in working_envelope.values())
        z_min = min(pos[2] for pos in working_envelope.values())
        z_max = max(pos[2] for pos in working_envelope.values())

        return x_min, x_max, y_min, y_max, z_min, z_max

    def reset(self, seed=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        Parameters:
            seed (int, optional): Seed for the environment's random number generator to ensure reproducibility.

        Returns:
            tuple: (initial_observation, info)
        """
        if self.seed:
            np.random.seed(self.seed)

        # Get workspace boundaries
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_working_envelope_coords()

        # Set a random goal position within the working envelope
        self.goal_position = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ], dtype=np.float32)

        # Reset the simulation and obtain the initial observation
        observation = self.sim.reset(num_agents=1)

        # Extract the robot's unique identifier
        self.robot_Id = list(observation.keys())[0]

        # Retrieve the current pipette position
        pipette_pos = observation[self.robot_Id]['pipette_position']

        # Validate the pipette position length
        assert len(pipette_pos) == 3, f"Invalid pipette position: {pipette_pos}"

        # Combine pipette position with goal position to form the observation
        observation = np.array([
            *pipette_pos,
            *self.goal_position
        ], dtype=np.float32)

        # Reset step counter
        self.steps = 0
        self.previous_distance = np.linalg.norm(pipette_pos - self.goal_position)
        self.previous_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return observation, {}

    def step(self, action):
        """
        Executes one time step within the environment.

        Parameters:
            action (np.ndarray): An action provided by the agent, consisting of [x_velocity, y_velocity, z_velocity].

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Append drop command (0 indicates no drop action)
        action = [*action, 0]

        # Run the simulation step with the given action
        observation = self.sim.run([action])

        # Ensure the robot identifier is defined
        assert self.robot_Id is not None, 'Robot Id is not defined'

        # Retrieve the current pipette position
        pipette_pos = observation[self.robot_Id]['pipette_position']

        # Validate the pipette position length
        assert len(pipette_pos) == 3, f"Invalid pipette position: {pipette_pos}"

        # Combine pipette position with goal position to form the observation
        observation = np.array([
            *pipette_pos,
            *self.goal_position
        ], dtype=np.float32)

        # Calculate distance to goal
        distance = np.linalg.norm(pipette_pos - self.goal_position)

        # Base reward: negative distance
        reward = -distance

        # Determine if the episode should be truncated based on step count
        terminated = distance < self.distance_threshold

        truncated = self.steps > self.max_steps

        # Increment step counter
        self.steps += 1

        info = {
            'distance_to_goal': distance
        }

        # Return the observation, reward, termination flags, and additional info
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Renders the environment. Since rendering is handled by PyBullet's GUI,
        this method can be extended for additional rendering functionalities if needed.

        Parameters:
            mode (str): The mode to render with. Currently, only 'human' is supported.

        Returns:
            None
        """
        pass  # Rendering is managed by the Simulation class

    def close(self):
        """
        Closes the environment and performs necessary cleanup.

        Returns:
            None
        """
        self.sim.close()
