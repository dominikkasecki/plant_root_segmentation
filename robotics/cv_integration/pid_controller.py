import logging

import matplotlib.pyplot as plt
import numpy as np

from ot2_gym_wrapper import OT2Env


class PIDController:
    def __init__(self, kp, ki, kd, dt=0.1, integral_limit=None):
        """
        Initializes the PID controller.
        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param dt: Time step for updates
        :param integral_limit: Tuple (min, max) to clamp the integral term (anti-windup)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = integral_limit

    def update(self, target, current):
        """
        Compute the control signal using the PID formula.
        :param target: Desired position
        :param current: Current position
        :return: Control signal
        """
        error = target - current
        self.integral += error * self.dt

        # Anti-windup: Clamp the integral term
        if self.integral_limit is not None:
            self.integral = max(self.integral_limit[0], min(self.integral, self.integral_limit[1]))

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative
        return control_signal

    def reset(self):
        """Resets the PID controller's state variables."""
        self.prev_error = 0.0
        self.integral = 0.0


def test_pid():
    """
    Tests the PID controller with the OT2 environment by controlling the X, Y, and Z axes.
    Displays the position tracking and outputs step-by-step differences between the current
    and target positions.
    """
    max_steps = 3000  # Maximum simulation steps
    env = OT2Env(seed=None, render=True, max_steps=max_steps)  # Initialize the simulation environment

    # Initialize PID controllers for each axis
    pid_x = PIDController(kp=4.97, ki=0.0098, kd=0.203, dt=0.01, integral_limit=(-10, 10))
    pid_y = PIDController(kp=1.58, ki=0.003, kd=0.23, dt=0.01, integral_limit=(-10, 10))
    pid_z = PIDController(kp=2.2, ki=0.001, kd=0.02, dt=0.01, integral_limit=(-10, 10))

    # Reset the environment and get the initial state
    observation, _ = env.reset()
    current_position = observation[:3]  # Current pipette position
    target_position = observation[3:]  # Target position
    logging.info(f"Target Position: {target_position}")

    positions = [current_position]  # Track positions for visualization

    # Run the simulation loop
    for step in range(max_steps):
        # Compute control signals for each axis
        control_x = pid_x.update(target_position[0], current_position[0])
        control_y = pid_y.update(target_position[1], current_position[1])
        control_z = pid_z.update(target_position[2], current_position[2])

        # Clip actions to match environment action space
        action = np.clip(
            np.array([control_x, control_y, control_z], dtype=np.float32),
            env.action_space.low,
            env.action_space.high,
        )

        # Apply the action and step the environment
        observation, reward, terminated, truncated, _ = env.step(action)
        current_position = observation[:3]  # Update current position

        positions.append(current_position)

        # Output step number and difference between current and target positions
        diff = np.abs(target_position - current_position)
        print(f"Step {step + 1}: Difference = {diff}")

        # Stop the simulation if terminated or truncated
        if terminated or truncated:
            logging.info(f"Goal reached at step {step + 1} with final position {current_position}")
            break

    env.close()  # Close the environment

    # Evaluate and display final errors
    errors = np.abs(target_position - current_position)
    thresholds = {"C": 0.01, "D": 0.001}
    print("\n--- Accuracy Check ---")
    for axis, error, threshold in zip("XYZ", errors, [thresholds["C"]] * 3):
        print(f"{axis}-axis Error: {error:.6f}")
        for name, value in thresholds.items():
            if error <= value:
                print(f"  -> Meets {name} requirement (<= {value} m).")
            else:
                print(f"  -> Does NOT meet {name} requirement.")

    # Plot the position tracking
    positions = np.array(positions)
    plt.figure(figsize=(10, 6))
    for i, axis in enumerate(["X-axis", "Y-axis", "Z-axis"]):
        plt.plot(positions[:, i], label=f"{axis} Position")
        plt.axhline(target_position[i], linestyle="--", label=f"Target {axis}")

    plt.xlabel("Steps")
    plt.ylabel("Position (m)")
    plt.title("PID Controller Position Tracking")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_pid()
