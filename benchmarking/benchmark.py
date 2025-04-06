# benchmark_move.py
import logging
import time

import numpy as np

from ot2_gym_wrapper import OT2Env
from pid_controller import PIDController
from rl_controller import RLController

# Configure logging to display information during benchmarking
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16,  # Default font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 18,  # X and Y labels font size
    'legend.fontsize': 15,  # Legend font size
    'xtick.labelsize': 14,  # X-axis tick labels font size
    'ytick.labelsize': 14  # Y-axis tick labels font size
})


def plot_average_error(ax, controllers, average_errors):
    """
    Plots a bar chart comparing the average error of different controllers on a given Axes.
    """
    bars = ax.bar(controllers, average_errors, color=['skyblue', 'salmon'])
    ax.set_xlabel('Controller')
    ax.set_ylabel('Average Error (meters)')
    ax.set_title('Average Error Comparison')
    ax.set_ylim(0.0, max(average_errors) * 1.1)

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)


def plot_average_speed(ax, controllers, average_speeds):
    """
    Plots a bar chart comparing the average speed of different controllers on a given Axes.
    """
    bars = ax.bar(controllers, average_speeds, color=['skyblue', 'salmon'])
    ax.set_xlabel('Controller')
    ax.set_ylabel('Average Time (seconds)')
    ax.set_title('Average Time to Reach Goal')
    ax.set_ylim(0, max(average_speeds) * 1.2)

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)


def plot_success_rate(ax, controllers, success_rates):
    """
    Plots a bar chart comparing the success rates of different controllers on a given Axes.
    """
    bars = ax.bar(controllers, success_rates, color=['skyblue', 'salmon'])
    ax.set_xlabel('Controller')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Comparison')
    ax.set_ylim(0, 100)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)


def plot_error_distribution_boxplot(ax, pid_errors, rl_errors):
    """
    Plots boxplots comparing the error distributions of PID and RL controllers on a given Axes.
    """
    ax.boxplot([pid_errors, rl_errors], labels=['PID Error', 'RL Error'], patch_artist=True,
               boxprops=dict(facecolor='skyblue'),
               medianprops=dict(color='black'))
    ax.set_ylabel('Error (meters)')
    ax.set_title('Error Distribution Comparison')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)


def plot_error_over_targets(ax, targets, pid_errors, rl_errors):
    """
    Plots a line chart comparing PID and RL controller errors over target numbers on a given Axes.
    """
    ax.plot(targets, pid_errors, marker='o', label='PID Error', linestyle='-', color='skyblue')
    ax.plot(targets, rl_errors, marker='s', label='RL Error', linestyle='--', color='salmon')
    ax.set_xlabel('Target Number')
    ax.set_ylabel('Error (meters)')
    ax.set_title('Error Over Targets')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)


def plot_speed_over_targets(ax, targets, pid_speeds, rl_speeds):
    """
    Plots a line chart comparing PID and RL controller speeds over target numbers on a given Axes.
    """
    ax.plot(targets, pid_speeds, marker='o', label='PID Time', linestyle='-', color='skyblue')
    ax.plot(targets, rl_speeds, marker='s', label='RL Time', linestyle='--', color='salmon')
    ax.set_xlabel('Target Number')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time to Reach Targets')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)


def create_dashboard(controllers, average_errors, average_speeds, success_rates,
                     pid_errors, rl_errors, targets, pid_speeds, rl_speeds,
                     save_path='performance_dashboard.png'):
    """
    Creates a dashboard with multiple subplots to compare PID and RL controllers.
    """
    # Adjust the subplot grid to 3 rows x 2 columns for better spacing
    fig, axs = plt.subplots(3, 2, figsize=(30, 22))
    fig.suptitle('PID vs RL Controllers Performance Comparison', fontsize=24, fontweight='bold')

    # Top Row
    plot_average_error(axs[0, 0], controllers, average_errors)
    plot_average_speed(axs[0, 1], controllers, average_speeds)

    # Middle Row
    plot_success_rate(axs[1, 0], controllers, success_rates)
    plot_error_distribution_boxplot(axs[1, 1], pid_errors, rl_errors)

    # Bottom Row
    plot_error_over_targets(axs[2, 0], targets, pid_errors, rl_errors)
    plot_speed_over_targets(axs[2, 1], targets, pid_speeds, rl_speeds)

    # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # [left, bottom, right, top]

    # Save and display the dashboard
    plt.savefig('performance_dashboard.svg', format='svg')
    plt.savefig(save_path, dpi=300)  # Increased DPI for better quality
    plt.show()


def generate_random_target_position(env):
    """
    Generates a random target position within the working envelope of the environment.

    Parameters:
        env (OT2Env): The simulation environment.

    Returns:
        np.ndarray: A random [x, y, z] target position in meters.
    """
    # Get workspace boundaries
    x_min, x_max, y_min, y_max, z_min, z_max = env.get_working_envelope_coords()

    # Set a random goal position within the working envelope
    target_position = np.array([
        np.random.uniform(x_min, x_max),
        np.random.uniform(y_min, y_max),
        np.random.uniform(z_min, z_max)
    ], dtype=np.float32)

    return target_position


def move_to_target_position(target_position, env, distance_threshold, max_steps_per_goal):
    """
    Moves the robot's pipette to a target position using both PID and RL controllers.
    Prints the final positions and errors for each controller.

    Parameters:
        target_position (np.ndarray): The desired [x, y, z] position in meters.
        env (OT2Env): The simulation environment.
        distance_threshold
    """
    logging.info(f"\n=== Moving to Target Position: {target_position} ===")

    # Initialize PID Controllers for each axis with specific gains
    pid_x = PIDController(kp=4.97, ki=0.0098, kd=0.203, dt=0.01, integral_limit=(-10, 10))
    pid_y = PIDController(kp=1.58, ki=0.003, kd=0.23, dt=0.01, integral_limit=(-10, 10))
    pid_z = PIDController(kp=2.2, ki=0.001, kd=0.02, dt=0.01, integral_limit=(-10, 10))
    logging.info("Initialized PID Controllers for X, Y, and Z axes.")

    # Initialize RL Controller with the path to your trained model
    rl_model_path = "model_files/rl_best_model_accuracy_0_001.zip"  # Replace with your actual model path
    rl_controller = RLController(model_path=rl_model_path)
    logging.info("Initialized RL Controller.")

    # --- Moving Using PID Controllers ---
    logging.info("\n--- Moving to Target Using PID Controllers ---")
    # Reset PID controllers
    pid_x.reset()
    pid_y.reset()
    pid_z.reset()

    # Reset environment and get initial observation
    obs, _ = env.reset()

    env.goal_position = np.array(target_position, dtype=np.float32)
    logging.info(f"Set Goal Position to: {env.goal_position}")
    done = False
    step = 0

    current_pipette_pos = None
    # Start timing for PID controller
    start_time_pid = time.time()

    while not done and step < max_steps_per_goal:
        # Extract current pipette position and goal position from observation
        current_pipette_pos = obs[:3]
        goal_position = obs[3:]

        # Calculate control signals using PID controllers
        control_x = pid_x.update(target=goal_position[0], current=current_pipette_pos[0])
        control_y = pid_y.update(target=goal_position[1], current=current_pipette_pos[1])
        control_z = pid_z.update(target=goal_position[2], current=current_pipette_pos[2])

        # Create the action array
        action = np.array([control_x, control_y, control_z], dtype=np.float32)

        # Clip actions to match environment action space
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Execute the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Update current position from the new observation
        current_pipette_pos = obs[:3]
        goal_position = obs[3:]

        # Calculate the distance to the goal
        distance = np.linalg.norm(goal_position - current_pipette_pos)
        # logging.info(
        #     f"Step {step + 1}: Current Position: {current_pipette_pos}, Distance to Goal: {distance:.6f} meters")

        # Check if the robot has reached the goal position
        if distance < distance_threshold:  # Threshold: 0.1 mm
            logging.info("PID Controllers: Goal position reached.")
            done = True

        # Check for episode termination or truncation
        if terminated or truncated:
            logging.warning(
                f"PID Controllers: Episode {'terminated' if terminated else 'truncated'} at step {step + 1}. Resetting environment.")
            obs, _ = env.reset()
            done = True

        # Increment step counter
        step += 1

    # End timing for PID controller
    end_time_pid = time.time()
    pid_speed = end_time_pid - start_time_pid

    # Record final position and error for PID
    final_pid_position = current_pipette_pos
    pid_error = np.linalg.norm(target_position - final_pid_position)
    logging.info(f"PID Controllers Final Position: {final_pid_position}")
    logging.info(f"PID Controllers Error: {pid_error:.6f} meters")

    # --- Moving Using RL Controller ---
    logging.info("\n--- Moving to Target Using RL Controller ---")
    # Reset environment and RL controller
    rl_controller.reset()
    env.goal_position = np.array(target_position, dtype=np.float32)
    logging.info(f"Set Goal Position to: {env.goal_position}")

    # Start timing for RL controller
    start_time_rl = time.time()
    # Use RL controller to move to the target position
    final_rl_position = rl_controller.move_to(target_position, env)

    # End timing for RL controller
    end_time_rl = time.time()
    rl_speed = end_time_rl - start_time_rl
    rl_error = np.linalg.norm(target_position - final_rl_position)
    logging.info(f"RL Controller Final Position: {final_rl_position}")
    logging.info(f"RL Controller Error: {rl_error:.6f} meters")

    pid_success = rl_success = 0

    if pid_error < distance_threshold:
        pid_success = 1

    if rl_error < distance_threshold:
        rl_success = 1
    return {
        "PID": {
            "error": pid_error,
            "speed": pid_speed,
            "success": pid_success
        },
        "RL": {
            "error": rl_error,
            "speed": rl_speed,
            "success": rl_success
        }
    }


def main():
    """
    Main function to generate target positions and move the robot to each position.
    """
    # Initialize the simulation environment
    distance_threshold = 0.001
    max_steps = max_steps_per_goal = 5000
    env = OT2Env(render=False, max_steps=max_steps, distance_threshold=distance_threshold)

    # Define the number of target positions to generate
    num_targets = 1000

    targets = list(range(1, num_targets + 1))

    # Define tracking lists
    pid_errors = []
    pid_speeds = []
    rl_errors = []
    rl_speeds = []
    pid_successes = []
    rl_successes = []

    for i in range(num_targets):
        logging.info(f"\n=== Benchmarking Target {i + 1} ===")
        # Generate a random target position
        target_position = generate_random_target_position(env)
        logging.info(f"Generated Random Target Position: {target_position}")

        # Move to the generated target position
        metrics = move_to_target_position(target_position, env, distance_threshold,
                                          max_steps_per_goal=max_steps_per_goal)

        # Append metrics to tracking lists
        pid_errors.append(metrics["PID"]["error"])
        pid_speeds.append(metrics["PID"]["speed"])
        rl_errors.append(metrics["RL"]["error"])
        rl_speeds.append(metrics["RL"]["speed"])
        pid_successes.append(metrics["PID"]["success"])
        rl_successes.append(metrics["RL"]["success"])

    # Close the environment after benchmarking
    env.close()
    logging.info("Benchmarking Completed.")

    # --- Reporting Metrics ---
    logging.info("\n=== Benchmarking Results ===")
    for i in range(num_targets):
        logging.info(f"\n--- Target {i + 1} ---")
        logging.info(f"PID Controller -> Error: {pid_errors[i]:.6f} meters, Speed: {pid_speeds[i]:.2f} seconds")
        logging.info(f"RL Controller  -> Error: {rl_errors[i]:.6f} meters, Speed: {rl_speeds[i]:.2f} seconds")

    # Calculate Success Rates
    pid_success_rate = (np.sum(pid_successes) / num_targets) * 100
    rl_success_rate = (np.sum(rl_successes) / num_targets) * 100

    avg_pid_error = np.mean(pid_errors)
    std_pid_error = np.std(pid_errors)
    avg_pid_speed = np.mean(pid_speeds)
    std_pid_speed = np.std(pid_speeds)

    avg_rl_error = np.mean(rl_errors)
    std_rl_error = np.std(rl_errors)
    avg_rl_speed = np.mean(rl_speeds)
    std_rl_speed = np.std(rl_speeds)

    # Additional Error Metrics
    mae_pid = np.mean(np.abs(pid_errors))
    mae_rl = np.mean(np.abs(rl_errors))

    rmse_pid = np.sqrt(np.mean(np.square(pid_errors)))
    rmse_rl = np.sqrt(np.mean(np.square(rl_errors)))

    max_error_pid = np.max(pid_errors)
    max_error_rl = np.max(rl_errors)

    # Success Rates
    pid_success_rate = (np.sum(pid_successes) / num_targets) * 100
    rl_success_rate = (np.sum(rl_successes) / num_targets) * 100

    logging.info("\n=== Summary Statistics ===")
    logging.info(f"PID Controller -> Average Error: {avg_pid_error:.6f} meters, STD Error: {std_pid_error:.6f} meters")
    logging.info(
        f"PID Controller -> Average Speed: {avg_pid_speed:.2f} seconds, STD Speed: {std_pid_speed:.2f} seconds")
    logging.info(f"RL Controller  -> Average Error: {avg_rl_error:.6f} meters, STD Error: {std_rl_error:.6f} meters")
    logging.info(f"RL Controller  -> Average Speed: {avg_rl_speed:.2f} seconds, STD Speed: {std_rl_speed:.2f} seconds")

    logging.info(
        f"PID Controller -> MAE: {mae_pid:.6f} meters, RMSE: {rmse_pid:.6f} meters, Max Error: {max_error_pid:.6f} meters")
    logging.info(
        f"RL Controller  -> MAE: {mae_rl:.6f} meters, RMSE: {rmse_rl:.6f} meters, Max Error: {max_error_rl:.6f} meters")

    logging.info(f"PID Controller -> Success Rate: {pid_success_rate:.2f}%")
    logging.info(f"RL Controller  -> Success Rate: {rl_success_rate:.2f}%")

    # --- Visualization ---
    controllers = ['PID Controller', 'RL Controller']
    average_errors = [avg_pid_error, avg_rl_error]
    average_speeds = [avg_pid_speed, avg_rl_speed]
    success_rates = [pid_success_rate, rl_success_rate]

    # Call the dashboard plotting function
    create_dashboard(controllers, average_errors, average_speeds, success_rates,
                     pid_errors, rl_errors, targets, pid_speeds, rl_speeds,
                     save_path='performance_dashboard.png')


if __name__ == "__main__":
    main()
