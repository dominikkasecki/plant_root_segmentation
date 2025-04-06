import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_root_tips_with_image(image_path, root_tips):
    """
    Plot the cropped image with root tips marked.

    Args:
        image_path (str): The image path of  the original image.
        root_tips (list of tuple): List of (x, y) coordinates of root tips.

    Returns:
        None
    """
    # Plot the cropped image
    image = cv2.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image with Root Tips")

    # Plot root tips as red dots
    for tip in root_tips:
        plt.plot(tip[0], tip[1], 'ro', markersize=5, label='Root Tip' if tip == root_tips[0] else "")

    plt.legend()
    plt.axis("off")
    plt.show()


def delete_directory(directory_path):
    """
    Delete the specified directory and all its contents, then recreate it as an empty directory.

    Parameters:
    ----------
    directory_path : str
        Path to the directory to delete and recreate.

    Returns:
    -------
    None
    """
    path = Path(directory_path)
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_dir():
                delete_directory(item)  # Recursively delete subdirectories
            else:
                item.unlink()  # Delete files
        path.rmdir()  # Remove the now-empty directory
        print(f"Deleted directory: {directory_path}")
    else:
        print(f"Directory does not exist or is not a directory: {directory_path}")

    # Recreate the empty directory
    path.mkdir(parents=True, exist_ok=True)
    print(f"Recreated empty directory: {directory_path}")


def extract_petri_dish(image_path):
    """
    Extract the region of interest (ROI) corresponding to a Petri dish from an image.

    Parameters:
    ----------
    image_path : str
        Path to the input image containing the Petri dish.

    Returns:
    -------
    tuple
        A tuple containing:
        - cropped_roi_gray.shape (tuple): Shape of the cropped region (height, width).
    """
    # Read the image
    input_img = cv2.imread(image_path)
    if input_img is None:
        print(f"Error: Unable to load image from {image_path}")
        return None, None, None, (0, 0)

    # Convert to grayscale
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding with a lower threshold to better capture edges
    _, binary = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)  # Lowered threshold from 127

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours of the main dish
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return input_img, binary, input_img, (0, 0)

    # Find the largest rectangular contour (should be the Petri dish)
    main_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle with increased padding
    x, y, w, h = cv2.boundingRect(main_contour)
    padding = 30  # Increased padding to ensure we capture the full dish

    # Add extra validation
    aspect_ratio = float(w) / h
    if not (0.9 <= aspect_ratio <= 1.1):  # Check if it's roughly square
        print(f"Warning: Unusual aspect ratio detected: {aspect_ratio}")
        # Adjust to make it square
        max_dim = max(w, h)
        w = h = max_dim

    # Ensure we capture the full square dish
    largest_side = max(w, h) + 2 * padding
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the crop coordinates
    half_side = largest_side // 2
    x_min = max(0, center_x - half_side)
    y_min = max(0, center_y - half_side)
    x_max = min(input_img.shape[1], center_x + half_side)
    y_max = min(input_img.shape[0], center_y + half_side)

    # Add symmetry check
    if (x_max - x_min) != (y_max - y_min):
        # Ensure square crop by taking the larger dimension
        size = max(x_max - x_min, y_max - y_min)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        half_size = size // 2

        x_min = max(0, center_x - half_size)
        y_min = max(0, center_y - half_size)
        x_max = min(input_img.shape[1], center_x + half_size)
        y_max = min(input_img.shape[0], center_y + half_size)

    # Extract the ROI
    cropped_roi = input_img[y_min:y_max, x_min:x_max]
    cropped_roi_gray = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)

    return cropped_roi_gray.shape


def capture_plate_image(env):
    """
    Capture the plate image from the environment and handle exceptions.

    Parameters:
    ----------
    env : object
        The environment object with a `get_plate_image` method.

    Returns:
    -------
    np.ndarray or None
        The captured plate image as a numpy array, or None if an error occurs.
    """
    try:
        # Step 1: Capture the plate image
        image = env.get_plate_image()
        logging.info("Captured plate image successfully.")
    except FileNotFoundError as fnf_error:
        logging.error(f"File Not Found Error: {fnf_error}")
        env.close()
        return None
    except ValueError as ve:
        logging.error(f"Value Error: {ve}")
        env.close()
        return None

    # Optional: Visualize the captured image
    plt.imshow(image, cmap='gray')
    plt.title("Captured Plate Image")
    plt.axis('off')
    plt.show()

    return image


def pause_for_observation(env, pause_steps=100):
    """
    Pause the environment for a specified number of steps to observe its state.

    Parameters:
    ----------
    env : object
        The environment object with a `step` method for performing actions.
    pause_steps : int, optional
        Number of steps to pause with no movement (default is 100).

    Returns:
    -------
    None
    """
    logging.info("All root tips inoculated. Pausing for observation...")

    no_movement_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # No movement action

    for pause_step in range(pause_steps):
        _, _, _, _, _ = env.step(no_movement_action)
        logging.info(f"Pause Step {pause_step + 1}: Observing environment.")
