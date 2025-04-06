import logging

import numpy as np


def pixel_to_mm(pixel_coords, conversion_factor_x, conversion_factor_y):
    """
    Converts pixel coordinates to mm-space.

    Parameters:
        pixel_coords (tuple): (x_pixel, y_pixel)
        conversion_factor_x (float): mm per pixel in x-axis
        conversion_factor_y (float): mm per pixel in y-axis

    Returns:
        np.array: [x_mm, y_mm]
    """
    x_pixel, y_pixel = pixel_coords
    # Center of the pixel
    x_mm = (x_pixel + 0.5) * conversion_factor_x
    y_mm = (y_pixel + 0.5) * conversion_factor_y
    return np.array([x_mm, y_mm])


def mm_to_robot(mm_coords, plate_position_robot):
    """
    Converts mm-space coordinates to robot's coordinate space.

    Parameters:
        mm_coords (np.array): [x_mm, y_mm]
        plate_position_robot (np.array): [x_robot, y_robot, z_robot] in meters

    Returns:
        np.array: [x_robot, y_robot, z_robot]
    """
    x_mm, y_mm = mm_coords

    # Convert mm to meters and add plate position
    x_robot = (x_mm / 1000) + plate_position_robot[0]
    y_robot = (y_mm / 1000) + plate_position_robot[1]
    z_robot = plate_position_robot[2]  # Assuming z remains constant
    return np.array([x_robot, y_robot, z_robot])


def process_detected_positions(detected_positions, conversion_factor_x, conversion_factor_y, plate_position_robot):
    """
    Convert detected root positions from image coordinates to robot coordinates,
    with sorting and logging.

    Parameters:
    ----------
    detected_positions : list of tuple
        List of detected root positions in image pixel coordinates [(x, y), ...].
    conversion_factor_x : float
        Conversion factor from pixels to millimeters along the x-axis.
    conversion_factor_y : float
        Conversion factor from pixels to millimeters along the y-axis.
    plate_position_robot : tuple
        Robot's reference position of the plate in millimeter coordinates (x, y, z).

    Returns:
    -------
    list of tuple
        Sorted list of detected root positions in robot coordinates.
    """

    if detected_positions is None:
        logging.error("Root tip detection returned None. Check your detection function.")
        env.close()
        return
    logging.info(f"Detected root tips (image coordinates): {detected_positions}")
    detected_root_positions_robot = []

    # Collect mm_coords_swapped for sorting
    mm_coords_swapped_list = []
    for pixel in detected_positions:
        mm_coords = pixel_to_mm(pixel, conversion_factor_x, conversion_factor_y)
        mm_coords_swapped = (mm_coords[1], mm_coords[0])  # Swap x and y
        mm_coords_swapped_list.append(mm_coords_swapped)

    # Sort by the x-coordinate (the first element in mm_coords_swapped)
    mm_coords_swapped_list_sorted = sorted(mm_coords_swapped_list, key=lambda coords: coords[1])

    # Convert sorted mm_coords_swapped to robot coordinates
    for mm_coords_swapped in mm_coords_swapped_list_sorted:
        robot_coords = mm_to_robot(mm_coords_swapped, plate_position_robot)
        detected_root_positions_robot.append(robot_coords)
        logging.info(f"Sorted Swapped mm {mm_coords_swapped} -> Robot {robot_coords}")

    return detected_root_positions_robot


def calculate_conversion_factors(conversion_values, plate_size_mm):
    """
    Calculate conversion factors for converting pixel dimensions to millimeters.

    Parameters:
    ----------
    conversion_values : tuple
        A tuple containing the dimensions of the image in pixels as (height_px, width_px).
    plate_size_mm : float
        The size of the plate in millimeters (assumed square).

    Returns:
    -------
    tuple
        Conversion factors as (conversion_factor_x, conversion_factor_y), in mm/pixel.
    """
    # Extract image dimensions
    image_height_px, image_width_px = conversion_values

    # Calculate conversion factors
    conversion_factor_x = plate_size_mm / image_width_px  # mm per pixel in x-axis
    conversion_factor_y = plate_size_mm / image_height_px  # mm per pixel in y-axis

    # Log and print conversion factors
    print(f"Image width px: {image_width_px}")
    print(f"Image height px: {image_height_px}")
    logging.info(f"Conversion Factor X: {conversion_factor_x:.4f} mm/pixel")
    logging.info(f"Conversion Factor Y: {conversion_factor_y:.4f} mm/pixel")

    return conversion_factor_x, conversion_factor_y
