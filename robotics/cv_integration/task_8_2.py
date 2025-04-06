from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_img(path):
    """
    Load a grayscale image from a specified file path and binarize it.

    Parameters:
    ----------
    path : str
        The file path of the image to load.

    Returns:
    -------
    numpy.ndarray
        A binary image (dtype 'uint8'), where pixel values are 1 for non-zero
        intensity in the original image and 0 otherwise.
    """
    img = cv2.imread(path, 0)  # Load image in grayscale mode
    img = (img > 0).astype('uint8')  # Binarize the image
    return img


def show_img(img, cmap='gray', title=None):
    """
    Display an image using matplotlib.

    Parameters:
    ----------
    img : numpy.ndarray
        The image to display.
    cmap : str, optional
        The colormap used for displaying the image (default is 'gray').
    title : str, optional
        The title of the displayed image (default is None).

    Returns:
    -------
    None
    """
    plt.figure(dpi=300)
    plt.axis('off')  # Hide axes for cleaner display
    if title:
        plt.title(title)  # Set the title if provided
    plt.imshow(img, cmap=cmap)  # Display the image with the specified colormap
    plt.show()


def preprocess_mask(mask, close_kernel_size=7, dilate_kernel_size=3):
    """
    Preprocess a binary mask to close gaps, clean noise, and enhance connectivity.

    Parameters:
    ----------
    mask : numpy.ndarray
        Binary mask to preprocess. Pixel values should be 0 or 255.
    close_kernel_size : int, optional
        Size of the structuring element used for the morphological closing operation (default is 7).
    dilate_kernel_size : int, optional
        Size of the structuring element used for the dilation operation (default is 3).

    Returns:
    -------
    numpy.ndarray
        Processed binary mask after applying closing, dilation, and further closing operations.
    """
    # Apply closing operation with larger kernel to close gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Apply dilation with slightly larger kernel
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=2)

    # Final closing to ensure connectivity
    mask_final = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel_close)

    return mask_final


def classify_component(stats, img_height):
    """
    Classify components in an image based on their size, aspect ratio, and position.

    Parameters:
    ----------
    stats : tuple
        A tuple containing the following information for a component:
        - x (int): The x-coordinate of the bounding box.
        - y (int): The y-coordinate of the bounding box.
        - width (int): The width of the bounding box.
        - height (int): The height of the bounding box.
        - area (int): The area of the component.
    img_height : int
        The height of the image, used for calculating relative measurements.

    Returns:
    -------
    bool
        True if the component satisfies the classification criteria, otherwise False.
    """
    x, y, width, height, area = stats

    height_ratio = height / img_height
    aspect_ratio = height / width if width > 0 else float('inf')
    vertical_position = (y + height) / img_height
    print(f"Coordinates (X, Y): ({x}, {y})")
    print(f"Area of this root: {area}")
    print(f"Height Ratio (is_tall_enough): {height_ratio:.2f}")
    print(f"Aspect Ratio (is_somewhat_vertical): {aspect_ratio:.2f}")
    print(f"Vertical Position: {vertical_position:.2f}")

    print('=' * 50)

    # Size-based classification
    if area <= 1500:  # Tiny roots
        is_tall_enough = height_ratio >= 0.005
        is_somewhat_vertical = aspect_ratio > 0.15
        extends_down = vertical_position > 0.01
        return is_tall_enough and is_somewhat_vertical and extends_down

    elif area <= 2500:  # Small roots
        is_tall_enough = height_ratio >= 0.05
        is_somewhat_vertical = aspect_ratio > 0.5
        extends_down = vertical_position > 0.15

    elif 1500 < area <= 4500:  # Medium roots
        is_tall_enough = height_ratio >= 0.1
        is_somewhat_vertical = aspect_ratio > 0.8
        extends_down = vertical_position > 0.2

    else:  # Large roots
        is_tall_enough = height_ratio >= 0.15
        extends_down = vertical_position > 0.20
        if height_ratio < 0.25:
            is_somewhat_vertical = aspect_ratio > 1.0
        else:
            is_somewhat_vertical = aspect_ratio > 1.8

    return (is_tall_enough or extends_down) and is_somewhat_vertical


def find_root_bottom_tip_and_centroid(labels, component_idx):
    """
    Find both the root bottom tip (lowest point) and the centroid for a component.

    Args:
        labels: Label matrix from connected components
        component_idx: Index of the component to analyze

    Returns:
        tuple: ((centroid_x, centroid_y), (bottom_tip_x, bottom_tip_y))
    """
    # Mask for the current component
    component_mask = (labels == component_idx)
    y_coords, x_coords = np.nonzero(component_mask)

    if len(x_coords) == 0 or len(y_coords) == 0:
        return (0, 0), (0, 0)

    # Find the bottom tip (lowest y-coordinate)
    bottom_idx = np.argmax(y_coords)  # Max y-coordinate
    bottom_tip_x = x_coords[bottom_idx]
    bottom_tip_y = y_coords[bottom_idx]

    # Calculate centroid
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))

    return (centroid_x, centroid_y), (bottom_tip_x, bottom_tip_y)


def filter_components(labels, stats, min_cut_off=50, max_cut_off=1600):
    """
    Filter labeled components in an image based on adaptive size thresholds and spatial constraints.

    Parameters:
    ----------
    labels : numpy.ndarray
        The labeled image where each connected component has a unique label.
    stats : numpy.ndarray
        Statistics for each connected component as returned by `cv2.connectedComponentsWithStats`.
        Includes the following for each component:
        - [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA].
    min_cut_off : int, optional
        Minimum vertical position (top boundary) for a component to be considered valid (default is 50).
    max_cut_off : int, optional
        Maximum vertical position (top boundary) for a component to be considered valid (default is 1600).

    Returns:
    -------
    tuple
        - filtered_output (numpy.ndarray): A 3-channel image with valid components colored uniquely.
        - filtered_mask (numpy.ndarray): A binary mask of the valid components.
        - valid_count (int): The number of valid components.
        - valid_centroids (list of tuple): List of coordinates for the tips of valid components.
        - bottom_root_tips (list of tuple): List of coordinates for the bottom tips of valid components.

    Notes:
    ------
    - Adaptive size thresholds are applied based on the largest component's area.
    - Valid components are further filtered using `classify_component`.
    - Each valid component is uniquely colored in the output image for visualization.
    """
    img_height, img_width = labels.shape

    # Find the largest component area to determine adaptive size thresholds
    component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background (index 0)
    if len(component_areas) > 0:
        largest_area = np.max(component_areas)
        print(f"Largest area: {largest_area}")
        print(f"Smallest area: {np.min(component_areas)}")

        # Adaptive size thresholds
        if largest_area <= 1500:  # Tiny roots
            min_size = 30
            max_size = 1500
            max_cut_off = 1050
        elif largest_area <= 2500:  # Small roots
            min_size = 150
            max_size = 2500
        elif largest_area <= 7000:  # Medium roots
            min_size = 600
            max_size = 7000
        else:  # Large roots
            min_size = 5000
            max_size = None
    else:
        min_size = 500
        max_size = None

    # Initialize outputs and variables
    filtered_output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    filtered_mask = np.zeros(labels.shape, dtype=np.uint8)
    num_labels = stats.shape[0]
    colors = np.random.randint(0, 256, size=(num_labels, 3), dtype=np.uint8)

    valid_components = []

    bottom_root_tips = []

    # Process each component
    for i in range(1, num_labels):  # Skip background
        component_area = stats[i, cv2.CC_STAT_AREA]
        if ((max_size is None or component_area <= max_size) and
                component_area >= min_size and
                min_cut_off < stats[i, cv2.CC_STAT_TOP] < max_cut_off and
                classify_component(stats[i], img_height)):
            # Calculate centroid and tip coordinates

            valid_components.append((i, stats[i, cv2.CC_STAT_LEFT]))

            (_, bottom_tip) = find_root_bottom_tip_and_centroid(labels, i)
            bottom_root_tips.append(bottom_tip)

    # Sort valid components by horizontal position (left boundary)
    valid_components.sort(key=lambda x: x[1])

    # Generate output images
    for idx, (comp_idx, _) in enumerate(valid_components, 1):
        filtered_output[labels == comp_idx] = colors[idx]
        filtered_mask[labels == comp_idx] = 255

    print(f"Processing with size range: {min_size} - {max_size if max_size else 'unlimited'}")

    return filtered_output, filtered_mask, len(valid_components), bottom_root_tips


def segment_roots(img_path):
    """
    Segment root tips from a given mask image using preprocessing, filtering, and connected components analysis.

    Parameters:
    ----------
    img_path : str
        Path to the mask image to be processed.

    Returns:
    -------
    list
        A list of bottom root tip coordinates extracted from the segmented components.
    """
    # Load mask
    predicted_root_mask = load_img(img_path)
    show_img(predicted_root_mask, title="Original Mask")

    # Enhanced preprocessing with adjusted parameters
    mask_processed = preprocess_mask(
        predicted_root_mask,
        close_kernel_size=5,
        dilate_kernel_size=3
    )
    show_img(mask_processed, title="Preprocessed Mask")

    # Connected components analysis
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_processed, connectivity=8, ltype=cv2.CV_32S
    )

    # Filter components and extract root tips
    filtered_output, filtered_mask, num_components, bottom_root_tips = filter_components(
        labels,
        stats,
        min_cut_off=50,
        max_cut_off=1600
    )
    show_img(filtered_output, title=f"Detected Components: {num_components}")

    return bottom_root_tips


def segment_predictions(predictions_dir='./predictions'):
    """
    Segment root tips from predicted mask files in a specified directory.

    Parameters:
    ----------
    predictions_dir : str, optional
        Path to the directory containing prediction mask files (default is './predictions').

    Returns:
    -------
    list
        A list of results from `segment_roots` for each prediction file, containing root tips or related data.
    """
    predictions_dir = Path(predictions_dir)

    # Get and sort prediction file paths
    path_to_strs = [str(img_path) for img_path in list(predictions_dir.iterdir())]
    print(path_to_strs)
    path_to_strs = sorted(path_to_strs, key=lambda x: int(x.split('_')[2]))

    root_tips = []
    for pred_file in path_to_strs:
        print(pred_file)

        # Segment roots from each predicted mask file
        res = segment_roots(str(pred_file))
        root_tips.append(res)

    return root_tips
