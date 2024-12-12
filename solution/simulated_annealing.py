import numpy as np
from visualizer import visualize_matrix
from janina_preprocessing import preprocess_image, extract_features, extract_gripper_positions

def objective_function(x, y, angle, inv_part_mask, gripper_mask):
    rows, cols = inv_part_mask.shape
    rows_gripper, cols_gripper = gripper_mask.shape

    rotated_gripper_mask = np.zeros_like(inv_part_mask)
    # visualize_matrix(radius_mask, "solution/visualization/radius_mask.png")
    x_shift = x - gripper_mask.shape[1] // 2
    y_shift = y - gripper_mask.shape[0] // 2

    # Calculate the valid range for the shifted matrix
    x_start = max(0, x_shift)
    x_end = min(cols, x_shift + cols_gripper)

    y_start = max(0, y_shift)
    y_end = min(rows, y_shift + rows_gripper)

    # Determine corresponding ranges in the original matrix
    orig_x_start = max(0, -x_shift)
    orig_x_end = orig_x_start + (x_end - x_start)

    orig_y_start = max(0, -y_shift)
    orig_y_end = orig_y_start + (y_end - y_start)

    # Copy the valid range from the original matrix to the shifted matrix
    rotated_gripper_mask[y_start:y_end, x_start:x_end] = gripper_mask[orig_y_start:orig_y_end, orig_x_start:orig_x_end]

    visualize_matrix(rotated_gripper_mask, "solution/visualization/rotated_gripper_mask.png")

    biggest_dist = np.sqrt((cols//2)**2 + (rows//2)**2)
    intersection = np.sum(np.bitwise_and(inv_part_mask, gripper_mask))
    dist_to_center = np.sqrt((cols//2 - x)**2 + (rows//2 - y)**2)
    if intersection == 0:
        return dist_to_center
    return intersection + biggest_dist


def simulated_annealing(initial_solution):
    pass

def find_best_gripper_position(part_image_path, gripper_image_path):
    metal_image, gray_metal, binary_metal = preprocess_image(part_image_path)
    metal_contour, holes_mask = extract_features(binary_metal)
    gripper_positions, gripper_center, binary = extract_gripper_positions(gripper_image_path)

    initial_solution = (0, 0, 0)
    center, angle = simulated_annealing(initial_solution)

    return center[0], center[1], angle

