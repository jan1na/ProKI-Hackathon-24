import numpy as np
from deeplabv3_resnet101_preprocessing import predict
from janina_preprocessing import get_gripper_binary
from scipy.ndimage import rotate

inv_part_mask: np.ndarray
gripper_mask: np.ndarray
temp_init = 100
max_iter = 1000
alpha = 0.999
temp_min = 1e-6

# Set random seed for reproducibility
np.random.seed(42)


def objective_function(x: int, y: int, angle: float) -> (float, bool):
    """
    Objective function to minimize the distance to the center of the part.

    :param x: x-coordinate of the gripper
    :param y: y-coordinate of the gripper
    :param angle: angle of the gripper
    :return: distance to the center of the part, not outside the image
    """
    # Rotate the gripper mask by the given angle
    rotated_small_matrix = rotate(gripper_mask, angle, reshape=True, order=1, mode='constant', cval=0)

    rows, cols = inv_part_mask.shape
    rows_gripper, cols_gripper = rotated_small_matrix.shape

    rotated_gripper_mask = np.zeros_like(inv_part_mask)
    x_shift = x - cols_gripper // 2
    y_shift = y - rows_gripper // 2

    # Calculate the valid range for the shifted matrix
    x_start = np.maximum(0, x_shift)
    x_end = np.minimum(cols, x_shift + cols_gripper)
    y_start = np.maximum(0, y_shift)
    y_end = np.minimum(rows, y_shift + rows_gripper)

    orig_x_start = np.maximum(0, -x_shift)
    orig_x_end = orig_x_start + (x_end - x_start)

    orig_y_start = np.maximum(0, -y_shift)
    orig_y_end = orig_y_start + (y_end - y_start)

    # Copy the valid range from the original matrix to the shifted matrix
    rotated_gripper_mask[y_start:y_end, x_start:x_end] = rotated_small_matrix[orig_y_start:orig_y_end,
                                                         orig_x_start:orig_x_end]

    # Check if the rotated gripper mask is valid
    if rotated_gripper_mask.sum() < rotated_small_matrix.sum():
        return float('inf') - (rotated_small_matrix.sum() - rotated_gripper_mask.sum()), False

    biggest_dist = np.sqrt((cols // 2) ** 2 + (rows // 2) ** 2)
    intersection = np.sum(np.bitwise_and(inv_part_mask, rotated_gripper_mask))
    dist_to_center = np.sqrt((cols / 2 - x) ** 2 + (rows / 2 - y) ** 2)
    if intersection == 0:
        return dist_to_center, True
    return intersection + biggest_dist, True


def simulated_annealing(initial_solution: (int, int, float)) -> (int, int, float):
    """
    Simulated Annealing algorithm to find the best gripper position.

    :param initial_solution: initial solution
    :return: x, y, angle of the best gripper position
    """
    # Start with the initial solution
    current_solution = initial_solution
    current_value, not_outside = objective_function(current_solution[0], current_solution[1], current_solution[2])

    # Set the initial temperature
    temperature = temp_init

    # Keep track of the best solution found
    best_solution = current_solution
    best_value = current_value

    # Start the main loop of the simulated annealing process
    for i in range(max_iter):
        # Generate a neighbor solution by making a small random change
        shifts = np.random.randint(-1, 2, size=2)
        neighbor_solution = [
            max(0, min(current_solution[0] + shifts[0], inv_part_mask.shape[1] - 1)),
            max(0, min(current_solution[1] + shifts[1], inv_part_mask.shape[0] - 1)),
            (current_solution[2] + np.random.uniform(-5, 5)) % 360
        ]

        neighbor_value, not_outside = objective_function(*neighbor_solution)

        # Calculate the difference in objective values
        delta_e = neighbor_value - current_value

        # If the neighbor is better, accept it
        if delta_e < 0:
            current_solution = neighbor_solution
            current_value = neighbor_value
        else:
            # If the neighbor is worse, accept it with a probability
            acceptance_probability = np.exp(-delta_e / temperature)
            if np.random.rand() < acceptance_probability:
                current_solution = neighbor_solution
                current_value = neighbor_value

        # Update the best solution found so far
        if current_value < best_value:
            best_solution = current_solution
            best_value = current_value

        # Reduce the temperature according to the cooling schedule
        temperature = temperature * alpha
        if temperature < temp_min:
            break

    return best_solution


def find_best_gripper_position(part_image_path: str, gripper_image_path: str, num_runs: int = 3) -> (int, int, float):
    """
    Find the best gripper position using simulated annealing.

    :param part_image_path: path to the part image
    :param gripper_image_path: path to the gripper image
    :param num_runs: number of runs for the simulated annealing algorithm
    :return: x, y, angle of the best gripper position
    """
    global gripper_mask, inv_part_mask
    part_mask = predict(part_image_path)
    inv_part_mask = 1 - part_mask

    gripper_mask = get_gripper_binary(gripper_image_path)

    initial_solution = (inv_part_mask.shape[1] // 2, inv_part_mask.shape[0] // 2, 0)
    best_solution = initial_solution
    best_value = float('inf')

    for _ in range(num_runs):
        angle = 0
        # Find an initial angle with which the gripper is not outside the image
        while angle < 360:
            current_value, not_outside = objective_function(*initial_solution)
            if not_outside:
                break
            else:
                angle += 5
                initial_solution = (initial_solution[0], initial_solution[1], angle)

        current_solution = simulated_annealing(initial_solution)
        current_value, not_outside = objective_function(*current_solution)
        if current_value < best_value:
            best_solution = current_solution
            best_value = current_value

    return best_solution[0], best_solution[1], best_solution[2]
