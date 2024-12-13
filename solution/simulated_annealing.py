import numpy as np
from visualizer import visualize_matrix, overlay_png_on_png
from janina_preprocessing import preprocess_image, extract_features, extract_gripper_positions
from scipy.ndimage import rotate
inv_part_mask: np.ndarray
gripper_mask: np.ndarray
temp_init = 100
max_iter = 1000
alpha = 0.999
temp_min = 1e-6

def objective_function(x, y, angle):
    # Rotate the smaller matrix around its center
    rotated_small_matrix = rotate(gripper_mask, angle, reshape=True, order=1, mode='constant', cval=0)

    rows, cols = inv_part_mask.shape
    rows_gripper, cols_gripper = rotated_small_matrix.shape

    rotated_gripper_mask = np.zeros_like(inv_part_mask)
    # visualize_matrix(radius_mask, "solution/visualization/radius_mask.png")
    x_shift = x - cols_gripper // 2
    y_shift = y - rows_gripper // 2

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
    rotated_gripper_mask[y_start:y_end, x_start:x_end] = rotated_small_matrix[orig_y_start:orig_y_end, orig_x_start:orig_x_end]

    # visualize_matrix(rotated_gripper_mask, "solution/visualization/rotated_gripper_mask.png")

    biggest_dist = np.sqrt((cols//2)**2 + (rows//2)**2)
    intersection = np.sum(np.bitwise_and(inv_part_mask, rotated_gripper_mask))
    dist_to_center = np.sqrt((cols//2 - x)**2 + (rows//2 - y)**2)
    if intersection == 0:
        print("no intersection")
        return dist_to_center
    return intersection + biggest_dist


def simulated_annealing(initial_solution):
    # Start with the initial solution
    current_solution = initial_solution
    current_value = objective_function(current_solution[0], current_solution[1], current_solution[2])

    # Set the initial temperature
    temperature = temp_init

    # Keep track of the best solution found
    best_solution = current_solution
    best_value = current_value

    # List to store solutions for plotting or analysis
    solutions = [(current_solution, current_value)]

    # Start the main loop of the simulated annealing process
    for i in range(max_iter):
        # print("Iteration:", i, "Best value:", best_value)
        # Generate a neighbor solution by making a small random change
        neighbor_solution = [max(0, min(current_solution[0] + np.random.randint(-1, 2), inv_part_mask.shape[1] - 1)),
                             max(0, min(current_solution[1] + np.random.randint(-1, 2), inv_part_mask.shape[0] - 1)),
                             current_solution[2] + np.random.uniform(-10, 10) % 360]

        neighbor_value = objective_function(*neighbor_solution)

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

        # Record the current solution for analysis
        solutions.append((current_solution, current_value))

        # Reduce the temperature according to the cooling schedule
        temperature = temperature * alpha
        if temperature < temp_min:
            break

    return best_solution

def find_best_gripper_position(part_image_path, gripper_image_path):
    global gripper_mask, inv_part_mask
    metal_image, gray_metal, binary_metal, alpha_metal = preprocess_image(part_image_path)
    metal_contour, holes_mask, inv_part_mask = extract_features(binary_metal)
    gripper_positions, gripper_center, gripper_mask = extract_gripper_positions(gripper_image_path)

    initial_solution = (holes_mask.shape[1]//2, holes_mask.shape[0]//2, 0)
    best_solution = simulated_annealing(initial_solution)

    overlay_png_on_png(
        png_path=part_image_path,
        overlay_path=gripper_image_path,
        output_path='solution/visualization/final_solution_' + str(part_image_path).split('/')[-2] + '.png',
        x=best_solution[0],
        y=best_solution[1],
        angle=best_solution[2])

    return best_solution[0], best_solution[1], best_solution[2]

