from pathlib import Path
from argparse import ArgumentParser

from rich.progress import track
import pandas as pd

import cv2
import numpy as np
from scipy.optimize import minimize
import cairosvg
from visualizer import overlay_png_on_png, visualize_matrix
import math

i = 0


def generate_clean_mask(image_path):
    # Read the input image
    image_unchanged = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(image_unchanged)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image[a == 0] = 255

    smoothed_image = cv2.GaussianBlur(image, (5, 5), 5)
    visualize_matrix(smoothed_image, "solution/visualization/1_smoothed_image.png")

    # Threshold to create an initial binary mask
    _, binary_mask = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)

    visualize_matrix(binary_mask, "solution/visualization/2_binary_mask.png")

    # Remove small noise and dust
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_area = 10  # Adjust as needed
    cleaned_mask = np.zeros_like(binary_mask)
    print("num_lables:", num_labels)
    visualize_matrix(labels, "solution/visualization/3_labels.png")


    for i in range(1, num_labels):  # Skip the background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255

    visualize_matrix(cleaned_mask, "solution/visualization/4_cleand_mask.png")

    # Edge detection and contour filling for borders
    edges = cv2.Canny(cleaned_mask, threshold1=50, threshold2=150)
    visualize_matrix(edges, "solution/visualization/edges.png")
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours:", contours)
    # visualize_matrix(contours, "solution/visualization/contours.png")

    final_mask = np.zeros_like(cleaned_mask)
    cv2.drawContours(final_mask, contours, -1, (255), thickness=cv2.FILLED)

    visualize_matrix(final_mask, "solution/visualization/5_final_mask.png")

    return final_mask

def load_and_preprocess_part(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary


def load_gripper_from_svg(svg_path):
    # Convert SVG to PNG using cairosvg
    with open(svg_path, "rb") as svg_file:
        cairosvg.svg2png(file_obj=svg_file, write_to="gripper.png")
    # Load the PNG version of the gripper
    gripper_image = cv2.imread("gripper.png", cv2.IMREAD_GRAYSCALE)
    _, binary_gripper = cv2.threshold(gripper_image, 127, 255, cv2.THRESH_BINARY)
    return binary_gripper


def gripper_collision_check(part_mask, gripper_mask, x, y, theta):
    # Rotate the gripper mask
    rotation_matrix = cv2.getRotationMatrix2D((gripper_mask.shape[1] // 2, gripper_mask.shape[0] // 2), theta, 1)
    rotated_gripper = cv2.warpAffine(gripper_mask, rotation_matrix, (part_mask.shape[1], part_mask.shape[0]))

    # Get dimensions of the rotated gripper
    g_h, g_w = rotated_gripper.shape
    p_h, p_w = part_mask.shape
    print("HERE")
    print(part_mask)
    visualize_matrix(rotation_matrix, "solution/visualization/rotated_gripper.png")
    visualize_matrix(part_mask, "solution/visualization/part_mask.png")
    visualize_matrix(gripper_mask, "solution/visualization/gripper_mask.png")

    # Ensure gripper fits within part mask bounds
    if x < 0 or y < 0 or x + g_w > p_w or y + g_h > p_h:
        print("TRUE")
        return True  # Treat out-of-bounds as a collision

    print("not true")
    # Place the rotated gripper into the same frame size as the part mask
    translated_gripper = np.zeros_like(part_mask)
    translated_gripper[y:y + g_h, x:x + g_w] = rotated_gripper

    # Check for collision
    collision = np.logical_and(part_mask, translated_gripper)
    return np.any(collision)


def objective_function(params, part_mask, gripper_mask, part_path, gripper_path):
    x, y, theta = params
    global i
    print("params:", params)
    if not math.isnan(params[0]):
        print("visualize")
        overlay_png_on_png(
            png_path=part_path,
            overlay_path=gripper_path,
            output_path='solution/visualization/combination_' + str(i) + '.png',
            x=x,
            y=y,
            angle=theta)
        i += 1
    else:
        print("param is none")
    if gripper_collision_check(part_mask, gripper_mask, int(x), int(y), theta):
        return float('inf')  # Penalize collisions
    # Minimize distance from part center (for balance)
    part_center = np.array(part_mask.shape) / 2
    gripper_center = np.array([x, y])
    return np.linalg.norm(part_center - gripper_center)


def find_best_gripper_position(part_path, gripper_path):
    part_mask = generate_clean_mask(part_path)
    gripper_mask = load_and_preprocess_part(gripper_path)
    print("gripper_path:", gripper_path)

    # Initial guess
    initial_guess = [(part_mask.shape[1] // 2) - (gripper_mask.shape[1] // 2),
                     (part_mask.shape[0] // 2) - (gripper_mask.shape[0] // 2), 0]
    print(initial_guess)
    overlay_png_on_png(
        png_path=part_path,
        overlay_path=gripper_path,
        output_path='solution/visualization/initial_guess.png',
        x=initial_guess[0],
        y=initial_guess[1],
        angle=initial_guess[2])

    # Bounds
    bounds = [(0, part_mask.shape[1]), (0, part_mask.shape[0]), (0, 360)]

    print("params:", initial_guess)
    print("part_mask:", part_mask)
    print("gripper_mask:", gripper_mask)
    print(part_mask.shape)
    print(gripper_mask.shape)

    # Optimization
    result = minimize(objective_function, initial_guess, args=(part_mask, gripper_mask, part_path, gripper_path), bounds=bounds)
    return result.x  # Optimal (x, y, θ)

def compute_amazing_solution(
    part_image_path: Path, gripper_image_path: Path
) -> tuple[float, float, float]:
    """Compute the solution for the given part and gripper images.

    :param part_image_path: Path to the part image
    :param gripper_image_path: Path to the gripper image
    :return: The x, y and angle of the gripper
    """

    return 100.1, 95, 91.2


def main():
    """The main function of your solution.

    Feel free to change it, as long as it maintains the same interface.
    """

    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input)

    # compute the solution for each row
    results = []
    for _, row in track(
        input_df.iterrows(),
        description="Computing the solutions for each row",
        total=len(input_df),
    ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = find_best_gripper_position(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()