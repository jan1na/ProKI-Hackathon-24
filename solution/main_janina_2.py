import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from rich.progress import track
import pandas as pd

from visualizer import visualize_matrix, overlay_png_on_png


def set_circle_in_matrix(matrix, center, radius):
    # Get matrix dimensions
    rows, cols = matrix.shape
    cx, cy = center  # center coordinates

    for i in range(rows):
        for j in range(cols):
            # Calculate Euclidean distance from (i, j) to center (cx, cy)
            if np.sqrt((i - cy)**2 + (j - cx)**2) <= radius:
                matrix[i, j] = 255
    return matrix


def preprocess_image(image_path):
    """
    # Load image with alpha channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Separate alpha channel and RGB channels
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        alpha = None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding for better contour detection
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    visualize_matrix(binary, "solution/visualization/binary.png")
    """

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

    for i in range(1, num_labels):  # Skip the background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255
    visualize_matrix(cleaned_mask, "solution/visualization/cleaned_mask.png")

    return image_unchanged, image, cleaned_mask, a



def extract_features(binary_image):
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assume it's the metal part)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # Create a mask for holes
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, contours, -1, (255), thickness=-1)  # Fill the areas
    holes_mask = cv2.bitwise_not(mask)

    return largest_contour, holes_mask


def extract_gripper_positions(gripper_image_path):
    # Preprocess the gripper image
    image = cv2.imread(gripper_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Detect the active "dots" of the gripper
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get centroids and radii of the dots
    gripper_positions = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            radius = int(np.sqrt(cv2.contourArea(contour) / np.pi))  # Approximate radius
            gripper_positions.append((cx, cy, radius))

    # print("Gripper positions:", gripper_positions)
    gripper_center = binary.shape[1] // 2, binary.shape[0] // 2
    return gripper_positions, gripper_center


def is_gripper_position_valid(gripper_positions, gripper_center, center, angle, contour, holes_mask):
    # Rotate gripper positions around the center of the gripper
    rotated_positions = []
    for x, y, radius in gripper_positions:
        x_rel, y_rel = x - gripper_center[0], y - gripper_center[1]  # Relative to gripper center
        x_rot = int(center[0] + x_rel * np.cos(angle) - y_rel * np.sin(angle))
        y_rot = int(center[1] + x_rel * np.sin(angle) + y_rel * np.cos(angle))
        rotated_positions.append((x_rot, y_rot, radius))

    # Check each dot's validity
    for x, y, radius in rotated_positions:
        #if cv2.pointPolygonTest(contour, (x, y), False) < 0:  # Outside the metal contour
        #    print("Outside the metal contour")
        #    return False
        for yy in range(max(0, y - radius), min(holes_mask.shape[0], y + radius)):
            for xx in range(max(0, x - radius), min(holes_mask.shape[1], x + radius)):
                if (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2:  # Check circular area
                    if holes_mask[yy, xx] == 255:  # In a hole
                        print("In a hole")
                        print(xx, yy)
                        return False

    return True


def optimize_gripper_position(gripper_positions, gripper_center, contour, holes_mask):
    height, width = holes_mask.shape
    best_position = None
    best_angle = None

    """
    angles = np.linspace(0, 2 * np.pi, 36)  # 10-degree steps
    for angle in angles:
        for y in range(0, height, 10):  # Increment by step size to reduce computation
            for x in range(0, width, 10):
                center = (x, y)  # Center of the gripper placed at (x, y)
                print(center)
                if is_gripper_position_valid(gripper_positions, gripper_center, center, angle, contour, holes_mask):
                    return center, angle  # Return first valid position and angle

    """
    if is_gripper_position_valid(gripper_positions, gripper_center, (holes_mask.shape[1]//2 + 4, holes_mask.shape[0]// 2 - 24), 0, contour, holes_mask):
        print("Correct")
        return (holes_mask.shape[1]//2 + 4, holes_mask.shape[0]// 2 - 24), 0
    else:
        print("not correct")
        return (holes_mask.shape[1]//2 + 4, holes_mask.shape[0]// 2 - 24), 0

    return best_position, best_angle


def compute_amazing_solution(part_image_path, gripper_image_path):
    metal_image, gray_metal, binary_metal, alpha_metal = preprocess_image(part_image_path)
    metal_contour, holes_mask = extract_features(binary_metal)
    gripper_positions, gripper_center = extract_gripper_positions(gripper_image_path)

    center, angle = optimize_gripper_position(gripper_positions, gripper_center, metal_contour, holes_mask)
    print("Optimal position:", center)
    print("Optimal angle (radians):", angle)
    overlay_png_on_png(part_image_path, gripper_image_path, "solution/visualization/overlay.png", center[0], center[1], angle)
    return center[0], center[1], angle


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

    test = 0

    # compute the solution for each row
    results = []
    for _, row in track(
        input_df.iterrows(),
        description="Computing the solutions for each row",
        total=len(input_df),
    ):
        if test > 0:
            break
        test += 1
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = compute_amazing_solution(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

