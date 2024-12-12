import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from argparse import ArgumentParser
from rich.progress import track
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from visualizer import visualize_matrix, overlay_png_on_png


def set_circle_in_matrix(matrix, center, radius):
    y, x = np.ogrid[:matrix.shape[0], :matrix.shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    matrix[dist_from_center <= radius] = 255
    return matrix


def create_gripper_kernel(radius):
    y, x = np.ogrid[:2*radius+1, :2*radius+1]
    center = radius
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    kernel = mask.astype(float)
    return kernel


def preprocess_image(image_path):
    image_unchanged = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_unchanged[:, :, 3] if image_unchanged.shape[2] == 4 else None

    gray_image = cv2.cvtColor(image_unchanged, cv2.COLOR_BGRA2GRAY if alpha_channel is not None else cv2.COLOR_BGR2GRAY)
    gray_image[alpha_channel == 0] = 255 if alpha_channel is not None else gray_image

    smoothed_image = gaussian_filter(gray_image, sigma=1)
    _, binary_mask = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_area = 10
    large_components = (stats[:, cv2.CC_STAT_AREA] >= min_area)
    cleaned_mask = np.where(np.isin(labels, np.flatnonzero(large_components)), 255, 0).astype(np.uint8)

    return image_unchanged, gray_image, cleaned_mask


def extract_features(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    holes_mask = cv2.bitwise_not(mask) // 255
    return largest_contour, holes_mask


def extract_gripper_positions(gripper_image_path):
    image = cv2.imread(gripper_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gripper_positions = [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00']), int(np.sqrt(cv2.contourArea(c) / np.pi)))
                         for c in contours if (M := cv2.moments(c))['m00'] != 0]

    gripper_center = (binary.shape[1] // 2, binary.shape[0] // 2)
    return gripper_positions, gripper_center


def is_gripper_position_valid(rotated_positions, holes_mask):
    for x, y, radius in rotated_positions:
        y_start, y_end = max(0, y - radius), min(holes_mask.shape[0], y + radius)
        x_start, x_end = max(0, x - radius), min(holes_mask.shape[1], x + radius)

        sub_region = holes_mask[y_start:y_end, x_start:x_end]
        circle = (np.ogrid[:2*radius, :2*radius][0] - radius)**2 + (np.ogrid[:2*radius, :2*radius][1] - radius)**2 <= radius**2
        if np.any(circle & sub_region):
            return False
    return True


def optimize_gripper_position(gripper_positions, gripper_center, holes_mask):
    kernel = create_gripper_kernel(gripper_positions[0][2])
    convolved = convolve2d(holes_mask, kernel, mode='same', boundary='fill', fillvalue=0)
    visualize_matrix(convolved, "solution/visualization/convolved.png")

    valid_positions = np.argwhere(convolved == kernel.sum())
    best_position, best_angle = None, None

    for y, x in valid_positions:
        rotated_positions = [(x + int((cx - gripper_center[0]) * np.cos(0) - (cy - gripper_center[1]) * np.sin(0)),
                              y + int((cx - gripper_center[0]) * np.sin(0) + (cy - gripper_center[1]) * np.cos(0)),
                              radius) for cx, cy, radius in gripper_positions]

        if is_gripper_position_valid(rotated_positions, holes_mask):
            return (x, y), 0  # Return first valid position with zero rotation

    return best_position, best_angle


def compute_amazing_solution(part_image_path, gripper_image_path):
    metal_image, gray_metal, binary_metal = preprocess_image(part_image_path)
    metal_contour, holes_mask = extract_features(binary_metal)
    gripper_positions, gripper_center = extract_gripper_positions(gripper_image_path)

    center, angle = optimize_gripper_position(gripper_positions, gripper_center, holes_mask)
    overlay_png_on_png(part_image_path, gripper_image_path, "solution/visualization/overlay.png", center[0], center[1], angle)
    return center[0], center[1], angle


def main():
    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    results = []

    for _, row in track(input_df.iterrows(), description="Processing", total=len(input_df)):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"

        x, y, angle = compute_amazing_solution(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
