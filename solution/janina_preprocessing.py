import cv2
import numpy as np
from visualizer import visualize_matrix
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
    holes_mask = holes_mask//255
    visualize_matrix(mask, "solution/visualization/mask.png")
    print(holes_mask)
    #visualize_matrix(holes_mask, "solution/visualization/holes_mask.png")

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
    return gripper_positions, gripper_center, binary
