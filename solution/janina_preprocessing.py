import cv2
import numpy as np
from visualizer import visualize_matrix
from scipy.ndimage import convolve
from sklearn.cluster import KMeans
from PIL import Image


def k_means_clustering(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

    # Reshape image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2)  # Use 2 clusters (foreground and background)
    kmeans.fit(pixels)
    labels = kmeans.predict(pixels)

    # Reconstruct the segmented image (where each pixel belongs to one of the clusters)
    segmented_image = labels.reshape(image.shape[0], image.shape[1])

    # Display or process the segmented image
    # You can convert it to a binary mask by setting a threshold if needed
    binary_mask = segmented_image.astype(np.uint8)  # 0 for background,
    visualize_matrix(binary_mask, "solution/visualization/k_means_binary_mask.png")


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

    """
    import cv2
    import numpy as np

    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Detect edges
    edges = cv2.Canny(gray, 100, 200)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)

    # Step 4: Close gaps in the edges using morphological operations (Dilation + Erosion)
    kernel = np.ones((7, 7), np.uint8)  # Larger kernel for better edge closure
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Closed Edges', closed_edges)
    cv2.waitKey(0)

    # Step 5: Fill regions enclosed by edges
    h, w = closed_edges.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Padding for flood fill
    flood_filled = closed_edges.copy()
    cv2.floodFill(flood_filled, mask, (0, 0), 255)  # Flood fill from the corner
    cv2.imshow('Flood Filled', flood_filled)
    cv2.waitKey(0)

    # Step 6: Invert the flood-filled image
    inverted_flood = cv2.bitwise_not(flood_filled)
    cv2.imshow('Inverted Flood', inverted_flood)
    cv2.waitKey(0)

    # Step 7: Combine flood-filled with closed edges to retain metal part
    filled_regions = cv2.bitwise_and(closed_edges, inverted_flood)
    cv2.imshow('Filled Regions', filled_regions)
    cv2.waitKey(0)

    # Step 8: Identify connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_regions, connectivity=8)

    # Step 9: Find the largest connected component (excluding background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Exclude background
    largest_component = (labels == largest_label).astype(np.uint8) * 255
    cv2.imshow('Largest Component', largest_component)

    # Step 10: Convert to binary matrix
    binary_matrix = (largest_component > 0).astype(int)

    # Display the final result
    cv2.imshow('Final Binary Mask', largest_component)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
    inv_holes_mask = mask//255
    visualize_matrix(mask, "solution/visualization/mask.png")
    print(holes_mask)
    #visualize_matrix(holes_mask, "solution/visualization/holes_mask.png")

    return largest_contour, holes_mask, inv_holes_mask


def extract_gripper_positions(gripper_image_path):
    # Preprocess the gripper image
    image = cv2.imread(gripper_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    visualize_matrix(binary, "solution/visualization/gripper_binary.png")

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

    label_image = np.array(Image.open(gripper_image_path).convert("L"))
    binary = (label_image != 0).astype(int)

    # TEST: make gripper radius bigger
    # _________________________________________________________________________________
    # Define a kernel to set the neighborhood
    size = 11
    radius = size // 2
    kernel = np.zeros((size, size))  # 3x3 kernel will set all neighbors to 1
    for i in range(size):
        for j in range(size):
            # Calculate Euclidean distance from (i, j) to center (cx, cy)
            if np.sqrt((i - radius)**2 + (j - radius)**2) <= radius:
                kernel[i, j] = 1
    #print(kernel)

    # pad binary image
    binary = np.pad(binary, pad_width=radius, mode='constant', constant_values=0)

    # Apply convolution
    binary = convolve(binary, kernel, mode='constant', cval=0)

    # Convert result to binary (0 or 1)
    binary = (binary > 0).astype(int)
    # _________________________________________________________________________________


    visualize_matrix(binary, "solution/visualization/gripper_binary.png")
    return gripper_positions, gripper_center, binary
