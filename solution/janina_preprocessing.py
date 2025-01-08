import numpy as np
from scipy.ndimage import convolve
from PIL import Image


def get_gripper_binary(gripper_image_path: str) -> np.ndarray:
    """
    Get the binary mask of the gripper.

    :param gripper_image_path: Path to the gripper image.
    :return: binary mask of the gripper.
    """
    label_image = np.array(Image.open(gripper_image_path).convert("L"))
    binary = (label_image != 0).astype(int)

    # Define a kernel to dilate the binary mask to have a safety distance
    size = 11
    radius = size // 2
    kernel = np.zeros((size, size))  # 3x3 kernel will set all neighbors to 1
    for i in range(size):
        for j in range(size):
            # Calculate Euclidean distance from (i, j) to center (cx, cy)
            if np.sqrt((i - radius)**2 + (j - radius)**2) <= radius:
                kernel[i, j] = 1

    # pad binary image so the bigger gripper fits into the matrix
    binary = np.pad(binary, pad_width=radius, mode='constant', constant_values=0)

    # Apply convolution
    binary = convolve(binary, kernel, mode='constant', cval=0)

    # Convert result to binary (0 or 1)
    binary = (binary > 0).astype(int)

    return binary
