import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def overlay_png_on_png(png_path: str, overlay_path: str, output_path: str, x: int, y: int, angle: float):
    """
    Combines a PNG onto another PNG at a specified position and angle.

    :param png_path: Path to the background PNG image.
    :param overlay_path: Path to the overlay PNG image.
    :param output_path: Path to save the result.
    :param x: Horizontal position of the overlay image.
    :param y: Vertical position of the overlay image.
    :param angle: Rotation angle of the overlay image in degrees.
    """
    # Load the PNG image
    png_image = Image.open(png_path).convert("RGBA")

    # Open the overlay PNG as a PIL image
    svg_image = Image.open(overlay_path).convert("RGBA")

    # Enhance the dots in the SVG by isolating non-transparent pixels and brightening them
    pixels = svg_image.load()
    for j in range(svg_image.size[1]):
        for i in range(svg_image.size[0]):
            r, g, b, a = pixels[i, j]
            if a > 0:  # Enhance only visible (non-transparent) dots
                r = min(255, int(r * 1.5))
                g = min(255, int(g * 1.5))
                b = min(255, int(b * 1.5))
                pixels[i, j] = (r, g, b, 255)

    # Rotate the SVG image
    if angle != 0:
        svg_image = svg_image.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Ensure the canvas is large enough for placement
    canvas = Image.new("RGBA", png_image.size, (255, 255, 255, 0))

    # Paste the rotated SVG image onto the canvas at the specified position
    canvas.paste(svg_image, (int(x) - svg_image.size[0] // 2, int(y) - svg_image.size[1] // 2), svg_image)

    # Composite the canvas and PNG
    result_image = Image.alpha_composite(png_image, canvas)
    # Draw a small circle at the middel of the gripper
    draw = ImageDraw.Draw(result_image)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0, 255))  # Red circle with radius 5

    # Save the result
    result_image.save(output_path, "PNG")


def visualize_matrix(matrix: np.ndarray, path: str):
    """
    Visualizes a matrix as a grayscale image.

    :param matrix: Input matrix to visualize.
    :param path: Path to save the visualization.
    """
    # Visualize the matrix as a grayscale image
    plt.imshow(matrix, cmap="gray")
    plt.savefig(path)
    plt.close()


def visualize_gripper_on_mask(mask_matrix: np.ndarray, gripper_path: str, x: int, y: int, angle: float, path: str):
    """
    Visualizes a gripper on a mask matrix at a specified position and angle.

    :param mask_matrix: Input mask matrix.
    :param gripper_path: Path to the gripper image.
    :param x: Horizontal position of the gripper.
    :param y: Vertical position of the gripper.
    :param angle: Rotation angle of the gripper in degrees.
    :param path: Path to save the visualization.
    """
    # Load the gripper image
    gripper_image = Image.open(gripper_path).convert("RGBA")

    # Enhance the dots in the gripper image
    pixels = gripper_image.load()
    for j in range(gripper_image.size[1]):
        for i in range(gripper_image.size[0]):
            r, g, b, a = pixels[i, j]
            if a > 0:  # Enhance only visible (non-transparent) dots
                r = min(255, int(r * 1.5))
                g = min(255, int(g * 1.5))
                b = min(255, int(b * 1.5))
                pixels[i, j] = (r, g, b, 255)

    # Rotate the gripper image
    rotated_gripper = gripper_image.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Create a white canvas
    canvas = Image.new("RGBA", mask_matrix.shape[::-1], (255, 255, 255, 0))

    # Paste the rotated gripper image onto the canvas at the specified position
    canvas.paste(rotated_gripper, (int(x) - rotated_gripper.size[0] // 2, int(y) - rotated_gripper.size[1] // 2),
                 rotated_gripper)

    # Convert mask_matrix to a PIL image
    mask_image = Image.fromarray(np.uint8(mask_matrix * 255)).convert("RGBA")

    # Composite the canvas and the mask
    result_image = Image.alpha_composite(mask_image, canvas)

    # Draw a small circle at the middle of the gripper
    draw = ImageDraw.Draw(result_image)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0, 255))  # Red circle with radius 5

    # Save the result
    result_image.save(path, "PNG")
