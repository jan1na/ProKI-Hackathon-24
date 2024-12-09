from PIL import Image, ImageOps, ImageDraw
import cairosvg
import io
import math
import matplotlib.pyplot as plt

def overlay_png_on_png(png_path, overlay_path, output_path, x, y, angle):
    """
    Combines a PNG onto another PNG at a specified position and angle.

    :param png_path: Path to the input PNG file.
    :param overlay_path: Path to the input overlay PNG file.
    :param output_path: Path to save the resulting image.
    :param x: X-coordinate for the top-left corner of the overlay PNG on the base PNG.
    :param y: Y-coordinate for the top-left corner of the overlay PNG on the base PNG.
    :param angle: Rotation angle for the overlay PNG in degrees.
    """

    print("x:", x)
    print("y:", y)
    # Load the PNG image
    png_image = Image.open(png_path).convert("RGBA")
    print(png_image.size)

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
    canvas.paste(svg_image, (int(x) - svg_image.size[0]//2, int(y) - svg_image.size[1]//2), svg_image)

    # Composite the canvas and PNG
    result_image = Image.alpha_composite(png_image, canvas)
    # Draw a small circle at the specified position
    draw = ImageDraw.Draw(result_image)
    draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 0, 0, 255))  # Red circle with radius 5
    draw.ellipse((197 - 2, 200 - 2, 197 + 2, 200 + 2), fill=(0, 255, 0, 255))  # Red circle with radius 5
    print(result_image.size)

    # Save the result
    result_image.save(output_path, "PNG")


def visualize_matrix(matrix, path):
    plt.imshow(matrix)
    # plt.colorbar()
    plt.savefig(path)
    plt.show()
