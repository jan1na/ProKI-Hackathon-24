from PIL import Image, ImageOps
import cairosvg
import io
import math

def overlay_svg_on_png(png_path, svg_path, output_path, x, y, angle):
    """
    Combines an SVG onto a PNG at a specified position and angle.

    :param png_path: Path to the input PNG file.
    :param svg_path: Path to the input SVG file.
    :param output_path: Path to save the resulting image.
    :param x: X-coordinate for the top-left corner of the SVG on the PNG.
    :param y: Y-coordinate for the top-left corner of the SVG on the PNG.
    :param angle: Rotation angle for the SVG in degrees.
    """
    # Load the PNG image
    png_image = Image.open(png_path).convert("RGBA")

    # Convert the SVG to a rasterized image using cairosvg
    with open(svg_path, "rb") as svg_file:
        svg_data = svg_file.read()
        png_data = cairosvg.svg2png(bytestring=svg_data)

    # Open the rendered SVG as a PIL image
    svg_image = Image.open(io.BytesIO(png_data)).convert("RGBA")

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
    canvas.paste(svg_image, (int(x), int(y)), svg_image)

    # Composite the canvas and PNG
    result_image = Image.alpha_composite(png_image, canvas)

    # Save the result
    result_image.save(output_path, "PNG")

# Example usage
if __name__ == "__main__":
    overlay_svg_on_png(
        png_path="background.png",
        svg_path="overlay.svg",
        output_path="output.png",
        x=100,
        y=150,
        angle=45
    )