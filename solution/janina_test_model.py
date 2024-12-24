import json
from segmentation_models import Unet
from PIL import Image
from visualizer import visualize_matrix
from sklearn.cluster import KMeans

# Step 1: Load config and metadata
with open('solution/saved_model/config.json', 'r') as f:
    config = json.load(f)

with open('solution/saved_model/metadata.json', 'r') as f:
    metadata = json.load(f)

# Extract parameters from config or set defaults
BACKBONE = config.get('backbone_name', 'resnet34')
INPUT_SHAPE = config.get('input_shape', (None, None, 3))
CLASSES = config.get('classes', 1)
ACTIVATION = config.get('activation', 'sigmoid')

print(INPUT_SHAPE)

# Step 2: Rebuild the U-Net model
model = Unet(
    backbone_name=BACKBONE,
    encoder_weights='imagenet',  # Use the same encoder weights as in training
    classes=CLASSES,
    activation=ACTIVATION,
    input_shape=INPUT_SHAPE
)

# Step 3: Load the weights
model.load_weights('solution/saved_model/model.weights.h5')

print("Model loaded successfully!")

import os
import numpy as np
import tensorflow as tf

def pad_to_divisible(image, divisor=32):
    h, w = image.shape[:2]
    new_h = (h + divisor - 1) // divisor * divisor
    new_w = (w + divisor - 1) // divisor * divisor

    new_h = 1024
    new_w = 1024

    padded_image = np.zeros((new_h, new_w, 3), dtype=np.float32)
    padded_image[:h, :w, :] = image

    return padded_image


# Define a function to preprocess images without resizing
def preprocess_input(image_path):
    """
    Preprocesses a single image by:
    - Loading it as an array.
    - Normalizing pixel values to [0, 1].
    """
    # Load the image
    image = tf.keras.utils.load_img(image_path)  # Keep the original size


    # Convert to numpy array and normalize
    image_ = tf.keras.utils.img_to_array(image) / 255.0

    padded_image = pad_to_divisible(image_)

    # Load the PNG image
    original_image = Image.open(image_path)

    # Convert the image to RGB if it's not already in that mode
    rgb_image = original_image.convert("RGB")

    # Convert the image to a numpy array
    rgb_array = np.array(rgb_image)

    # Add batch dimension
    return np.expand_dims(padded_image, axis=0), rgb_array


# Define a function to load and preprocess all "mask" images
def load_and_preprocess_images(folder_path):
    """
    Traverses a folder structure to find and preprocess all images containing "mask" in their filename.
    """
    image_list = []
    preprocessed_image_list = []
    # Walk through the folder structure
    for root, _, files in os.walk(folder_path):
        for file in files:
            if "image" in file.lower():  # Case-insensitive match
                full_path = os.path.join(root, file)
                preprocessed_image, image = preprocess_input(full_path)
                preprocessed_image_list.append(preprocessed_image)
                image_list.append(image)

    # Concatenate all preprocessed images into a batch
    if image_list:
        return preprocessed_image_list, image_list  # Stack into a single numpy array
    else:
        print("No images containing 'mask' found.")
        return None


# Usage example
folder_path = 'data/train_data/images'  # Replace with your folder path
preprocessed_images, images = load_and_preprocess_images(folder_path)

if preprocessed_images is not None:
    print(f"Loaded and preprocessed {len(preprocessed_images)} images.")
else:
    print("No images to process.")

from PIL import Image
import numpy as np

def crop_to_original(padded_output, original_shape):
    h, w = original_shape[:2]
    return padded_output[:h, :w]

def k_means_clustering(binary_mask):
    pixels = binary_mask.reshape((-1, 1))
    print("pixels", pixels.shape)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2)  # Use 2 clusters (foreground and background)
    kmeans.fit(pixels)
    labels = kmeans.predict(pixels)

    # Reconstruct the segmented image (where each pixel belongs to one of the clusters)
    segmented_image = labels.reshape(binary_mask.shape[0], binary_mask.shape[1])

    # Display or process the segmented image
    # You can convert it to a binary mask by setting a threshold if needed
    binary_mask = segmented_image.astype(np.uint8)  # 0 for background,
    return binary_mask


for i, (preprocessed_image, image) in enumerate(zip(preprocessed_images, images)):
    # Predict the binary mask using the model
    binary_mask = model.predict(preprocessed_image)
    print("binary_mask", binary_mask.shape)

    # Crop the binary mask to match the original image size
    binary_mask = crop_to_original(binary_mask[0], image.shape[:2])  # Use only height and width
    print(binary_mask.shape)
    threshold = 0.4

    visualize_matrix(binary_mask, f"solution/visualization/mask_{i}.png")

    mask_k_means = k_means_clustering(binary_mask)
    visualize_matrix(mask_k_means, f"solution/visualization/mask_k_means_{i}.png")

    binary_mask_normalized = (binary_mask - np.min(binary_mask)) / (np.max(binary_mask) - np.min(binary_mask))
    binary_mask = (binary_mask_normalized >= threshold).astype(np.uint8)

    visualize_matrix(binary_mask, f"solution/visualization/binary_mask_{i}.png")

    # Ensure binary_mask is 2D
    if binary_mask.ndim != 2:
        binary_mask = binary_mask.squeeze()  # Remove unnecessary dimensions

    # Ensure the image is in uint8 format
    image_array = image.astype('uint8')

    # Convert the numpy array to a PIL Image
    rgb_pil = Image.fromarray(image_array, mode="RGB")

    # Convert the binary mask to an alpha channel (0 for transparent, 255 for opaque)
    alpha_channel = (binary_mask * 255).astype('uint8')

    # Convert the RGB image to RGBA by adding the alpha channel
    rgba_image = Image.new("RGBA", rgb_pil.size)
    rgba_image.paste(rgb_pil, (0, 0), mask=None)
    rgba_image.putalpha(Image.fromarray(alpha_channel))

    # Save the RGBA image as a PNG file
    rgba_image.save(f"solution/visualization/test_{i}.png")

