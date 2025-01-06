import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the device (use CPU as requested)
device = torch.device("cpu")

# Load the SAM model
sam_model_type = "vit_b"  # Model type (ViT-H in this case)
sam_checkpoint = 'solution/sam_vit_b_01ec64.pth'  # Path to your checkpoint

# Load the SAM model
model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device)

# Initialize the mask generator
mask_generator = SamAutomaticMaskGenerator(model)

# Set up input and output directories
input_dir = 'data/train_data/images'  # Directory containing your input images
output_dir = 'solution/predictions/'  # Directory to save the masks

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to save masks as images
def save_mask(mask, save_path):
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert to an image format (binary mask)
    mask_image.save(save_path)


# Process each image in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is an image (you can add more extensions if needed)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")

        # Generate the segmentation masks
        print("start")
        masks = mask_generator.generate(np.asarray(image))
        print("end")

        # Save the first mask (or process all masks if needed)
        if masks:
            first_mask = masks[0]['segmentation']  # Extract the segmentation mask

            # Define the path to save the mask
            mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
            mask_save_path = os.path.join(output_dir, mask_filename)

            # Save the mask
            save_mask(first_mask, mask_save_path)

            print(f"Saved mask for {filename} at {mask_save_path}")
        else:
            print(f"No mask generated for {filename}")
