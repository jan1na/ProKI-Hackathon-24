import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torch.nn.functional as F_
import torchvision
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from visualizer import visualize_matrix

import cv2
import numpy as np


def get_mask(image):
    # Ensure the image is in grayscale and of type uint8
    if len(image.shape) == 3:  # If the image is colored (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    image = np.uint8(image)  # Convert to 8-bit unsigned integer if it's not already

    # Step 1: Apply Gaussian Blur to smooth edges and reduce noise
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    visualize_matrix(blurred, f"solution/visualization/{image_name}_blurred.png")
    blurred = image

    # Step 2: Apply Canny Edge Detection to find edges
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    visualize_matrix(edges, f"solution/visualization/{image_name}_edges.png")

    # Step 3: Apply morphological operations to refine edges
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

    # Step 4: Find contours to extract the largest area
    contours, _ = cv2.findContours(eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Create a mask and fill the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Step 6: Ensure a coherent region with value 1
    final_mask = mask / 255  # Convert mask to 0s and 1s

    return final_mask


def get_refined_mask(output, kernel_size=3, threshold=0.9999):
    # Step 1: Apply sigmoid and threshold to create binary mask
    output_predictions = (torch.sigmoid(output.squeeze()) > threshold).float()


    # Assuming `output` is the raw output of your model
    output_sigmoid = torch.sigmoid(output.squeeze())

    # Scale to range [0, 255]
    scaled_output = (output_sigmoid * 1000).clamp(0, 1000).to(torch.uint8)
    output_predictions = (scaled_output > 950).float()

    moved = output.squeeze() - output.squeeze().min()
    output_predictions = (moved > 0.6 * moved.max()).float()



    # Step 2: Apply erosion using a 3x3 square kernel (or any custom size)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=output.device)
    eroded_mask = F_.conv2d(output_predictions.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2)

    # Step 3: Apply a dynamic threshold based on kernel sum for erosion
    eroded_mask = (eroded_mask == kernel.sum()).float().squeeze()

    # Optionally apply dilation after erosion to refine the result
    dilated_mask = F_.conv2d(eroded_mask.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2)
    refined_mask = (dilated_mask > 1).float().squeeze()
    visualize_matrix(refined_mask.cpu().numpy(), f"solution/visualization/{image_name}_dilated_mask.png")

    return eroded_mask


# Path to your model and test data
model_path = "solution/deeplabv3_binary_segmentation.pth"
test_data_dir = r"data/test_data"

# Load the pre-trained model and redefine the classifier for binary segmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1))  # Binary segmentation
model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1))  # Auxiliary classifier
model = model.to(device)

# Load the saved weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the transformations for test-time
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor without resizing
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Define a 3x3 kernel for erosion
kernel = torch.ones(1, 1, 3, 3).float()  # 1x1x3x3 kernel for erosion


# Function to restore original size
def restore_original_size(pred, original_size):
    pred = F.to_pil_image(pred.squeeze(0).cpu())  # Convert tensor to PIL Image
    pred = pred.crop((0, 0, original_size[0], original_size[1]))  # Crop to original size
    return pred


# Function to pad images to 1024x1024
def pad_image(image):
    original_size = image.size  # (width, height)
    pad_width = (1024 - original_size[0]) if original_size[0] < 1024 else 0
    pad_height = (1024 - original_size[1]) if original_size[1] < 1024 else 0
    padding = (0, 0, pad_width, pad_height)  # (left, top, right, bottom)
    padded_image = F.pad(image, padding, fill=0)
    return padded_image, original_size


# Function to make predictions
def predict(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    padded_image, original_size = pad_image(image)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)['out']
        visualize_matrix(output.squeeze().cpu().numpy(), f"solution/visualization/{image_name}_output.png")

        mask = get_mask(np.array(output.squeeze().cpu().numpy()))
        visualize_matrix(mask, f"solution/visualization/{image_name}_mask.png")




        #print(torch.sigmoid(output.squeeze()))
        output_predictions = (torch.sigmoid(output.squeeze()) > 0.9).float()


        refined_output = get_refined_mask(output, kernel_size=3, threshold=0.9)
        visualize_matrix(refined_output.cpu().numpy(), f"solution/visualization/{image_name}_refined_output.png")
        # output_predictions = output.squeeze()
        # Apply erosion
        eroded_mask = F_.conv2d(output_predictions.unsqueeze(0).unsqueeze(0), kernel, padding=1)
        eroded_mask = (eroded_mask == 9).float().squeeze()
        visualize_matrix(eroded_mask.cpu().numpy().squeeze(), f"solution/visualization/{image_name}_eroded_mask.png")

    # Restore the original size
    #restored_output = restore_original_size(output_predictions, original_size)
    #visualize_matrix(restored_output, f"solution/visualization/{image_name}_restored_output.png")
    return image, refined_output


# Process all images in the directory
for image_name in os.listdir(test_data_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
        image_path = os.path.join(test_data_dir, image_name)

        # Make predictions
        image, output_predictions = predict(model, image_path)

        # Visualize the result
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(output_predictions, cmap="gray")
        plt.title("Predicted Segmentation")
        plt.savefig(f"solution/visualization/{image_name}_prediction.png")
        plt.close()
