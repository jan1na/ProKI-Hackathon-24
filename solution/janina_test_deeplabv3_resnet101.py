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


def get_refined_mask(output, kernel_size=3):

    moved = output.squeeze() - output.squeeze().min()
    output_predictions = (moved > 0.5 * moved.max()).float()

    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=output.device)
    eroded_mask = F_.conv2d(output_predictions.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2)

    eroded_mask = (eroded_mask == kernel.sum()).float().squeeze()

    # Scale output to [0, 1]
    moved = output.squeeze() - output.squeeze().min()
    moved = moved / moved.max()

    # Threshold to create a binary mask
    #output_predictions = (moved > 0.7).float()

    # Use a smaller kernel for erosion
    small_kernel = torch.ones(1, 1, 2, 2).float()
    dilated_mask = F_.conv2d(output_predictions.unsqueeze(0).unsqueeze(0), small_kernel, padding=1)
    dilated_mask = (dilated_mask > 0).float()

    # Combine erosion and dilation for refinement
    refined_mask = F_.conv2d(dilated_mask, small_kernel, padding=1)
    refined_mask = (refined_mask > 0).float().squeeze()

    return eroded_mask


# Path to your model and test data
model_path = "solution/deeplabv3_binary_segmentation_42_i_test_overfitting.pth"
test_data_dir = r"data/train_data/images"

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
transform = transforms.Compose([transforms.ToTensor(),  # Convert to Tensor without resizing
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
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)['out']
        visualize_matrix(output.squeeze().cpu().numpy(), f"solution/predictions/{image_name}_output.png")

        output_predictions = (torch.sigmoid(output.squeeze()) > 0.9).float()

        refined_output = get_refined_mask(output, kernel_size=3)
        # visualize_matrix(refined_output.cpu().numpy(), f"solution/visualization/{image_name}_refined_output.png")
        # Apply erosion
        eroded_mask = F_.conv2d(output_predictions.unsqueeze(0).unsqueeze(0), kernel, padding=1)
        eroded_mask = (eroded_mask == 9).float().squeeze()
        # visualize_matrix(eroded_mask.cpu().numpy().squeeze(), f"solution/visualization/{image_name}_eroded_mask.png")

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
        plt.savefig(f"solution/predictions/{image_name}_prediction.png")
        plt.close()
