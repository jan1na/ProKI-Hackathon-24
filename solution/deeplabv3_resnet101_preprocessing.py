import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from PIL import Image
import torch.nn as nn

kernel_size = 3
model_path = "solution/deeplabv3_binary_segmentation_42_i_correct.pth"


def get_refined_mask(output):
    moved = output.squeeze() - output.squeeze().min()
    output_predictions = (moved > 0.85 * moved.max()).float()

    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=output.device)

    eroded_mask = F.conv2d(output_predictions.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2)
    eroded_mask = (eroded_mask > 1).float().squeeze()

    return eroded_mask


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


# Function to make predictions
def predict(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)['out']
        refined_output = get_refined_mask(output)

    return np.asarray(refined_output, dtype=np.int32)
