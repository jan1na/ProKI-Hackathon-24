import os
import matplotlib.pyplot as plt
from PIL import Image

from deeplabv3_resnet101_preprocessing import predict

test_data_dir = r"data/train_data/images"


# Process all images in the directory
for image_name in os.listdir(test_data_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
        image_path = os.path.join(test_data_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        # Make predictions
        output_predictions = predict(image_path)

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
