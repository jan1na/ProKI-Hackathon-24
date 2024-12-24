import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

import segmentation_models as sm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from visualizer import visualize_matrix

# Check available devices
print("Available devices:")
print(device_lib.list_local_devices())

# Set memory growth for GPUs
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Using GPU: {physical_devices[0].name}")
    except RuntimeError as e:
        print("Memory growth must be set at program startup. Exception:", e)
else:
    print("No GPU found. Using CPU.")

# Paths
IMAGE_PATH = './data/train_data/images/'
MASK_PATH = './data/train_data/masks/'

# Albumentations Data Augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast()
])

# Function to Apply Transformations
def apply_transforms(image, mask, transform):
    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']


num_transformations = 10

# Load Data with Transformations
def load_and_transform_data(image_path, mask_path, transform=None):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_path))
    mask_files = sorted(os.listdir(mask_path))

    for img_file, mask_file in zip(image_files, mask_files):
        img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(image_path, img_file))) / 255.0  # Normalize to [0, 1]
        mask = (tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(mask_path, mask_file), color_mode='grayscale')) > 0).astype(
            np.float32)

        if transform:
            for i in range(num_transformations):
                img_transformed, mask_transformed = apply_transforms(img.astype(np.float32), mask.astype(np.float32), transform)
                images.append(img_transformed)
                masks.append(mask_transformed)


        images.append(img)
        masks.append(mask)

    print("Loaded Images Shape:", len(images))
    return images, masks

# Padding Function
def pad_to_divisible(image, mask, divisor=32):
    h, w = image.shape[:2]
    new_h = (h + divisor - 1) // divisor * divisor
    new_w = (w + divisor - 1) // divisor * divisor

    new_h = 1024
    new_w = 1024

    padded_image = np.zeros((new_h, new_w, 3), dtype=np.float32)
    padded_image[:h, :w, :] = image
    visualize_matrix(padded_image, 'solution/visualization/padded_image.png')

    padded_mask = np.zeros((new_h, new_w), dtype=np.float32)
    padded_mask[:h, :w] = mask.squeeze()
    visualize_matrix(padded_mask, 'solution/visualization/padded_mask.png')

    return padded_image, padded_mask

# Load Training and Validation Data
train_images, train_masks = load_and_transform_data(IMAGE_PATH, MASK_PATH, train_transform)
val_images, val_masks = load_and_transform_data(IMAGE_PATH, MASK_PATH)

# Padding Training and Validation Data
padded_train_images, padded_train_masks = [], []
padded_val_images, padded_val_masks = [], []

for img, mask in zip(train_images, train_masks):
    p_img, p_mask = pad_to_divisible(img, mask)
    padded_train_images.append(p_img)
    padded_train_masks.append(p_mask)

for img, mask in zip(val_images, val_masks):
    p_img, p_mask = pad_to_divisible(img, mask)
    padded_val_images.append(p_img)
    padded_val_masks.append(p_mask)

padded_train_images = np.array(padded_train_images)
padded_train_masks = np.array(padded_train_masks)
padded_val_images = np.array(padded_val_images)
padded_val_masks = np.array(padded_val_masks)

print("Padded Training Images Shape:", padded_train_images.shape)
print("Padded Training Masks Shape:", padded_train_masks.shape)
print("Padded Validation Images Shape:", padded_val_images.shape)
print("Padded Validation Masks Shape:", padded_val_masks.shape)

# Pre-trained U-Net Model
BACKBONE = 'resnet34'
model = sm.Unet(
    backbone_name=BACKBONE,
    encoder_weights='imagenet',
    classes=1,
    activation='sigmoid',
    input_shape=(None, None, 3)  # Allow dynamic sizes
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_unet_model.keras', monitor='val_loss', save_best_only=True)

# Training
history = model.fit(
    padded_train_images,
    padded_train_masks,
    validation_data=(padded_val_images, padded_val_masks),
    batch_size=16,
    epochs=50,
    callbacks=[early_stop, checkpoint]
)

# Postprocessing: Crop Predicted Masks Back to Original Size
def crop_to_original(padded_output, original_shape):
    h, w = original_shape[:2]
    return padded_output[:h, :w]

# Predict and Visualize
predicted_masks = model.predict(padded_val_images)
original_masks = []

for i, (pred, original_img) in enumerate(zip(predicted_masks, val_images)):
    cropped_mask = crop_to_original(pred, original_img.shape)
    visualize_matrix(cropped_mask, f'visualization/cropped_mask_{i}.png')
    original_masks.append(cropped_mask)

# Example Evaluation Metric: IoU
def iou_score(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    return intersection / union

# Evaluate IoU on the Validation Set
val_iou = []
for true_mask, pred_mask in zip(val_masks, original_masks):
    val_iou.append(iou_score(true_mask > 0.5, pred_mask > 0.5))

print("Average IoU on Validation Set:", np.mean(val_iou))
