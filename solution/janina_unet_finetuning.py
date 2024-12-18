import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import segmentation_models as sm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Paths
IMAGE_PATH = './data/train_data/images/'
MASK_PATH = './data/train_data/masks/'


# Load Data
def load_data(image_path, mask_path):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_path))
    mask_files = sorted(os.listdir(mask_path))

    for img_file, mask_file in zip(image_files, mask_files):
        img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(image_path, img_file))) / 255.0  # Normalize images to [0, 1]
        mask = (tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(mask_path, mask_file), color_mode='grayscale')) > 0).astype(
            np.float32)

        images.append(img)
        masks.append(mask)

    return images, masks


# Padding Function
def pad_to_divisible(image, mask, divisor=32):
    h, w = image.shape[:2]
    new_h = (h + divisor - 1) // divisor * divisor
    new_w = (w + divisor - 1) // divisor * divisor

    #TODO: Fix padding
    new_h = 1024
    new_w = 1024

    padded_image = np.zeros((new_h, new_w, 3), dtype=np.float32)
    padded_image[:h, :w, :] = image

    padded_mask = np.zeros((new_h, new_w), dtype=np.float32)
    padded_mask[:h, :w] = mask.squeeze()

    return padded_image, padded_mask


# Load and pad the dataset
images, masks = load_data(IMAGE_PATH, MASK_PATH)
padded_images, padded_masks = [], []

for img, mask in zip(images, masks):
    p_img, p_mask = pad_to_divisible(img, mask)
    padded_images.append(p_img)
    padded_masks.append(p_mask)

padded_images = np.array(padded_images)
padded_masks = np.array(padded_masks)

# Albumentations Data Augmentation
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.Rotate(limit=45),
    A.RandomBrightnessContrast(),
    ToTensorV2()
])

val_transform = A.Compose([
    ToTensorV2()
])

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
    padded_images,
    padded_masks,
    validation_split=0.2,
    batch_size=16,
    epochs=50,
    callbacks=[early_stop, checkpoint]
)


# Postprocessing: Crop Predicted Masks Back to Original Size
def crop_to_original(padded_output, original_shape):
    h, w = original_shape[:2]
    return padded_output[:h, :w]


# Predict and Crop
predicted_masks = model.predict(padded_images)
original_masks = []

for pred, original_img in zip(predicted_masks, images):
    cropped_mask = crop_to_original(pred, original_img.shape)
    original_masks.append(cropped_mask)

original_masks = np.array(original_masks)


# Example Evaluation Metric: IoU
def iou_score(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    return intersection / union


# Evaluate IoU on the Test Set
test_iou = []
for true_mask, pred_mask in zip(masks, original_masks):
    test_iou.append(iou_score(true_mask > 0.5, pred_mask > 0.5))

print("Average IoU on Test Set:", np.mean(test_iou))

