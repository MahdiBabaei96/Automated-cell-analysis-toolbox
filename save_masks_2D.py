import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from segmentation_model_architecture import vgg16_unet
from utils import *
from metrics import *
from tifffile import imread, imsave
from PIL import Image

# Function to load images from a directory (specific for .tif files)
def load_tif_images_from_folder(folder, target_size=(1024, 512)):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".tif"):
            img_path = os.path.join(folder, filename)
            img = imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)  # Resize image
                if len(img.shape) == 2:  # If grayscale, convert to 3-channel
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                images.append(img)
                filenames.append(filename)
    return np.array(images), filenames

# Load your model
# model_path = 'Models/819-Precision-New-1024_vgg16_unet_w0.hdf5'
#######################
# THIS MODEL IS ROTATED SO I USED ROTATION ON INPUT AND UN_ROTATION ON PREDICTED OUTPUT MASK
model_path = "results/New-512-1024.hdf5"
input_shape = (1024, 512, 3)  # Ensure this matches the shape used during training
model = vgg16_unet(input_shape)
model.summary()
model.load_weights(model_path)

# Folder paths
image_folder = 'Dataset/frames'
output_folder = 'Dataset/frames_output_folder'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load images
images, filenames = load_tif_images_from_folder(image_folder)

if len(images) == 0:
    print("No .tif images found in the folder.")
else:
    print(f"Processing {len(images)} images...")

    # Normalize images
    # images = images / 255.0  

    # Predict masks
    predictions = model.predict(images, batch_size=1)  # Reduce batch size


    # Save masks
    for pred, filename in zip(predictions, filenames):
        mask_path = os.path.join(output_folder, filename)
        imsave(mask_path, cv2.rotate((pred * 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE))  # Convert mask to 8-bit and save

    print(f"Saved {len(predictions)} predicted masks in '{output_folder}'")
