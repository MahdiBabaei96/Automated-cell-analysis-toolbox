import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from segmentation_model_architecture import vgg16_unet
from utils import *
from metrics import *
from tifffile import imread


print("Current working directory:", os.getcwd())

# Load your model
model_path = 'New-1024_vgg16_unet_w0.hdf5'
input_shape = (1024, 1024, 3)  # Make sure this matches the shape used during training
model = vgg16_unet(input_shape)
# Load the weights
model.load_weights(model_path)

# Assuming you have a folder of images to predict
image_folder = 'D:/Tissue Engineering/Main/Data/B10'
images = load_images_from_folder(image_folder, color_mode=cv2.IMREAD_COLOR, target_size=(1024, 1024))
# Ensure the image dimensions are correct
width, height, depth = images[0].shape
print(f"B-scan image dimensions: {depth}x{height}x{width}")

all_features = []
for i in range(len(images)):
    # Extract the M-scan
    mscan = images[i]

    # Check if the M-scan is empty
    if mscan.size == 0:
        print(f"Empty M-scan at index {i}")
        continue
    
    print(f"Processing B-scan {i}")

    # Resize to model's input shape
    mscan_resized = cv2.resize(mscan, (1024, 1024))

    # Predict masks using the loaded model
    predicted_mask = model.predict(np.expand_dims(mscan_resized, axis=0))[0]
    display_prediction(mscan_resized.squeeze(), predicted_mask.squeeze())

    # Extract features from the predicted mask
    features, contours = extract_features_from_mask_findContours(predicted_mask)
    print(len(contours))
    all_features.append(features)

    for i in range(len(contours)):
        cX, cY, area, perimeter = features[i]
        print(calculate_surface_roughness_new(area, 1))
        # Fit ellipse requires at least 5 points
        if len(contours[i]) >= 5:
            ellipse = cv2.fitEllipse(contours[i])
            (x, y), (major_axis, minor_axis), angle = ellipse
            medium_radius = (major_axis + minor_axis) / 4
            smallest_radius = min(major_axis, minor_axis) / 2
            largest_radius = max(major_axis, minor_axis) / 2
            flatness = medium_radius / smallest_radius
            elongation = largest_radius / medium_radius
            print(flatness)
            print(elongation)
        print("****")
    

    # Optional: Display each M-scan's predicted mask and features
    plt.figure(figsize=(10, 5))
    plt.imshow(predicted_mask, cmap='gray')
    for contour in contours:
        cv2.drawContours(predicted_mask, [contour], -1, (0, 255, 0), 2)
    for feature in features:
        cX, cY, area, perimeter = feature
        plt.scatter(cX, cY, color='red')
        plt.text(cX, cY, f'A:{int(area)}, P:{int(perimeter)}', fontsize=8, color='yellow')
    plt.title(f'M-scan {i+1} Mask with Features')
    plt.tight_layout()
    plt.show()

# Display aggregated features for all M-scans
print("Aggregated Features for all M-scans:")
for feature in all_features:
    print(f"Position: ({feature[0]}, {feature[1]}), Area: {feature[2]}, Perimeter: {feature[3]}")