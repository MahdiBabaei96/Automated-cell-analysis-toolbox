import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.client import device_lib
import time
from tifffile import imread

# Function to load images from a directory
def load_images_from_folder(folder, color_mode=cv2.IMREAD_GRAYSCALE, target_size=(1024, 1024)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), color_mode)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
    return np.array(images)

def flipImage_old(image):
    images = [image.copy(), cv2.flip(image.copy(), -1), cv2.flip(image.copy(), 0), cv2.flip(image.copy(), 1)]
    return images

def flipImage(image):
    images = [image.copy(), cv2.flip(image.copy(), 1)]
    return images

def display_images(original, augmented, label):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.ravel()

    # axs[0].imshow(original, cmap='gray')
    # axs[0].set_title('Original Image')
    
    for i, aug in enumerate(augmented, 0):
        axs[i].imshow(aug, cmap='gray')
        axs[i].set_title(f'Augmented Image {i}')
    
    for i, lbl in enumerate(label, 4):
        axs[i].imshow(lbl, cmap='gray')
        axs[i].set_title(f'Label {i-3}')
    
    plt.show()

def display_input_and_prediction(input_image, true_mask, predicted_mask):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Input Image')

    axs[1].imshow(true_mask, cmap='gray')
    axs[1].set_title('True Mask')
    
    axs[2].imshow(predicted_mask, cmap='gray')
    axs[2].set_title('Predicted Mask')
    
    plt.show()

def display_prediction(input_image, predicted_mask):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Input Image')
    
    axs[1].imshow(predicted_mask, cmap='gray')
    axs[1].set_title('Predicted Mask')
    
    plt.show()

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

def extract_patches(image, patch_size):
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            patches.append(patch)
    return patches

def recompose_image(patches, image_size, patch_size):
    image = np.zeros(image_size)
    n_patches_per_row = image_size[1] // patch_size
    for idx, patch in enumerate(patches):
        i = (idx // n_patches_per_row) * patch_size
        j = (idx % n_patches_per_row) * patch_size
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))
        try:
            image[i:i+patch_size, j:j+patch_size] = patch
        except ValueError as e:
            print(f"Error at patch {idx} with shape {patch.shape}: {e}")
    return image

# Function to compute the distance between two cells
def compute_distance(cell1, cell2):
    return np.linalg.norm(np.array(cell1[:2]) - np.array(cell2[:2]))

# Function to extract features from segmented masks
def extract_features_from_mask_findContours(mask, frame_number):
    # Convert mask to binary if necessary
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    _, binary_mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)

    # Find contours of the cells
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for contour in contours:
        M = cv2.moments(contour)
        # Avoid division by area of zero 
        if M["m00"] == 0:  
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        simplified_contour = [tuple(point[0]) for point in contour]
        # Add more features as needed
        features.append([cX, cY, area, perimeter,simplified_contour,frame_number])  
    
    # Find all white pixels in the binary mask
    # white_pixels = np.column_stack(np.where(binary_mask > 0.1))
    # for (cX, cY) in white_pixels:
    #     if [cX, cY] not in features:  # Avoid duplicates
    #         features.append([cX, cY, 1, 0])  # Area of 1 pixel, perimeter of 0
    
    return features, contours

# Function to extract features from segmented masks using connected components
def extract_features_from_mask_connectedComponents(mask):
    # Convert mask to binary if necessary
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    _, binary_mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(binary_mask.astype(np.uint8))

    features = []
    for label in range(1, num_labels):  # Skip the background label 0
        component_mask = (labels_im == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:  # Avoid division by zero
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            # Add more features as needed
            features.append([cX, cY, area, perimeter])
    
    return features, labels_im

def load_bscan_image(file_path, color_mode=cv2.IMREAD_GRAYSCALE, target_size=None):
    img = imread(file_path)
    return img

def add_elements_to_list(list_of_lists, elements_to_add):
    for element in elements_to_add:
        added = False
        for sublist in list_of_lists:
            if element in sublist:
                sublist.extend(e for e in elements_to_add if e not in sublist)
                added = True
    return list_of_lists

def add_elements_to_list_new(list_of_lists, elements_to_add):
    for sublist in list_of_lists:
        if elements_to_add[0] in sublist:
            sublist.append(elements_to_add[1])
    return list_of_lists