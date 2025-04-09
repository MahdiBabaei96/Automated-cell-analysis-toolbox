import os
import cv2
import numpy as np
from tifffile import imread
from PIL import Image
import matplotlib.pyplot as plt

# Define paths
oct_folder = 'Dataset/frames_2-1024-512'
mask_folder = 'Dataset/frames_color_output_folder'
output_video_path = 'oct_segmentation_video.mp4'

# Video settings
frame_width = 800  # Increased width
frame_height = 400  # Increased height
frame_size = (frame_width, frame_height)  # Side-by-side width * 2 (for OCT and Mask)
fps = 1  # Each frame is shown for 1 second

# Define ROI (Region of Interest)
x, y, w, h = 390, 200, 200, 80  # (Top-left: x=350, y=400), Width=250, Height=200

# Load and sort images
oct_images = sorted([f for f in os.listdir(oct_folder) if f.endswith('.tif')])
mask_images = sorted([f for f in os.listdir(mask_folder) if f.endswith('.tif')])

# Ensure both have the same number of frames
assert len(oct_images) == len(mask_images), "Mismatch in number of OCT and mask images."

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Process and combine images
for oct_img_name, mask_img_name in zip(oct_images, mask_images):
    # Load and resize OCT image
    oct_path = os.path.join(oct_folder, oct_img_name)
    oct_image = imread(oct_path)
    oct_image = cv2.resize(oct_image, (1024, 512))  # Resize to 1024x512

    # Load and resize Mask image
    mask_path = os.path.join(mask_folder, mask_img_name)
    mask_image = imread(mask_path)
    mask_image = cv2.resize(mask_image, (1024, 512))

    # Convert grayscale to RGB (if needed)
    if oct_image.shape[-1] == 4:  # Handle alpha channel if present
        oct_image = cv2.cvtColor(oct_image, cv2.COLOR_BGRA2BGR)
    elif len(oct_image.shape) == 2:  # Convert grayscale to RGB
        oct_image = cv2.cvtColor(oct_image, cv2.COLOR_GRAY2RGB)
    oct_image = cv2.cvtColor(oct_image, cv2.COLOR_RGB2BGR)
    if mask_image.shape[-1] == 4:  # Handle alpha channel if present
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2BGR)
    elif len(mask_image.shape) == 2:  # Convert grayscale to RGB
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
    # Crop ROI (Zoomed-in region)
    oct_crop = oct_image[y:y+h, x:x+w]
    mask_crop = mask_image[y:y+h, x:x+w]
    oct_crop_resized = cv2.resize(oct_crop, (frame_width // 2, frame_height), interpolation=cv2.INTER_LANCZOS4)
    mask_crop_resized = cv2.resize(mask_crop, (frame_width // 2, frame_height), interpolation=cv2.INTER_LANCZOS4)

    
    # Concatenate images side by side
    combined_frame = np.hstack((oct_crop_resized, mask_crop_resized))
    print(f"Combined frame shape: {combined_frame.shape}")
    if combined_frame.shape[1] != frame_size[0] or combined_frame.shape[0] != frame_size[1]:
        print(f"Warning: Frame size mismatch. Expected {frame_size}, got {combined_frame.shape[:2]}.")
        continue
    
    # Write frame to video
    video_writer.write(combined_frame)

# Release the video writer
video_writer.release()
print(f"Zoomed-in video saved as {output_video_path}")
