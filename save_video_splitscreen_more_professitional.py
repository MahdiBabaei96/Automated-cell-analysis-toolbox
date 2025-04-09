import os
import cv2
import numpy as np
from tifffile import imread

def crossfade(frame1, frame2, steps=3):
    """Generate a list of frames transitioning from frame1 to frame2."""
    fades = []
    for i in range(1, steps + 1):
        alpha = i / steps
        blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        fades.append(blended)
    return fades


# Paths
OCT_FOLDER = 'Dataset/frames_2-1024-512'
MASK_FOLDER = 'Dataset/frames_color_output_folder'
OUTPUT_VIDEO_PATH = 'oct_segmentation_video.mp4'

# Video settings
FRAME_WIDTH = 800
FRAME_HEIGHT = 400
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)
FPS = 1  # Frame per second

# Region of Interest (ROI)
ROI_X, ROI_Y, ROI_W, ROI_H = 390, 200, 200, 80  # Top-left and size of ROI

# Prepare images
oct_images = sorted([f for f in os.listdir(OCT_FOLDER) if f.endswith('.tif')])
mask_images = sorted([f for f in os.listdir(MASK_FOLDER) if f.endswith('.tif')])
assert len(oct_images) == len(mask_images), "Mismatch in number of OCT and mask images."

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, FRAME_SIZE)

# Function to preprocess images
def preprocess_image(path, is_mask=False):
    img = imread(path)
    img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_AREA)

    # Convert grayscale or alpha to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # 🛑 Key fix: Convert RGB → BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


# Add overlay text to a frame
def add_label(frame, label_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    
    # Shadow effect
    cv2.putText(frame, label_text, (10, 30), font, scale, shadow_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label_text, (10, 30), font, scale, color, thickness, cv2.LINE_AA)
    return frame

# Process each frame
for oct_name, mask_name in zip(oct_images, mask_images):
    # Load and preprocess images
    oct_path = os.path.join(OCT_FOLDER, oct_name)
    mask_path = os.path.join(MASK_FOLDER, mask_name)
    oct_img = preprocess_image(oct_path)
    mask_img = preprocess_image(mask_path, is_mask=True)

    # Crop ROI and resize
    oct_crop = oct_img[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    mask_crop = mask_img[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    oct_zoom = cv2.resize(oct_crop, (FRAME_WIDTH // 2, FRAME_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    mask_zoom = cv2.resize(mask_crop, (FRAME_WIDTH // 2, FRAME_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

    # Add labels
    oct_zoom = add_label(oct_zoom, "OCT Image")
    mask_zoom = add_label(mask_zoom, "Segmentation Mask")

    # Combine and write frame
    combined = np.hstack((oct_zoom, mask_zoom))
    if combined.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        print(f"Frame size mismatch. Skipping frame.")
        continue
    # If this is not the first frame, add transition
    if 'prev_frame' in locals():
        transition_frames = crossfade(prev_frame, combined, steps=0)
        for t_frame in transition_frames:
            video_writer.write(t_frame)

    video_writer.write(combined)
    prev_frame = combined


video_writer.release()
print(f"[✔] Zoomed-in video saved as: {OUTPUT_VIDEO_PATH}")
