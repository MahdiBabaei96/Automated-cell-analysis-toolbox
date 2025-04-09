import os
import cv2
import numpy as np
from tifffile import imsave, imread

# Folder paths
mask_folder = 'Dataset/frames_output_folder'  # Folder with predicted masks
output_folder = 'Dataset/frames_color_output_folder'  # Output folder

# Create output folder if it doesn’t exist
os.makedirs(output_folder, exist_ok=True)

# Define reference centroids (estimated manually or from first image)
reference_centroids = None

def match_contours_by_centroid(contours, ref_centroids):
    """Finds contours that match reference centroids based on closest distance."""
    matched_contours = []
    remaining_contours = contours.copy()

    for ref_cx, ref_cy in ref_centroids:
        min_distance = float("inf")
        best_match = None

        for contour in remaining_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distance = np.sqrt((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    best_match = contour

        if best_match is not None:
            matched_contours.append(best_match)
            remaining_contours = [c for c in remaining_contours if not np.array_equal(c, best_match)]


    return matched_contours

def process_mask(mask_path, output_path):
    """Processes a single mask image to keep only selected contours based on reference centroids."""
    global reference_centroids

    # Read mask
    mask = imread(mask_path)

    # Ensure it's a binary mask (threshold to enforce black & white)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find all contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"No contours found in {mask_path}")
        return

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # If it's the first image, establish reference centroids
    if reference_centroids is None:
        reference_centroids = []
        for i in [3, 4, 6]:  # Selecting 2nd, 3rd, and 5th largest contours
            if i < len(contours):
                M = cv2.moments(contours[i])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    reference_centroids.append((cx, cy))
        print(f"Reference centroids established: {reference_centroids}")

    # Find contours that best match reference centroids
    selected_contours = match_contours_by_centroid(contours, reference_centroids)

    if not selected_contours:
        print(f"No matching contours found in {mask_path}. Skipping.")
        return

    # Create a blank RGB mask (black background)
    mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Colors for the three contours
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red

    # Avoid text overlap by tracking used positions
    used_positions = []

    # Fill selected contours with color and place coordinates above them
    for i, contour in enumerate(selected_contours):
        color = colors[i % len(colors)]  # Cycle through colors if needed
        cv2.drawContours(mask_colored, [contour], -1, color, thickness=cv2.FILLED)  # Fill contour

        # Compute centroid for annotation
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Adjust text position dynamically to avoid overlap
            text_y = cy + 30
            while any(abs(text_y - prev_y) < 20 for prev_y in used_positions):  # Ensure spacing
                text_y += 20  # Move text further up

            used_positions.append(text_y)  # Store used y-coordinates to prevent overlap

            # Ensure text stays within image bounds
            text_y = max(20, text_y)  # Prevent text from going above the image

            # Draw text with the same color as the contour
            coord_text = f"({cx}, {cy})"
            # cv2.putText(mask_colored, coord_text, (cx, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
            #             0.7, color, 2, cv2.LINE_AA)  # Text color same as contour

    # Save the annotated mask
    imsave(output_path, mask_colored)
    print(f"Saved annotated mask: {output_path}")

# Process all masks in the folder
for filename in os.listdir(mask_folder):
    if filename.lower().endswith(".tif"):
        input_path = os.path.join(mask_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_mask(input_path, output_path)

print("Processing completed!")
