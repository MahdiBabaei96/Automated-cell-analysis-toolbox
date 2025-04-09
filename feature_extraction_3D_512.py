import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# from tensorflow.keras.models import load_model
from segmentation_model_architecture import vgg16_unet, custom_vgg16_unet
from utils import *
from metric import *
from scipy.optimize import linear_sum_assignment
import pandas as pd
from scipy.spatial import Delaunay
import trimesh
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

print("Current working directory:", os.getcwd())

# Load your model
model_path = 'models/FullTraining-Model-NewAugmentation_FocalLoss_custom_vgg16_unet_V01.hdf5'
input_shape = (512, 512, 3)  # Make sure this matches the shape used during training
# model = vgg16_unet(input_shape)
model = custom_vgg16_unet(input_shape)
# Load the weights
model.load_weights(model_path)

# Reading the 3D volume 
image_folder = 'data/test/'

for filename in os.listdir(image_folder):
    if filename.endswith('.tif'):
        file_path = os.path.join(image_folder, filename)
        volume_image = load_bscan_image(file_path, color_mode=cv2.IMREAD_COLOR)


        volume_image = [cv2.resize(slice, (512, 512)) for slice in volume_image]
        volume_image = np.stack(volume_image, axis=0)
        depth, height, width = volume_image.shape
        print(f"B-scan image dimensions: {depth}x{height}x{width}")

        # Each B-scan is a vertical slice of the 3D Volume
        all_features = []
        intensity_list = []
        surface_roughness_list = []
        surface_roughness_list_new = []
        optical_attenuation_list = []
        volume = volume_image
        for i in range(depth):
            # Extract the M-scan
            mscan = volume_image[i, :, :]

            # Check if the M-scan is empty
            if mscan.size == 0:
                print(f"Empty M-scan at index {i}")
                continue
            
            # print(f"Processing M-scan {i+1}/{depth}")
            
            # Convert to 3 channels by stacking the grayscale image
            mscan_color = np.stack([mscan]*3, axis=-1)
            
            # Resize to model's input shape
            mscan_resized = cv2.resize(mscan_color, (512, 512))

            # Predict masks using the loaded model
            predicted_mask = model.predict(np.expand_dims(mscan_resized, axis=0))[0]
            # display_prediction(mscan_resized.squeeze(), predicted_mask.squeeze())
            
            # Extract features from the predicted mask
            features, contours = extract_features_from_mask_findContours(predicted_mask,i)
            all_features.append(features)
            # print(contours)

        # Initialize tracks
        tracks = []
        for frame_idx, frame in enumerate(all_features):
            if frame_idx == 0:
                # Initialize tracks with the first frame
                for cell in frame:
                    tracks.append([cell])
            else:
                # Match cells between the previous frame and the current frame
                previous_frame = all_features[frame_idx - 1]
                current_frame = frame
                
                cost_matrix = np.zeros((len(previous_frame), len(current_frame)))
                for i, prev_cell in enumerate(previous_frame):
                    for j, curr_cell in enumerate(current_frame):
                        cost_matrix[i, j] = compute_distance(prev_cell, curr_cell)
                        # print("previous_frame:", prev_cell)
                        # print("current_frame:", curr_cell)
                        # print(cost_matrix[i, j])

                # Solve the assignment problem
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Update tracks with matched cells
                matched_prev_cells = set()
                matched_curr_cells = set()
                
                # for i, j in zip(row_ind, col_ind):
                for i in range(len(previous_frame)):
                    for j in range(len(current_frame)):
                        if cost_matrix[i, j] < 10:  # Define a threshold for matching
                            a = []
                            a.append(previous_frame[i])
                            a.append(current_frame[j])
                            tracks = add_elements_to_list_new(tracks,a)
                            # tracks[i].append((current_frame[j], frame_idx))
                            matched_prev_cells.add(i)
                            matched_curr_cells.add(j)
                
                # Create new tracks for unmatched cells in the current frame
                for j, curr_cell in enumerate(current_frame):
                    if j not in matched_curr_cells:               
                        tracks.append([curr_cell])


        # Prepare points for 3D reconstruction
        all_points = []
        all_meshes = []
        all_meshes_hull = []
        for i in range(len(tracks)):
            # print(tracks[i])
            points = []
            if len(tracks[i])>=2:
                # print(tracks[i])
                for j in range(len(tracks[i])):
                    
                    z = tracks[i][j][5]
                    for k in range(len(tracks[i][j][4])):
                        x = tracks[i][j][4][k][0]
                        y = tracks[i][j][4][k][1]
                        points.append([x, y, z])

                points = np.array(points)
                
                # Perform Delaunay triangulation
                tri = Delaunay(points)
                
                # # Extract the simplices (triangles) from the triangulation
                triangles = tri.simplices
                
                # # Create a Trimesh object
                mesh = trimesh.Trimesh(vertices=points, faces=triangles)
                
                # attenuation = calculate_attenuation_for_cell(mesh, volume)
                
                # # mesh.show()
                
                # # Calculate the surface area
                # surface_area = mesh.area
                # # print(f"Surface Area of the Delaunay 3D shape: {surface_area}")

                # # hull = ConvexHull(points)
                # # # Create a Trimesh object
                # # mesh_hull = trimesh.Trimesh(vertices=points, faces=hull.simplices)
                # # # Calculate the surface area
                # # surface_area = mesh.area
                # # print(f"Surface Area of the mesh_hull 3D shape: {surface_area}")
                # # all_meshes_hull.append(mesh_hull)

                # pca = PCA(n_components=3)
                # pca.fit(points)
                # # Eigenvalues represent the variance along the principal components
                # eigenvalues = pca.explained_variance_
                # # Radii are proportional to the square root of the eigenvalues
                # radii = np.sqrt(eigenvalues)
                # # Assign radii to largest, medium, and smallest
                # R_largest, R_medium, R_smallest = sorted(radii, reverse=True)
                
                # Perform PCA
                # pca = PCA(n_components=3)
                # pca.fit(points)        
                # # Transform points to the PCA space
                # transformed_points = pca.transform(points)
                # # Calculate the extents (radii) along each principal component axis
                # radii = np.max(transformed_points, axis=0) - np.min(transformed_points, axis=0)
                # # Sort the radii to get the largest, medium, and smallest
                # sorted_radii = np.sort(radii)
                # # Extract the radii
                # Rsmallest, Rmedium, Rlargest = sorted_radii
                
                # Calculate Elongation
                # elongation = R_largest / R_medium
                
                # # Calculate Flatness
                # flatness = R_medium / R_smallest
                
                # surface_roughness = calculate_surface_roughness_new(surface_area, len(tracks[i]))
                all_meshes.append(mesh)
                all_points.extend(points)
                # tracks[i].append([surface_area, len(tracks[i]), surface_roughness, elongation, flatness, attenuation])
                
                # print(f"Total surface area of the cell: {surface_area}")
                # print(f"Total number of volume: {len(tracks[i])}")
                # print(f"Surface roughness:{surface_roughness}")
                # print(f"Elongation: {elongation}")
                # print(f"Flatness: {flatness}")
                # print('########')

        # Display tracks
        # for track in tracks:
            # print("Track:", track)

        # Save to CSV
        df = pd.DataFrame(tracks)
        out_path = 'results/features/'
        out_file = os.path.join(out_path, filename[:,-4])
        out_file = os.path.join(out_file, '.csv')
        df.to_csv(out_file, index=False, header=None)