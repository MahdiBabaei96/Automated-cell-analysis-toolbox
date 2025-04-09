import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from scipy.stats import linregress

def calculate_intensity(bscan):
    return np.mean(bscan)

def calculate_surface_roughness(bscan):
    edges = cv2.Canny(bscan, 100, 200)
    return np.mean(edges)

def calculate_optical_attenuation(bscan):
    # The average intensity profile along the depth (vertical direction)
    depth_profile = np.mean(bscan, axis=1)
    # Apply logarithm to the depth profile to linearize the exponential decay
    log_depth_profile = np.log(depth_profile + 1e-5)  # Adding a small value to avoid log(0)
    # The gradient (rate of change) of the log depth profile
    attenuation = -np.gradient(log_depth_profile)
    # The mean attenuation coefficient as a representative value
    return np.mean(attenuation)

def calculate_surface_area(bscan):
    edges = cv2.Canny(bscan, 100, 200)
    surface_area = np.sum(edges)  # Sum of edge pixels as a proxy for surface area
    return surface_area

def calculate_volume(bscan):
    volume = np.sum(bscan)  # Sum of intensities as a proxy for volume
    return volume

def calculate_surface_roughness_new(surface_area, volume):
    # surface_area = calculate_surface_area(bscan)
    # volume = calculate_volume(bscan)
    sr = (surface_area / (4 * np.pi))**(1/2) / (3 * (volume / (4 * np.pi))**(1/3))
    return sr

def calculate_flatness(bscan):
    # The standard deviation of the intensity values along the horizontal axis
    flatness = np.std(bscan, axis=1)
    # The mean flatness as a representative value
    return np.mean(flatness)

# Decorrelate consecutive B-scans
def calculate_decorrelation(bscans):
    decorrelations = []
    for i in range(len(bscans) - 1):
        decorrelation = np.mean(np.abs(bscans[i] - bscans[i + 1]))
        decorrelations.append(decorrelation)
    print(decorrelations)
    return np.mean(decorrelations)


def calculate_attenuation_for_cell(mesh, volume):
    # Assume mesh vertices correspond to indices in the volume
    vertices = mesh.vertices.astype(int)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    intensity_data = volume[z, y, x]  # Extracting intensity data for the cell
    
    # Calculate average attenuation for the cell (example with a simple linear fit)
    # Flatten data for simplicity, assuming homogeneity along one axis
    flattened_intensity = intensity_data.flatten()
    zs = np.arange(flattened_intensity.size) * 0.1  # Assuming 0.1 um per slice for depth axis
    
    # Linear regression on log of intensity data
    slope, intercept, r_value, p_value, std_err = linregress(zs, np.log(flattened_intensity + 1e-10))
    attenuation_coefficient = -slope
    
    return attenuation_coefficient

import numpy as np
from scipy.stats import linregress

def calculate_attenuation_for_cell2(mesh, volume, x_resolution=0.002, ascan_win=20, rms_threshold=25, win_step=4, dt=8, ds=8):
    # Assume mesh vertices correspond to indices in the volume
    vertices = mesh.vertices.astype(int)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Initialize an empty list to store attenuation coefficients for each cell
    attenuation_coefficients = []

    for x_start_index in range(0, len(x) - ascan_win, ascan_win):
        # Average intensity data within the window
        a_line = np.mean(volume[z[x_start_index:x_start_index + ascan_win],
                                y[x_start_index:x_start_index + ascan_win],
                                x[x_start_index:x_start_index + ascan_win]], axis=1)
        max_pos = np.argmax(a_line)

        # Initialize variables for dynamic fitting
        anchor = max_pos
        search_w = 40
        previous_rms = float('inf')
        previous_fit_flag = 0
        
        while anchor + search_w < len(a_line):
            search_data = a_line[anchor:anchor + search_w]
            zs = np.arange(search_data.size) * x_resolution
            
            # Perform linear regression on the log of the intensity data
            slope, _, _, _, rms = linregress(zs, np.log(search_data + 1e-10))
            ut = -slope  # Attenuation coefficient

            if ut <= 0:
                ut = 0
            
            if rms <= rms_threshold:
                if rms <= previous_rms:
                    # Continue fitting as RMS is decreasing
                    previous_rms = rms
                    search_w += win_step
                    previous_fit_flag = 1
                else:
                    # Save the optimal attenuation coefficient for the cell
                    attenuation_coefficients.append(ut)
                    search_w = 40
                    anchor += ds
                    previous_fit_flag = 1
                    break
            else:
                if previous_fit_flag == 1:
                    attenuation_coefficients.append(ut)
                    search_w = 40
                    anchor += ds
                    previous_fit_flag = 0
                    previous_rms = rms
                    break
                else:
                    anchor += dt
                    previous_fit_flag = 0
                    previous_rms = rms
    
    return attenuation_coefficients
