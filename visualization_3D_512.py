import csv
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from metric import *
from scipy.spatial import Delaunay
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA




# Pixel sizes from the table (in mm or �m depending on the units)
pixel_size_x = 2.00 / 512  # 2.00 mm over 1000 pixels = 0.002 mm/pixel (2 �m/pixel)
pixel_size_y = 2.00 / 512  # 2.00 mm over 1000 pixels = 0.002 mm/pixel (2 �m/pixel)
pixel_size_z = 1.99 / 1000  # 1.99 mm over 1024 pixels H 0.00194 mm/pixel (1.94 �m/pixel)

# Convert from mm to �m (assuming your cell volume is in �m�)
conversion_factor = 1000  # Convert mm to �m

csv_folder = 'results/features/'

for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        print(file_path)
        tracks = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                cleaned_row = [item for item in row if item.strip()]
                # Parse the string representations of lists into actual lists
                parsed_row = [eval(item) if item.startswith('[') else item for item in cleaned_row]
                # print(parsed_row)
                tracks.append(parsed_row)

        # print(tracks)
        # Prepare points for 3D reconstruction
        surface_area_all = []
        frames_all = []
        surface_roughness_all = []
        elongation_all = []
        flatness_all = []
        attenuation_all = []
        volume_all = []

        all_points = []
        all_meshes = []
        all_meshes_hull = []
        for i in range(len(tracks)):
            points = []  
            if len(tracks[i])>=2:
                for j in range(len(tracks[i])-1):  
                    z = tracks[i][j][5] * conversion_factor * pixel_size_z  # Convert Z to �m
                    for k in range(len(tracks[i][j][4])):
                        x = tracks[i][j][4][k][0] * conversion_factor * pixel_size_x  # Convert X to �m
                        y = tracks[i][j][4][k][1] * conversion_factor * pixel_size_y  # Convert Y to �m
                        points.append([x, y, z])
                # print('#########################')
                # print(points)
                points = np.array(points)
                
                if len(np.unique(points[:, 2])) <= 1:  # Skip if all Z-values are the same
                    # print(f"Skipping points from sample {i} due to lack of frame variation.")
                    continue
                
                pca = PCA(n_components=3)
                pca.fit(points)        
                # Transform points to the PCA space
                transformed_points = pca.transform(points)
                # Calculate the extents (radii) along each principal component axis
                radii = np.max(transformed_points, axis=0) - np.min(transformed_points, axis=0)
                # Sort the radii to get the largest, medium, and smallest
                sorted_radii = np.sort(radii)
                # Extract the radii
                Rsmallest, Rmedium, Rlargest = sorted_radii
                # if Rlargest > 15 or Rlargest < 10:
                    # continue

                pca = PCA(n_components=3)
                pca.fit(points)
                # Eigenvalues represent the variance along the principal components
                eigenvalues = pca.explained_variance_
                # Radii are proportional to the square root of the eigenvalues
                radii = np.sqrt(eigenvalues)
                # Assign radii to largest, medium, and smallest
                R_largest, R_medium, R_smallest = sorted(radii, reverse=True)
                # if R_largest > 15 or Rlargest < 10:
                    # continue
                
                # Perform Delaunay triangulation
                tri = Delaunay(points)
                # Extract the simplices (triangles) from the triangulation
                triangles = tri.simplices
                # Create a Trimesh object
                mesh = trimesh.Trimesh(vertices=points, faces=triangles)
                mesh = mesh.convex_hull  # Replace mesh with its convex hull
                # mesh.remove_degenerate_faces() 
                centroid = mesh.centroid
                distances = np.linalg.norm(mesh.vertices - centroid, axis=1)
                largest_radius = np.max(distances)
                # if largest_radius > 15 or largest_radius < 10:
                    # continue
                    
                cell_volume = mesh.volume
                # if int(cell_volume) < 4186 or int(cell_volume) > 14130:
                    # continue
                # print(Rlargest)
                # print(R_largest)
                # print(f"The largest radius is: {largest_radius}")
                # print(cell_volume)
                # print('$$$$$$$$$')
                pca = PCA(n_components=3)
                pca.fit(points)        
                # Transform points to the PCA space
                transformed_points = pca.transform(points)
                # Calculate the extents (radii) along each principal component axis
                radii = np.max(transformed_points, axis=0) - np.min(transformed_points, axis=0)
                # Sort the radii to get the largest, medium, and smallest
                sorted_radii = np.sort(radii)
                # Extract the radii
                Rsmallest, Rmedium, Rlargest = sorted_radii
                if Rlargest > 15 or Rlargest < 10:
                    continue

                pca = PCA(n_components=3)
                pca.fit(points)
                # Eigenvalues represent the variance along the principal components
                eigenvalues = pca.explained_variance_
                # Radii are proportional to the square root of the eigenvalues
                radii = np.sqrt(eigenvalues)
                # Assign radii to largest, medium, and smallest
                R_largest, R_medium, R_smallest = sorted(radii, reverse=True)
                # if R_largest > 15 or Rlargest < 10:
                    # continue
                
                    
                cell_volume = mesh.volume
                surface_area = mesh.area
                elongation = R_largest / R_medium
                # Calculate Flatness
                flatness = R_medium / R_smallest
                surface_roughness = calculate_surface_roughness_new(surface_area, len(tracks[i]))
                # attenuation = calculate_attenuation_for_cell(mesh, volume)
                volume_all.append(cell_volume)
                surface_area_all.append(int(surface_area))
                frames_all.append(int(len(tracks[i])))
                surface_roughness_all.append(int(surface_roughness))
                elongation_all.append(int(elongation))
                flatness_all.append(int(flatness))
                # attenuation_all.append(int(attenuation))
                # all_meshes.append((mesh, f"Surface Area: {surface_area}, Frames: {frames}, Surface Roughness: {surface_roughness}, Elongation: {elongation}, Flatness: {flatness}, Attenuation Coefficient: {attenuation}"))
                all_meshes.append((mesh, f"Surface Area: {surface_area}, Frames: {len(tracks[i])}, Surface Roughness: {surface_roughness}, Elongation: {elongation}, Flatness: {flatness}, Volume: {cell_volume}"))
                all_points.extend(points)
                # mesh.show()
        print(f"cells of all: {len(surface_area_all)}")
        print(f"cells of all: {len(all_meshes)}")
        print(f"Average of surface area: {np.mean(surface_area_all)}")
        print(f"Average of frame numbers: {np.mean(frames_all)}")
        print(f"Average of surface roughness: {np.mean(surface_roughness_all)}")
        print(f"Average of elongation: {np.mean(elongation_all)}")
        print(f"Average of flatness: {np.mean(flatness_all)}")
        print(f"Average of attenuation: {np.mean(attenuation_all)}")
        print(f"Average of volume: {np.mean(volume_all)}")


        # surface_area_all = np.array(surface_area_all)
        # # Set up the figure and axis
        # plt.figure(figsize=(10, 6))
        # # Plot the histogram directly using matplotlib
        # plt.hist(surface_area_all, bins=50, color='blue', edgecolor='black')
        # # Set labels and title
        # plt.title('Distribution of Surface Areas')
        # plt.xlabel('Surface Area')
        # plt.ylabel('Frequency')
        # # Show the plot
        # plt.show()

        # # Combine all meshes into one
        # combined_mesh = trimesh.util.concatenate(all_meshes)
        # combined_mesh.export('combined_mesh.obj')  # Save as OBJ file
        # combined_mesh.export('combined_mesh.stl')  # Save as STL file
        # combined_mesh.export('combined_mesh.ply') 
        # # Show the combined mesh using Trimesh
        # combined_mesh.show()

        # Create a 3D scatter plot using Matplotlib
        # all_points = np.array(all_points)

        # # Create a 3D scatter plot using Plotly
        # fig = go.Figure(data=[go.Scatter3d(
        #     x=all_points[:, 0],
        #     y=all_points[:, 1],
        #     z=all_points[:, 2],
        #     mode='markers',
        #     marker=dict(
        #         size=5,
        #         color='blue',    # Set color to blue
        #         opacity=0.8
        #     )
        # )])
        # # Add labels
        # fig.update_layout(
        #     scene=dict(
        #         xaxis_title='X',
        #         yaxis_title='Y',
        #         zaxis_title='Z'
        #     ),
        #     title='3D Scatter Plot'
        # )
        # # Save as an interactive HTML file
        # fig.write_html('Code/Results/plot/New/Default_0233_Mode3D-dead-711_scatter_plot.html')

        fig = go.Figure()
        # Add each mesh to the plot
        for mesh, label in all_meshes:
            fig.add_trace(go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                opacity=0.70,
                color='red',
                text=[label] * len(mesh.vertices),
                hoverinfo='text'
            ))
        all_points_np = np.vstack(all_points)
        # x_min, y_min, z_min = all_points_np.min(axis=0)
        # x_max, y_max, z_max = all_points_np.max(axis=0)
        x_min, y_min, z_min = 0,300,0
        x_max, y_max, z_max = 2000,2000,1990
        # Set axis ranges to encompass all points comfortably
        buffer_x = (x_max - x_min) * 0.1  # 10% of the range as buffer
        buffer_y = (y_max - y_min) * 0.1
        buffer_z = (z_max - z_min) * 0.1

        x_range = [x_min - buffer_x, x_max + buffer_x]
        # y_range = [60, y_max + buffer_y-140]
        y_range = [y_min - buffer_y, y_max + buffer_y]
        z_range = [z_min - buffer_z, z_max + buffer_z]

        # Draw the cube border around the plot
        cube_lines = [
            # Bottom square (z = z_range[0])
            ([x_range[0], x_range[1]], [y_range[0], y_range[0]], [z_range[0], z_range[0]]),
            ([x_range[1], x_range[1]], [y_range[0], y_range[1]], [z_range[0], z_range[0]]),
            ([x_range[1], x_range[0]], [y_range[1], y_range[1]], [z_range[0], z_range[0]]),
            ([x_range[0], x_range[0]], [y_range[1], y_range[0]], [z_range[0], z_range[0]]),
            
            # Top square (z = z_range[1])
            ([x_range[0], x_range[1]], [y_range[0], y_range[0]], [z_range[1], z_range[1]]),
            ([x_range[1], x_range[1]], [y_range[0], y_range[1]], [z_range[1], z_range[1]]),
            ([x_range[1], x_range[0]], [y_range[1], y_range[1]], [z_range[1], z_range[1]]),
            ([x_range[0], x_range[0]], [y_range[1], y_range[0]], [z_range[1], z_range[1]]),
            
            # Vertical lines connecting bottom and top squares
            ([x_range[0], x_range[0]], [y_range[0], y_range[0]], [z_range[0], z_range[1]]),
            ([x_range[1], x_range[1]], [y_range[0], y_range[0]], [z_range[0], z_range[1]]),
            ([x_range[1], x_range[1]], [y_range[1], y_range[1]], [z_range[0], z_range[1]]),
            ([x_range[0], x_range[0]], [y_range[1], y_range[1]], [z_range[0], z_range[1]]),
        ]
        # Add lines to create the cube border
        for line in cube_lines:
            fig.add_trace(go.Scatter3d(
                x=line[0],
                y=line[1],
                z=line[2],
                mode='lines',
                line=dict(color='Black', width=4),  # Adjust color and width as necessary
                showlegend=False
            ))
        # scale_length = 100  # Length of the scale bar (adjust based on your data)
        # fig.add_trace(go.Scatter3d(
        #     x=[0, scale_length],
        #     y=[0, 0],
        #     z=[0, 0],
        #     mode='lines',
        #     line=dict(color='black', width=5),
        #     name='Scale Bar'
        # ))
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, title=dict(text=''), zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, title=dict(text=''), zeroline=False),
                zaxis=dict(showticklabels=False, showgrid=False, title=dict(text=''), zeroline=False)
            ),
            margin=dict(l=50, r=50, b=50, t=50),
            paper_bgcolor='white',  # Transparent background
            scene_bgcolor='white'  # Light gray background to match your provided image
        )

        # Save as an interactive HTML file
        out_path = 'results/visualization/'
        out_file = os.path.join(out_path, filename[:-4]+'.html')
        # out_file = os.path.join(out_file, '.html')
        fig.write_html(out_file)
        # fig.write_html('Results/Default_1046_Mode3D_morecell.html')

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='b', marker='o')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # Save as PNG file
        # plt.savefig('Code/Results/plot/test_Default_0002_Mode3D-3d_scatter_plot.png')  
        # plt.savefig('Code/Results/plot/test_Default_0002_Mode3D-3d_scatter_plot.pdf')
        # plt.show()
