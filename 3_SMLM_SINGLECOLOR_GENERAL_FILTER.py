import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import os
from scipy.spatial import cKDTree

# General Description:
# This script processes 3D point data from a text file to generate 2D scatter plots (XY, XZ, YZ views),
# calculates basic statistics for certain properties, and saves the results as images and a text file.
# It includes filtering to retain points with sufficient density of neighbors and customizable visualization parameters.

# Visualization parameters
xy_aspect_ratio = 1
yz_aspect_ratio = 2
xz_aspect_ratio = 2
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 600
fontsize_axis_title = 18

# Input and processing parameters
input_file_path = "path/to/your/input_data.txt"  # Replace with the actual path to the input data file
filtering_radius = 0.2
filtering_min_neighbours = 10

def get_base_name(input_file_path):
    """Generate a base name from the input file path for consistent naming of output files."""
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name

def set_aspect_ratio(ax, ratio=1):
    """Set the aspect ratio of a 2D plot."""
    ax.set_aspect(ratio)

def filter_points(x, y, z, radius=filtering_radius, min_neighbors=filtering_min_neighbours):
    """Filter points based on density of neighboring points."""
    print(f"Starting filtering points with radius = {radius} and min_neighbors = {min_neighbors}")
    points = np.column_stack((x, y, z))
    tree = cKDTree(points)  # Create a KD-Tree for efficient neighbor search
    neighbor_counts = tree.query_ball_point(points, r=radius)  # Find neighbors within the specified radius
    num_neighbors = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])  # Exclude the point itself
    mask = num_neighbors >= min_neighbors  # Apply filter based on minimum neighbors
    print(f"Filtering complete. {np.sum(mask)} points retained out of {len(x)}.")
    return x[mask], y[mask], z[mask]

def plot_3d_scatter_from_txt(input_file_path, output_directory, sphere_radius=0.05, filter_radius=0.1, min_neighbors=5):
    """Generate scatter plots from 3D data and save as images."""
    print(f"Processing file: {input_file_path}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create output directory if it doesn't exist
        print(f"Created output directory: {output_directory}")

    print("Loading data...")
    df = pd.read_csv(input_file_path, sep='\t', header=None)  # Load tab-separated data
    print(f"Data loaded. {len(df)} points found.")
    
    # Extract relevant columns for X, Y, and Z coordinates
    x = df.iloc[:, 0].values / 1000  # Scale to micrometers
    y = df.iloc[:, 1].values / 1000
    z = df.iloc[:, 4].values / 1000
    
    # Apply filtering to retain points with sufficient density
    x, y, z = filter_points(x, y, z, radius=filter_radius, min_neighbors=min_neighbors)
    
    # Normalize Z values for color mapping
    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = plt.get_cmap('viridis')
    sphere_area = np.pi * (sphere_radius * 1000) ** 2  # Adjust point size

    # Generate scatter plots for different planes
    for plane, (x_data, y_data, aspect_ratio, file_suffix) in enumerate([
        (x, y, xy_aspect_ratio, "xy_view"),
        (x, z, xz_aspect_ratio, "xz_view"),
        (y, z, yz_aspect_ratio, "yz_view"),
    ]):
        fig, ax = plt.subplots(figsize=(15, 15 if plane == 0 else 6))
        scatter = ax.scatter(x_data, y_data, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
        ax.set_xlabel('X [μm]' if plane != 2 else 'Y [μm]', fontsize=fontsize_axis_title)
        ax.set_ylabel('Y [μm]' if plane == 0 else 'Z [μm]', fontsize=fontsize_axis_title)
        set_aspect_ratio(ax, aspect_ratio)
        fig.savefig(os.path.join(output_directory, f"{file_suffix}.png"), dpi=dpi, transparent=True, bbox_inches='tight')
        plt.tight_layout()
        print(f"Saved {file_suffix} plot.")

    print("Plot generation complete.")

def save_statistics(input_file_path, output_directory):
    """Calculate and save basic statistics of photon intensity and localization precision."""
    print("Calculating statistics...")
    df = pd.read_csv(input_file_path, sep='\t', header=None)
    photon_intensity = df.iloc[:, 2]  # Photon intensity column
    localization_precision = df.iloc[:, 3]  # Localization precision column

    # Calculate statistics
    stats = {
        "Mean Photon Intensity": photon_intensity.mean(),
        "Std Photon Intensity": photon_intensity.std(),
        "Mean Localization Precision (nm)": localization_precision.mean(),
        "Std Localization Precision (nm)": localization_precision.std()
    }

    # Save statistics to a file
    stats_file_path = os.path.join(output_directory, "statistics.txt")
    with open(stats_file_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")
    print(f"Saved statistics to {stats_file_path}")

# Example usage with placeholder paths
output_directory = "path/to/your/output_directory"  # Replace with the actual output directory path
plot_3d_scatter_from_txt(input_file_path, output_directory, sphere_radius=0.0009, filter_radius=filtering_radius, min_neighbors=filtering_min_neighbours)
save_statistics(input_file_path, output_directory)
