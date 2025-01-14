import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.spatial import KDTree
import os

# The code loads 3D point data from a file, filters it based on proximity to neighboring points and a region of
# interest (ROI), and rescales the coordinates. It generates three scatter plots (XY, XZ, YZ views), color-coded 
# by the z-coordinate values. The plots are saved as high-resolution PNG images with appropriate axis labels. 
# The code is designed to visualize the data from different perspectives for further analysis.

# Aspect ratios and figure settings
xy_aspect_ratio = 1  # Aspect ratio for XY plane view
yz_aspect_ratio = 0.5  # Aspect ratio for YZ plane view
xz_aspect_ratio = 0.5  # Aspect ratio for XZ plane view
xyz_aspect_ratio = [1, 1, 0.01]  # Aspect ratio for 3D view
dpi = 600  # DPI for high-quality figure output
fontsize_axis_title = 18  # Font size for axis titles

# File and filtering parameters (substituted sensitive data with placeholders)
input_file_path = 'path/to/your/data_file.txt'  # Input data file path (change to actual file path)
filtering_radius = 0.2  # Radius to filter neighboring points (in micrometers)
filtering_min_neighbours = 5  # Minimum number of neighboring points required
roi_center = (20, 10)  # Center of the region of interest (ROI)
roi_diameter = 5.0  # Diameter of the region of interest (ROI)
target_range = (0, 5)  # Rescaling range for x and y coordinates

# Helper function to extract base name from file path
def get_base_name(input_file_path):
    """Extracts the base name of the file from the input file path."""
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name

# Helper function to set aspect ratio for plots
def set_aspect_ratio(ax, ratio=1):
    """Sets the aspect ratio of the plot to the specified ratio."""
    ax.set_aspect(ratio)

# Function to filter points based on the density of neighboring points using a KDTree
def filter_points(x, y, z, filtering_radius, filtering_min_neighbours):
    """Filters points based on their neighboring points' density."""
    # Combine x, y, z into a single array of coordinates
    points = np.vstack((x, y, z)).T
    
    # Create a KDTree for efficient neighbor search
    tree = KDTree(points)
    
    # Query the tree to find the number of neighbors within filtering_radius
    neighbor_counts = tree.query_ball_point(points, r=filtering_radius)
    
    # Filter points based on the number of neighbors
    mask = np.array([len(neighbors) >= filtering_min_neighbours for neighbors in neighbor_counts])
    
    return x[mask], y[mask], z[mask]

# Main function to load data, filter it, and generate scatter plots
def plot_3d_scatter_from_txt(input_file_path, output_directory, sphere_radius=0.05, 
                             filter_radius=0.1, min_neighbors=5, 
                             roi_center=roi_center, roi_diameter=roi_diameter, 
                             target_range=target_range):
    """Loads data, applies filters, and generates 3D scatter plots for different views."""
    print(f"Processing file: {input_file_path}")
    
    # Set up output directory (adjusted based on input file's relative path)
    base_input_directory = 'path/to/base/directory'  # Base directory for input files
    input_directory = os.path.dirname(input_file_path)
    relative_path = os.path.relpath(input_directory, start=base_input_directory)
    full_output_directory = output_directory  # Use provided output directory
    
    # Create output directory if it doesn't exist
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)
        print(f"Created output directory: {full_output_directory}")
    
    print("Loading data...")
    # Load tab-separated data into a pandas DataFrame
    df = pd.read_csv(input_file_path, sep='\t', header=None)  # Adjust header if necessary
    print(f"Data loaded. {len(df)} points found.")
    
    # Extract x, y, z coordinates from the data
    x = df.iloc[:, 0].values / 1000  # First column for x values (converted to micrometers)
    y = df.iloc[:, 1].values / 1000  # Second column for y values (converted to micrometers)
    z = df.iloc[:, 4].values / 1000  # Fifth column for z values (converted to micrometers)
    
    # Apply filtering based on the number of neighboring points
    x, y, z = filter_points(x, y, z, filtering_radius=filter_radius, filtering_min_neighbours=min_neighbors)

    # Apply region of interest (ROI) filtering
    roi_radius = roi_diameter / 2.0
    roi_x_min, roi_x_max = roi_center[0] - roi_radius, roi_center[0] + roi_radius
    roi_y_min, roi_y_max = roi_center[1] - roi_radius, roi_center[1] + roi_radius
    mask_roi = (x >= roi_x_min) & (x <= roi_x_max) & (y >= roi_y_min) & (y <= roi_y_max)
    x, y, z = x[mask_roi], y[mask_roi], z[mask_roi]
    print(f"ROI filtering applied. {len(x)} points retained.")
    
    # Rescale coordinates to the target range
    x_rescaled = np.interp(x, (roi_x_min, roi_x_max), target_range)
    y_rescaled = np.interp(y, (roi_y_min, roi_y_max), target_range)
    
    # Normalize z values for color mapping
    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = plt.get_cmap('viridis')  # Colormap for scatter plot
    sphere_area = np.pi * (sphere_radius * 1000) ** 2  # Area of spheres for scatter plot markers
    
    # Generate base name for the output files
    base_name = get_base_name(input_file_path)
    
    print("Generating individual plots...")

    # XY Plane View (within ROI and rescaled)
    fig_xy, ax_xy = plt.subplots(figsize=(15, 15))
    sc_xy = ax_xy.scatter(x_rescaled, y_rescaled, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_xy.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xy.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    cbar = plt.colorbar(sc_xy, ax=ax_xy, label='Z [μm]', shrink=0.7)
    cbar.set_label(label='Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_xy, ratio=xy_aspect_ratio)
    fig_xy.savefig(os.path.join('./IMAGES/xy_view_specific.png'), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close(fig_xy)
    print("Saved XY view plot within ROI.")
    
    # XZ Plane View (not rescaled, filtered to ROI)
    fig_xz, ax_xz = plt.subplots(figsize=(10, 5))
    sc_xz = ax_xz.scatter(x, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_xz.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xz.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_xz, ratio=xz_aspect_ratio)
    fig_xz.savefig(os.path.join('./IMAGES/xz_view_specific.png'), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close(fig_xz)
    print("Saved XZ view plot within ROI.")

    # YZ Plane View (not rescaled, filtered to ROI)
    fig_yz, ax_yz = plt.subplots(figsize=(10, 5))
    sc_yz = ax_yz.scatter(y, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_yz.set_xlabel('Y [μm]', fontsize=fontsize_axis_title)
    ax_yz.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_yz, ratio=yz_aspect_ratio)
    fig_yz.savefig(os.path.join('./IMAGES/yz_view_specific.png'), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close(fig_yz)
    print("Saved YZ view plot within ROI.")

# Define output directory and run the plotting function with specified parameters
output_directory = 'path/to/output/directory'  # Specify the output directory
plot_3d_scatter_from_txt(input_file_path, output_directory, sphere_radius=0.0036, 
                         roi_center=roi_center, roi_diameter=roi_diameter, 
                         target_range=target_range, filter_radius=filtering_radius, 
                         min_neighbors=filtering_min_neighbours)
print("Processing finished")
