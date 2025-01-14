# Import necessary libraries
import pandas as pd  # For data handling and CSV file processing
import numpy as np  # For numerical operations, including array manipulations
import matplotlib.pyplot as plt  # For creating visualizations and plots
from matplotlib.colors import Normalize  # For normalizing colormap range in plots
import os  # For file and directory path manipulation
from scipy.spatial import cKDTree  # For efficient spatial operations, such as finding neighbors within a radius

# Constants for configuration
xy_aspect_ratio = 1  # Aspect ratio for the XY plane view
yz_aspect_ratio = 2  # Aspect ratio for the YZ plane view
xz_aspect_ratio = 2  # Aspect ratio for the XZ plane view
xyz_aspect_ratio = [1, 1, 0.01]  # Aspect ratio for 3D visualization
dpi = 600  # Resolution (dots per inch) for saved plots
fontsize_axis_title = 18  # Font size for axis titles
input_file_path = '/path/to/data/your_data.csv'  # Path to the CSV input file containing data
filtering_radius_0 = 0.2  # Filtering radius for points flagged as 0
filtering_min_neighbours_0 = 10  # Minimum number of neighbors for points flagged as 0
filtering_radius_1 = 0.2  # Filtering radius for points flagged as 1
filtering_min_neighbours_1 = 10  # Minimum number of neighbors for points flagged as 1
xlim = (0, 10)  # X-axis limits for the plot
ylim = xlim  # Y-axis limits for the plot (set to be the same as xlim)

# Region of Interest (ROI) settings
roi_center = (12, 15)  # Center coordinates of the ROI
roi_diameter = 10.0  # Diameter of the ROI
target_range = (0, 10)  # The target range for rescaling the coordinates
sphere_radius = 0.0012  # Radius of the sphere to be plotted

def get_base_name(input_file_path):
    """
    Extracts the base name from the input file path.
    
    Parameters:
    input_file_path (str): The full path of the input file.
    
    Returns:
    str: The base name created from the directory and file name.
    """
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]  # Extract file name without extension
    directory_path = os.path.dirname(input_file_path)  # Extract the directory path
    directory_parts = directory_path.split(os.sep)  # Split path into directory components
    base_name = '_'.join(directory_parts[-3:] + [file_name])  # Create base name from the last 3 directory parts and file name
    return base_name

def set_aspect_ratio(ax, ratio=1):
    """Set aspect ratio of the plot."""
    ax.set_aspect(ratio)  # Adjust the plot's aspect ratio

def filter_points(x, y, z, radius, min_neighbors):
    """
    Filters points based on the density of neighboring points using a k-d tree.
    
    Parameters:
    x, y, z (np.ndarray): The x, y, and z coordinates of the points.
    radius (float): The radius within which neighbors are considered.
    min_neighbors (int): The minimum number of neighbors required to retain a point.
    
    Returns:
    np.ndarray: The filtered x, y, z coordinates.
    """
    print(f"Starting filtering points with radius = {radius} and min_neighbors = {min_neighbors}")
    points = np.column_stack((x, y, z))  # Combine x, y, and z coordinates into a single array
    tree = cKDTree(points)  # Create a k-d tree for efficient neighbor search
    
    # Find neighbors within the specified radius for each point
    neighbor_counts = tree.query_ball_point(points, r=radius)
    num_neighbors = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])  # Subtract 1 to exclude the point itself
    
    # Create mask to filter points with enough neighbors
    mask = num_neighbors >= min_neighbors
    print(f"Filtering complete. {np.sum(mask)} points retained out of {len(x)}.")
    return x[mask], y[mask], z[mask]

def apply_roi_scaling(x, y, roi_center, roi_diameter, target_range):
    """
    Apply ROI and rescale x and y coordinates.
    
    Parameters:
    x, y (np.ndarray): The x and y coordinates of the points.
    roi_center (tuple): The center of the ROI.
    roi_diameter (float): The diameter of the ROI.
    target_range (tuple): The target range for rescaling the coordinates.
    
    Returns:
    np.ndarray: Rescaled x and y coordinates within the ROI.
    """
    x_min, y_min = roi_center[0] - roi_diameter / 2, roi_center[1] - roi_diameter / 2
    x_max, y_max = roi_center[0] + roi_diameter / 2, roi_center[1] + roi_diameter / 2
    
    # Filter points within the ROI
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    x_roi, y_roi = x[mask], y[mask]
    
    # Rescale x, y coordinates based on the target range
    x_roi_rescaled = target_range[0] + (x_roi - x_min) / (x_max - x_min) * (target_range[1] - target_range[0])
    y_roi_rescaled = target_range[0] + (y_roi - y_min) / (y_max - y_min) * (target_range[1] - target_range[0])
    
    return x_roi_rescaled, y_roi_rescaled

def plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=0.05, filter_radius_0=0.1, filter_radius_1=0.2, min_neighbors_0=5, min_neighbors_1=10):
    """
    Processes the input CSV file, filters points based on the flag, applies ROI scaling, and generates scatter plots.
    
    Parameters:
    input_file_path (str): The path to the CSV input file.
    output_directory (str): The directory where output plots will be saved.
    sphere_radius (float): The radius of the spheres to be plotted.
    filter_radius_0, filter_radius_1 (float): Filtering radii for points flagged as 0 and 1.
    min_neighbors_0, min_neighbors_1 (int): Minimum number of neighbors for points flagged as 0 and 1.
    """
    print(f"Processing file: {input_file_path}")
    # Extract directory structure for the output path
    base_input_directory = '/path/to/data/'  # Base directory for input data
    input_directory = os.path.dirname(input_file_path)  # Extract input directory
    relative_path = os.path.relpath(input_directory, start=base_input_directory)  # Calculate relative path
    full_output_directory = output_directory  # Full path for output directory
    
    # Create output directory if it doesn't exist
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)
        print(f"Created output directory: {full_output_directory}")
    
    print("Loading data...")
    # Load CSV data into a dataframe
    df = pd.read_csv(input_file_path)  # Adjust to CSV format
    print(f"Data loaded. {len(df)} points found.")
    
    # Extract x, y, z columns from the data (assuming 0th, 1st, and 4th columns for x, y, z respectively)
    x = df.iloc[:, 0].values / 1000  # First column for x values (convert from nm to μm)
    y = df.iloc[:, 1].values / 1000  # Second column for y values (convert from nm to μm)
    z = df.iloc[:, 4].values / 1000  # Fourth column for z values (convert from nm to μm)
    
    # Flag values from column 8
    flag = df.iloc[:, 8].fillna(0).astype(int).values  # Flag values (1 or 0) from the 9th column
    
    # Filter points flagged as 0 based on neighbors and radius
    x_0, y_0, z_0 = filter_points(x[flag == 0], y[flag == 0], z[flag == 0], radius=filter_radius_0, min_neighbors=min_neighbors_0)
    
    # Filter points flagged as 1 based on neighbors and radius
    x_1, y_1, z_1 = filter_points(x[flag == 1], y[flag == 1], z[flag == 1], radius=filter_radius_1, min_neighbors=min_neighbors_1)
    
    # Apply ROI scaling for points flagged as 0 and 1
    x_0_rescaled, y_0_rescaled = apply_roi_scaling(x_0, y_0, roi_center, roi_diameter, target_range)
    x_1_rescaled, y_1_rescaled = apply_roi_scaling(x_1, y_1, roi_center, roi_diameter, target_range)

    # Ensure the size of z matches filtered data
    z_0_rescaled = z_0[:len(x_0_rescaled)]  # Rescale z_0 to match filtered data
    z_1_rescaled = z_1[:len(x_1_rescaled)]  # Rescale z_1 to match filtered data

    # Prepare the plot for XY view
    norm = Normalize(vmin=z.min(), vmax=z.max())  # Normalize the color scale for z-values
    sphere_area = np.pi * (sphere_radius * 1000) ** 2  # Calculate sphere area for plotting

    base_name = get_base_name(input_file_path)  # Get base name for saving the plot

    # Create the plot for the XY view
    fig_xy, ax_xy = plt.subplots(figsize=(15, 15))
    
    # Scatter plot for flag 0 points
    sc_xy_0 = ax_xy.scatter(x_0_rescaled, y_0_rescaled, c=z_0_rescaled, cmap='viridis', s=sphere_area, edgecolor='none', linewidth=0)
    
    # Scatter plot for flag 1 points
    sc_xy_1 = ax_xy.scatter(x_1_rescaled, y_1_rescaled, c=z_1_rescaled, cmap='Reds', s=sphere_area, edgecolor='none', linewidth=0)
    
    ax_xy.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xy.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    
    # Add colorbars
    cbar_0 = plt.colorbar(sc_xy_0, ax=ax_xy, label='Z [μm] (CBM3a)', shrink=0.7)
    cbar_1 = plt.colorbar(sc_xy_1, ax=ax_xy, label='Z [μm] (LM19)', shrink=0.7)

    # Apply limits for axes
    if xlim:
        ax_xy.set_xlim(xlim)
    if ylim:
        ax_xy.set_ylim(ylim)

    set_aspect_ratio(ax_xy, ratio=xy_aspect_ratio)  # Apply aspect ratio to the plot

    # Save the plot
    fig_xy.savefig(os.path.join(full_output_directory, 'xy_view.png'), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.tight_layout()  # Ensure tight layout after applying limits
    print("Saved XY view plot.")

    print("Processing finished.")

# Define output directory and input file path for running the function
output_directory = '/path/to/output/images/'
plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=sphere_radius, filter_radius_0=filtering_radius_0, filter_radius_1=filtering_radius_1, min_neighbors_0=filtering_min_neighbours_0, min_neighbors_1=filtering_min_neighbours_1)
