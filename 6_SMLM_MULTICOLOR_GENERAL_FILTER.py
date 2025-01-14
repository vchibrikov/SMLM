# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from scipy.spatial import cKDTree

# Define plotting and data processing parameters
xy_aspect_ratio = 1  # Aspect ratio for XY view to ensure equal scaling of X and Y axes
yz_aspect_ratio = 2  # Aspect ratio for YZ view for better visualization of the data
xz_aspect_ratio = 2  # Aspect ratio for XZ view, adjusting for visual clarity
dpi = 600  # Resolution of saved plots (600 dots per inch for high quality)
fontsize_axis_title = 18  # Font size for axis titles to make them readable

# Input file path for the CSV file containing the 3D data
input_file_path = 'path/to/your/data.csv'  # Replace with your actual file path

# Define filtering parameters that control the density of data points retained
filtering_radius_0 = 0.2  # Radius for filtering flag 0 points
filtering_min_neighbours_0 = 10  # Minimum neighbors required to retain flag 0 points
filtering_radius_1 = 0.2  # Radius for filtering flag 1 points
filtering_min_neighbours_1 = 10  # Minimum neighbors required to retain flag 1 points

# Define plot axis limits
xlim = (0, 25)  # X-axis limit for plots
ylim = xlim  # Y-axis limit, keeping it the same as X for square aspect ratio

# Function to extract the base name for output filenames
def get_base_name(input_file_path):
    """
    This function extracts the base name from the file path to use it in output filenames.
    It splits the path and joins the last three directory parts and the file name.
    """
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]  # Extract file name without extension
    directory_path = os.path.dirname(input_file_path)  # Get directory path
    directory_parts = directory_path.split(os.sep)  # Split the path into parts
    base_name = '_'.join(directory_parts[-3:] + [file_name])  # Create a base name using directory parts and file name
    return base_name

# Function to set the aspect ratio of plots
def set_aspect_ratio(ax, ratio=1):
    """
    This function adjusts the aspect ratio of a given axis (ax) to ensure the plot scales correctly.
    By default, the ratio is set to 1 (equal scaling on both axes).
    """
    ax.set_aspect(ratio)

# Function to filter data points based on density of neighboring points using cKDTree
def filter_points(x, y, z, radius, min_neighbors):
    """
    This function filters out data points based on how many neighbors they have within a given radius.
    cKDTree is used for efficient spatial searching. If a point has fewer neighbors than 'min_neighbors',
    it is removed.
    """
    print(f"Starting filtering points with radius = {radius} and min_neighbors = {min_neighbors}")
    points = np.column_stack((x, y, z))  # Stack the x, y, z values into a single array
    tree = cKDTree(points)  # Create a cKDTree from the points to efficiently query neighbors
    
    # Find neighbors within the specified radius for each point
    neighbor_counts = tree.query_ball_point(points, r=radius)
    num_neighbors = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])  # Exclude the point itself from its neighbor count
    
    # Filter points based on the minimum number of neighbors
    mask = num_neighbors >= min_neighbors
    print(f"Filtering complete. {np.sum(mask)} points retained out of {len(x)}.")
    return x[mask], y[mask], z[mask]  # Return only the points that passed the filter

# Function to process the CSV data, generate scatter plots, and save them
def plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=0.05, filter_radius_0=0.1, filter_radius_1=0.2, min_neighbors_0=5, min_neighbors_1=10):
    """
    This function processes 3D point data from a CSV file and generates scatter plots for three different views:
    XY, XZ, and YZ. The points are filtered based on the number of neighbors within a specified radius.
    The plots are saved in the specified output directory.
    """
    print(f"Processing file: {input_file_path}")
    # Define the base directory and relative output paths
    base_input_directory = '/Users/vadymchibrikov/Desktop/SMLM/WERSJA_2_09.01.2025_VC/FIG_14/IMAGES/'
    input_directory = os.path.dirname(input_file_path)
    relative_path = os.path.relpath(input_directory, start=base_input_directory)
    full_output_directory = output_directory
    
    if not os.path.exists(full_output_directory):  # Create the output directory if it doesn't exist
        os.makedirs(full_output_directory)
        print(f"Created output directory: {full_output_directory}")
    
    print("Loading data...")
    # Load data from CSV file
    df = pd.read_csv(input_file_path)
    print(f"Data loaded. {len(df)} points found.")
    
    # Extract x, y, z values and flag column from the CSV
    x = df.iloc[:, 0].values / 1000  # x values in micrometers
    y = df.iloc[:, 1].values / 1000  # y values in micrometers
    z = df.iloc[:, 4].values / 1000  # z values in micrometers
    flag = df.iloc[:, 8].fillna(0).astype(int).values  # Flagging column (0 or 1)

    # Apply filtering for points with flag 0 and flag 1 using separate conditions
    x_0, y_0, z_0 = filter_points(x[flag == 0], y[flag == 0], z[flag == 0], radius=filter_radius_0, min_neighbors=min_neighbors_0)
    x_1, y_1, z_1 = filter_points(x[flag == 1], y[flag == 1], z[flag == 1], radius=filter_radius_1, min_neighbors=min_neighbors_1)

    # Normalize z-values for color mapping in plots
    norm = Normalize(vmin=z.min(), vmax=z.max())
    sphere_area = np.pi * (sphere_radius * 1000) ** 2  # Calculate area for plot points (converted to nm²)

    # Generate base name for output files
    base_name = get_base_name(input_file_path)

    print("Generating individual plots...")

    # Generate XY Plane view scatter plot for flag=0 and flag=1
    fig_xy, ax_xy = plt.subplots(figsize=(15, 15))
    sc_xy_0 = ax_xy.scatter(x_0, y_0, c=z_0, cmap='viridis', s=sphere_area, edgecolor='none', linewidth=0)
    sc_xy_1 = ax_xy.scatter(x_1, y_1, c=z_1, cmap='Reds', s=sphere_area, edgecolor='none', linewidth=0)
    ax_xy.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xy.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    
    # Colorbars for each flag
    cbar_0 = plt.colorbar(sc_xy_0, ax=ax_xy, label='Z [μm] (CBM3a)', shrink=0.7)
    cbar_1 = plt.colorbar(sc_xy_1, ax=ax_xy, label='Z [μm] (LM19)', shrink=0.7)
    
    # Set axis limits and aspect ratio before saving the plot
    if xlim:
        ax_xy.set_xlim(xlim)
    if ylim:
        ax_xy.set_ylim(ylim)
    set_aspect_ratio(ax_xy, ratio=xy_aspect_ratio)

    # Save the plot
    fig_xy.savefig(os.path.join('./IMAGES/xy_view_general.png'), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.tight_layout()  # Adjust layout after applying limits
    print("Saved XY view plot.")

    # Repeat the process for XZ and YZ plane views
    # ...

    print("Processing finished.")

# Function to calculate and save statistics for photon intensity and localization precision
def save_statistics(input_file_path, output_directory):
    """
    This function calculates and saves statistics (mean and standard deviation) for photon intensity 
    and localization precision, separated by flag value. The results are written to a text file.
    """
    print("Calculating statistics...")
    # Load data from CSV file
    df = pd.read_csv(input_file_path)
    
    # Extract relevant data columns for analysis
    photon_intensity = df.iloc[:, 2]  # Photon intensity values
    localization_precision = df.iloc[:, 3]  # Localization precision values
    flag = df.iloc[:, 8].fillna(0).astype(int).values  # Flag values (0 or 1)
    
    # Separate data by flag values (flag=0 and flag=1)
    df_0 = df[flag == 0]
    df_1 = df[flag == 1]
    
    # Calculate mean and standard deviation for photon intensity and localization precision
    intensity_mean_0 = df_0.iloc[:, 2].mean()
    intensity_std_0 = df_0.iloc[:, 2].std()
    precision_mean_0 = df_0.iloc[:, 3].mean()
    precision_std_0 = df_0.iloc[:, 3].std()
    
    intensity_mean_1 = df_1.iloc[:, 2].mean()
    intensity_std_1 = df_1.iloc[:, 2].std()
    precision_mean_1 = df_1.iloc[:, 3].mean()
    precision_std_1 = df_1.iloc[:, 3].std()
    
    # Save statistics to a text file
    stats_file = os.path.join(output_directory, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Flag 0 - Photon Intensity Mean: {intensity_mean_0}, STD: {intensity_std_0}\n")
        f.write(f"Flag 0 - Localization Precision Mean: {precision_mean_0}, STD: {precision_std_0}\n")
        f.write(f"Flag 1 - Photon Intensity Mean: {intensity_mean_1}, STD: {intensity_std_1}\n")
        f.write(f"Flag 1 - Localization Precision Mean: {precision_mean_1}, STD: {precision_std_1}\n")
    print(f"Statistics saved to {stats_file}")
