# Import necessary libraries
import pandas as pd  # For data handling
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
from matplotlib.widgets import LassoSelector  # For lasso selection in plots
from matplotlib.lines import Line2D  # For adding line objects
from scipy.interpolate import interp1d  # For interpolating lasso points
import os  # For file path and directory handling
from scipy.spatial import cKDTree  # For efficient spatial searches

# Constants for plot and filtering configuration
num_segment = 30  # Number of segments for lasso interpolation
lasso_thickness = 0.05  # Thickness of the lasso selection for point inclusion
dpi = 600  # Resolution of saved plots
fontsize_axis_title = 18  # Font size for axis titles
fig_len = 18  # Figure length for plots
fig_wid = 7  # Figure width for plots
xlim = (0, 20)  # X-axis limits for plotting
ylim = xlim  # Y-axis limits for plotting, equal to xlim
filtering_radius = 0.2  # Radius for spatial filtering of points
filtering_min_neighbours = 10  # Minimum number of neighbors to retain a point

class LassoAnalyzer:
    """
    This class handles the analysis of lasso selections, calculating segment-wise point counts and performing interpolation.
    """

    def __init__(self, ax, x, y, thickness=lasso_thickness, num_segments=num_segment):
        """
        Initializes the LassoAnalyzer instance.
        
        Parameters:
        ax (matplotlib.Axes): The axis object to plot the lasso selection.
        x (np.ndarray): The x-coordinates of the points.
        y (np.ndarray): The y-coordinates of the points.
        thickness (float): The thickness of the lasso selection.
        num_segments (int): The number of segments to divide the lasso into for analysis.
        """
        self.ax = ax
        self.x = x
        self.y = y
        self.thickness = thickness
        self.num_segments = num_segments
        self.line = None
        self.lasso_coords = []
        self.segment_lengths = []
        self.points_per_segment = []

    def onselect(self, verts):
        """
        Callback function triggered when a lasso selection is made.
        
        Parameters:
        verts (array): The vertices of the lasso selection.
        """
        self.lasso_coords = np.array(verts)  # Store the coordinates of the lasso
        if self.line is not None:
            self.line.remove()  # Remove the previous line if exists
        self.line = Line2D(self.lasso_coords[:, 0], self.lasso_coords[:, 1], color='blue', linewidth=2)  # Draw new line
        self.ax.add_line(self.line)  # Add the line to the plot
        self.calculate_segments()  # Calculate the uniform segments for the lasso selection
        self.ax.figure.canvas.draw_idle()  # Redraw the figure

    def calculate_segments(self):
        """
        Interpolates the lasso points into uniform segments for more granular analysis.
        """
        diffs = np.diff(self.lasso_coords, axis=0)  # Compute differences between consecutive lasso points
        segment_lengths = np.linalg.norm(diffs, axis=1)  # Calculate segment lengths
        total_length = np.sum(segment_lengths)  # Total length of the lasso path
        cumulative_lengths = np.cumsum(np.insert(segment_lengths, 0, 0))  # Cumulative length for each point
        interpolation = interp1d(cumulative_lengths, self.lasso_coords, axis=0, kind='linear')  # Interpolation function

        # Create uniform distances for the segments and clip to the valid range
        uniform_distances = np.linspace(0, total_length, self.num_segments + 1)
        uniform_distances = np.clip(uniform_distances, cumulative_lengths[0], cumulative_lengths[-1])

        self.lasso_coords = interpolation(uniform_distances)  # Interpolate lasso coordinates to the uniform distances

    def process_file(self, x, y):
        """
        Counts the number of points near each segment of the lasso selection.
        
        Parameters:
        x (np.ndarray): The x-coordinates of the points.
        y (np.ndarray): The y-coordinates of the points.
        
        Returns:
        list: The number of points per segment.
        """
        self.points_per_segment = []  # Initialize the list for storing point counts per segment
        for i in range(len(self.lasso_coords) - 1):
            start, end = self.lasso_coords[i], self.lasso_coords[i + 1]  # Get segment start and end points
            segment_vector = end - start  # Vector representing the segment direction
            segment_length = np.linalg.norm(segment_vector)  # Length of the segment
            segment_direction = segment_vector / segment_length if segment_length > 0 else np.zeros_like(segment_vector)  # Normalized direction
            points = np.column_stack((x, y))  # Stack the x and y coordinates into a single array
            rel_points = points - start  # Translate points to the segment start
            projections = np.dot(rel_points, segment_direction)  # Project points onto the segment direction
            distances = np.linalg.norm(rel_points - projections[:, None] * segment_direction, axis=1)  # Calculate perpendicular distances
            mask = (0 <= projections) & (projections <= segment_length) & (distances <= self.thickness)  # Create mask for points within the segment
            self.points_per_segment.append(np.sum(mask))  # Count points in the segment
        return self.points_per_segment

def get_base_name(path):
    """
    Extracts the base name (file name without extension) from the file path.
    
    Parameters:
    path (str): The file path.
    
    Returns:
    str: The base name of the file.
    """
    return os.path.splitext(os.path.basename(path))[0]

def filter_points(x, y, z, radius=filtering_radius, min_neighbors=filtering_min_neighbours):
    """
    Filters points based on their spatial density.
    
    Parameters:
    x (np.ndarray): The x-coordinates of the points.
    y (np.ndarray): The y-coordinates of the points.
    z (np.ndarray): The z-coordinates of the points.
    radius (float): The radius within which to count neighbors.
    min_neighbors (int): Minimum number of neighbors to retain a point.
    
    Returns:
    tuple: The filtered x, y, and z coordinates of the points.
    """
    points = np.column_stack((x, y, z))  # Stack x, y, z into a single array
    tree = cKDTree(points)  # Create a k-d tree for efficient neighbor queries
    counts = [len(tree.query_ball_point(p, radius)) for p in points]  # Count neighbors within the radius for each point
    mask = np.array(counts) >= min_neighbors  # Mask points that have enough neighbors
    return x[mask], y[mask], z[mask]  # Return the filtered points

def plot_profiles(lasso_profiles, output_path=None):
    """
    Plots the combined lasso profiles for multiple files.
    
    Parameters:
    lasso_profiles (dict): Dictionary of lasso profiles where the key is the file name and the value is a tuple of lengths and counts.
    output_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(fig_len, fig_wid))  # Create a figure
    for label, (lengths, counts) in lasso_profiles.items():
        plt.plot(lengths, counts, marker='o', label=label)  # Plot each profile
    plt.xlabel('Longitudinal profile [μm]', fontsize=fontsize_axis_title)  # Label the x-axis
    plt.ylabel('Illumination events number', fontsize=fontsize_axis_title)  # Label the y-axis
    plt.legend(fontsize=fontsize_axis_title)  # Add a legend
    plt.tight_layout()  # Adjust layout
    if output_path:
        plt.savefig(output_path, dpi=dpi, transparent=True, bbox_inches='tight')  # Save the plot if output path is provided
    plt.show()  # Display the plot

def plot_3d_scatter_with_lasso(input_files, output_dir):
    """
    Plots a 3D scatter plot and allows lasso selection. The selected lasso regions are analyzed across multiple files.
    
    Parameters:
    input_files (list): List of input file paths (CSV format).
    output_dir (str): The directory to save output plots.
    """
    lasso_profiles = {}  # Dictionary to store the lasso profiles for each file
    first_file = input_files[0]
    df = pd.read_csv(first_file, header=None)  # Read the first input file
    x, y, z = df.iloc[:, 0] / 1000, df.iloc[:, 1] / 1000, df.iloc[:, 4] / 1000  # Convert coordinates from nm to μm
    x, y, z = filter_points(x, y, z)  # Filter the points based on spatial density

    # Set up the scatter plot and lasso selection
    fig, ax = plt.subplots(figsize=(9, 9))  # Create a figure and axis
    scatter = ax.scatter(x, y, c=z, cmap='viridis', s=1)  # Plot the scatter points
    ax.set_xlim(xlim)  # Set x-axis limits
    ax.set_ylim(ylim)  # Set y-axis limits
    lasso_analyzer = LassoAnalyzer(ax, x, y)  # Initialize the lasso analyzer
    lasso = LassoSelector(ax, lasso_analyzer.onselect)  # Set up the lasso selector
    plt.show()  # Show the plot with lasso functionality

    # Process additional files using the same lasso
    for input_file in input_files:
        df = pd.read_csv(input_file, header=None)  # Read the data from the current file
        x, y, z = df.iloc[:, 0] / 1000, df.iloc[:, 1] / 1000, df.iloc[:, 4] / 1000  # Convert coordinates to μm
        x, y, z = filter_points(x, y, z)  # Filter points
        points_per_segment = lasso_analyzer.process_file(x, y)  # Get the point count for each segment
        cumulative_lengths = np.cumsum([0] + [np.linalg.norm(lasso_analyzer.lasso_coords[i + 1] - lasso_analyzer.lasso_coords[i])
                                              for i in range(len(lasso_analyzer.lasso_coords) - 1)])  # Compute cumulative segment lengths
        lasso_profiles[get_base_name(input_file)] = (cumulative_lengths[:-1], points_per_segment)  # Store the profile

    # Plot and save the combined lasso profiles
    output_path = os.path.join(output_dir, 'combined_lasso_profiles.png')
    plot_profiles(lasso_profiles, output_path)

# Example usage
input_files = [
    'path_to_input_file_1.csv',  # Replace with actual file paths
    'path_to_input_file_2.csv'  # Replace with actual file paths
]
output_dir = 'path_to_output_directory'  # Replace with actual output directory
plot_3d_scatter_with_lasso(input_files, output_dir)
