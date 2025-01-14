# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import os
from scipy.spatial import cKDTree
from matplotlib.widgets import LassoSelector
from matplotlib import path
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

# Explanation of the Script:
# This script processes 3D spatial data, filters points based on density, and enables user interaction 
# via a Lasso Selector tool to analyze regions of interest. It generates visualizations of the data
# and allows for profile analysis along the selected regions. Key features include filtering by density,
# interpolating lasso paths, and segmenting and counting points within user-defined regions.

# Constants for visualization and processing
xy_aspect_ratio = 1
yz_aspect_ratio = 2
xz_aspect_ratio = 2
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 600
fontsize_axis_title = 18
num_segment = 10
fig_len = 5
fig_wid = 5

# Input and output configuration
input_file_path = '/path/to/input/dataout1.txt'  # Replace with actual input path
output_directory = '/path/to/output/directory/'  # Replace with actual output directory
filtering_radius = 0.2
filtering_min_neighbours = 10
lasso_thickness = 0.1

# Class for handling lasso selection and related analysis
class LassoAnalyzer:
    def __init__(self, ax, x, y, thickness=lasso_thickness, num_segments=None):
        self.canvas = ax.figure.canvas
        self.ax = ax
        self.x = x
        self.y = y
        self.line = None
        self.lasso_coords = []
        self.segment_lengths = []
        self.points_per_segment = []
        self.thickness = thickness
        self.num_segments = num_segments

    def onselect(self, verts):
        """Handle lasso selection and perform analysis on selected points."""
        self.lasso_coords = np.array(verts)
        if self.line is not None:
            self.line.remove()

        # Draw the lasso line with specified thickness
        self.line = Line2D(self.lasso_coords[:, 0], self.lasso_coords[:, 1], color='green', linewidth=2)
        self.ax.add_line(self.line)

        # Calculate lasso length and analyze segments
        self.calculate_lasso_length()
        if self.num_segments is not None and len(self.lasso_coords) > 1:
            self.interpolate_lasso_segments()
        self.split_lasso_and_count_points()
        self.canvas.draw_idle()

    def interpolate_lasso_segments(self):
        """Interpolate points along the lasso path to define equal-length segments."""
        total_length = np.sum(np.linalg.norm(np.diff(self.lasso_coords, axis=0), axis=1))
        segment_length = total_length / self.num_segments

        distances = np.cumsum(np.linalg.norm(np.diff(self.lasso_coords, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
        interpolation = interp1d(distances, np.arange(len(self.lasso_coords)), kind='linear')
        new_distances = np.linspace(0, distances[-1], self.num_segments + 1)
        new_indices = interpolation(new_distances).astype(int)
        self.lasso_coords = self.lasso_coords[new_indices]

    def calculate_lasso_length(self):
        """Calculate the total length of the lasso."""
        diffs = np.diff(self.lasso_coords, axis=0)
        self.segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(self.segment_lengths)
        print(f"Lasso Length: {total_length} μm")

    def split_lasso_and_count_points(self):
        """Divide the lasso into segments and count points near each segment."""
        if self.num_segments is None or len(self.lasso_coords) < 2:
            return

        segment_indices = np.linspace(0, len(self.lasso_coords) - 1, self.num_segments + 1, dtype=int)
        self.points_per_segment = []
        for i in range(self.num_segments):
            segment_start = self.lasso_coords[segment_indices[i]]
            segment_end = self.lasso_coords[segment_indices[i + 1]]
            mask = self.points_within_thickness(segment_start, segment_end)
            points_in_segment = np.sum(mask)
            self.points_per_segment.append(points_in_segment)

    def points_within_thickness(self, segment_start, segment_end):
        """Identify points within a specified thickness around a segment."""
        segment_vector = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vector)
        segment_direction = segment_vector / segment_length if segment_length > 0 else np.zeros_like(segment_vector)
        point_vectors = np.column_stack((self.x, self.y)) - segment_start
        projections = np.dot(point_vectors, segment_direction)
        perpendicular_distances = np.linalg.norm(point_vectors - np.outer(projections, segment_direction), axis=1)
        within_bounds = (projections >= 0) & (projections <= segment_length)
        within_thickness = perpendicular_distances <= self.thickness
        return within_bounds & within_thickness

    def plot_profile(self, output_path=None):
        """Plot a profile of points per segment."""
        cumulative_length = np.concatenate(([0], np.cumsum(self.segment_lengths)))
        plt.figure(figsize=(fig_len, fig_wid))
        plt.plot(cumulative_length[:-1], self.points_per_segment, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel('Transverse profile [μm]', fontsize=fontsize_axis_title)
        plt.ylabel('Illumination events number', fontsize=fontsize_axis_title)
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi, transparent=True, bbox_inches='tight')
            print(f"Profile plot saved at {output_path}")
        plt.show()

# Additional utility functions (get_base_name, set_aspect_ratio, filter_points)

# Main function to process and visualize data
def plot_3d_scatter_with_lasso(input_file_path, output_directory, sphere_radius=0.05, filter_radius=0.1, min_neighbors=5, lasso_thickness=lasso_thickness):
    print(f"Processing file: {input_file_path}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Loading data...")
    df = pd.read_csv(input_file_path, sep='\t', header=None)
    x, y, z = df.iloc[:, 0].values / 1000, df.iloc[:, 1].values / 1000, df.iloc[:, 4].values / 1000
    x, y, z = filter_points(x, y, z, radius=filter_radius, min_neighbors=min_neighbors)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, c=z, cmap='viridis', s=(sphere_radius * 1000) ** 2, edgecolor='none')
    lasso_analyzer = LassoAnalyzer(ax, x, y, thickness=lasso_thickness, num_segments=num_segment)
    LassoSelector(ax, lasso_analyzer.onselect)
    plt.show()
