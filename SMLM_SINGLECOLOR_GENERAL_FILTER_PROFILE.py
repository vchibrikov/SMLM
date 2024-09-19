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

xy_aspect_ratio = 1
yz_aspect_ratio = 2
xz_aspect_ratio = 2
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 600
fontsize_axis_title = 16
num_segment = 30
fig_len = 5
fig_wid = 5

filtering_radius = 0.2
filtering_min_neighbours = 40
input_file_path = '/Users/input/file/path.csv'
lasso_thickness = 0.2

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
        self.lasso_coords = np.array(verts)
        if self.line is not None:
            self.line.remove()

        self.line = Line2D(self.lasso_coords[:, 0], self.lasso_coords[:, 1], color='red', linewidth=2)
        self.ax.add_line(self.line)

        self.calculate_lasso_length()

        if self.num_segments is not None and len(self.lasso_coords) > 1:
            self.interpolate_lasso_segments()

        self.split_lasso_and_count_points()

        self.canvas.draw_idle()

    def interpolate_lasso_segments(self):
        """Interpolate points along the lasso to achieve the desired number of segments."""
        if self.num_segments is None:
            return
        
        total_length = np.sum(np.linalg.norm(np.diff(self.lasso_coords, axis=0), axis=1))
        segment_length = total_length / self.num_segments

        distances = np.cumsum(np.linalg.norm(np.diff(self.lasso_coords, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)  # Add starting point

        interpolation = interp1d(distances, np.arange(len(self.lasso_coords)), kind='linear')
        new_distances = np.linspace(0, distances[-1], self.num_segments + 1)
        new_indices = interpolation(new_distances).astype(int)

        new_indices = np.clip(new_indices, 0, len(self.lasso_coords) - 1)
        self.lasso_coords = self.lasso_coords[new_indices]

    def calculate_lasso_length(self):
        """Calculate the total length of the drawn lasso line."""
        diffs = np.diff(self.lasso_coords, axis=0)
        self.segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(self.segment_lengths)
        print(f"Lasso Length: {total_length} μm")

    def split_lasso_and_count_points(self):
        """Split the lasso into segments and calculate the number of points near each segment."""
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

        self.segment_lengths = np.concatenate([
            [np.linalg.norm(self.lasso_coords[segment_indices[i + 1]] - self.lasso_coords[segment_indices[i]])
             for i in range(self.num_segments)]
        ])
        
        self.plot_profile()

    def points_within_thickness(self, segment_start, segment_end):
        """Find points within the lasso line's thickness (distance from the line)."""
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
        """Plot the profile: length of the lasso segment (x-axis) vs. points per segment (y-axis)."""
        cumulative_length = np.concatenate(([0], np.cumsum(self.segment_lengths)))

        if len(cumulative_length) != len(self.points_per_segment) + 1:
            print(f"Length mismatch: cumulative_length has {len(cumulative_length)} items, but points_per_segment has {len(self.points_per_segment)} items.")
            return  # Avoid plotting if there's a mismatch

        plt.figure(figsize=(fig_len, fig_wid))
        plt.plot(cumulative_length[:-1], self.points_per_segment,
                marker='o', color='green',
                linestyle='-', linewidth=2,
                markersize=8)

        plt.xlabel('Profile width [μm]', fontsize=fontsize_axis_title)
        plt.ylabel('Illumination events number', fontsize=fontsize_axis_title)

        if output_path is not None:
            plt.savefig(output_path, dpi=dpi, transparent=True)
            print(f"Profile plot saved at {output_path}")

        plt.show()


def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name


def set_aspect_ratio(ax, ratio=1):
    """Set aspect ratio of the plot."""
    ax.set_aspect(ratio)


def filter_points(x, y, z, radius=filtering_radius, min_neighbors=filtering_min_neighbours):
    """Filter points based on the density of neighboring points."""
    print(f"Starting filtering points with radius = {radius} and min_neighbors = {min_neighbors}")
    points = np.column_stack((x, y, z))
    tree = cKDTree(points)

    # Find neighbors within the specified radius for each point
    neighbor_counts = tree.query_ball_point(points, r=radius)
    num_neighbors = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])

    mask = num_neighbors >= min_neighbors
    print(f"Filtering complete. {np.sum(mask)} points retained out of {len(x)}.")
    return x[mask], y[mask], z[mask]


def plot_3d_scatter_with_lasso(input_file_path, output_directory, sphere_radius=0.05, filter_radius=0.1, min_neighbors=5, lasso_thickness=lasso_thickness):
    print(f"Processing file: {input_file_path}")
    base_input_directory = '/Users/path/to/input/five.csv'
    input_directory = os.path.dirname(input_file_path)
    relative_path = os.path.relpath(input_directory, start=base_input_directory)
    full_output_directory = os.path.join(output_directory, relative_path)

    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)
        print(f"Created output directory: {full_output_directory}")

    print("Loading data...")
    df = pd.read_csv(input_file_path, chunksize=100000)
    df = pd.concat(df)
    print(f"Data loaded. {len(df)} points found.")

    x = (df.iloc[:, 2].values / 1000)
    y = (df.iloc[:, 3].values / 1000)
    z = df.iloc[:, 4].values / 1000

    x, y, z = filter_points(x, y, z, radius = filter_radius, min_neighbors = min_neighbors)

    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = plt.get_cmap('viridis')
    sphere_area = np.pi * (sphere_radius * 1000) ** 2

    base_name = get_base_name(input_file_path)

    print("Generating scatter plot with Lasso Selector...")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)

    lasso_analyzer = LassoAnalyzer(ax, x, y, thickness=lasso_thickness, num_segments=num_segment)
    lasso = LassoSelector(ax, lasso_analyzer.onselect)

    plt.show()

    profile_output_path = os.path.join(full_output_directory, f'{base_name}_trans.png')
    lasso_analyzer.plot_profile(output_path=profile_output_path)


output_directory = '/Users/path/to/output/directory'
plot_3d_scatter_with_lasso(input_file_path, output_directory, sphere_radius=0.0006, filter_radius=filtering_radius, min_neighbors=filtering_min_neighbours, lasso_thickness=lasso_thickness)