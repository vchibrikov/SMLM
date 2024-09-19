import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.spatial import KDTree
import os

# Aspect ratios and figure settings
xy_aspect_ratio = 1
yz_aspect_ratio = 0.5
xz_aspect_ratio = 0.5
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 300
fontsize_axis_title = 18

input_file_path = '/Users/path/to/input/file.csv'
roi_center = (15, 15)
roi_diameter = 10.0
target_range = (0, 5)
filtering_radius = 0.2
filtering_min_neighbours = 80

def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name

def set_aspect_ratio(ax, ratio=1):
    """Set aspect ratio of the plot."""
    ax.set_aspect(ratio)

def filter_points(x, y, z, filtering_radius, filtering_min_neighbours):
    points = np.vstack((x, y, z)).T
    
    tree = KDTree(points)
    
    neighbor_counts = tree.query_ball_point(points, r=filtering_radius)
    
    mask = np.array([len(neighbors) >= filtering_min_neighbours for neighbors in neighbor_counts])
    
    return x[mask], y[mask], z[mask]

def plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=sphere_radius1, roi_center=(0, 0), roi_diameter=1, 
                             target_range=(0, 5), filtering_radius=filtering_radius, filtering_min_neighbours=filtering_min_neighbours):
    # Extract base name and create output directory
    base_input_directory = '/Users/path/to/input/directory'
    input_directory = os.path.dirname(input_file_path)
    relative_path = os.path.relpath(input_directory, start=base_input_directory)
    full_output_directory = os.path.join(output_directory, relative_path)
    
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)
    
    df = pd.read_csv(input_file_path, chunksize=100000)
    df = pd.concat(df)
    
    x = df.iloc[:, 2].values / 1000
    y = df.iloc[:, 3].values / 1000
    z = df.iloc[:, 4].values / 1000
    
    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = plt.get_cmap('viridis')
    sphere_area = np.pi * (sphere_radius * 1000) ** 2
    
    base_name = get_base_name(input_file_path)
    
    center_x, center_y = roi_center
    diameter = roi_diameter
    half_side = diameter / 2
    roi = [center_x - half_side, center_x + half_side, center_y - half_side, center_y + half_side]
    print(f"Defined ROI: {roi}")

    x_masked = (x >= roi[0]) & (x <= roi[1])
    y_masked = (y >= roi[2]) & (y <= roi[3])
    mask = x_masked & y_masked
    
    x, y, z = x[mask], y[mask], z[mask]

    x, y, z = filter_points(x, y, z, filtering_radius, filtering_min_neighbours)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_normalized = ((x - x_min) / (x_max - x_min)) * (target_range[1] - target_range[0]) + target_range[0]
    y_normalized = ((y - y_min) / (y_max - y_min)) * (target_range[1] - target_range[0]) + target_range[0]
    
    fig = plt.figure(figsize=(35, 20))
    spec = fig.add_gridspec(2, 2, height_ratios=[6 , 3])
    
    ax1 = fig.add_subplot(spec[0, 0])
    sc1 = ax1.scatter(x_normalized, y_normalized, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax1.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax1.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    cbar = plt.colorbar(sc1, ax=ax1, label='Z [μm]', shrink=0.7)
    cbar.set_label(label='Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax1, ratio=xy_aspect_ratio)
    
    ax2 = fig.add_subplot(spec[0, 1], projection='3d')
    sc2 = ax2.scatter(x_normalized, y_normalized, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax2.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax2.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    ax2.set_box_aspect(xyz_aspect_ratio)
    ax2.zaxis.set_ticks([])
    ax2.zaxis.set_ticklabels([])
    
    ax3 = fig.add_subplot(spec[1, 0])
    sc3 = ax3.scatter(x_normalized, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax3.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax3.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax3, ratio=xz_aspect_ratio)

    ax4 = fig.add_subplot(spec[1, 1])
    sc4 = ax4.scatter(y_normalized, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax4.set_xlabel('Y [μm]', fontsize=fontsize_axis_title)
    ax4.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax4, ratio=yz_aspect_ratio)
    
    plt.tight_layout()
    fig.savefig(os.path.join(full_output_directory, f'{base_name}_combined_views.png'), dpi=dpi, transparent=True)
        
    fig_xy, ax_xy = plt.subplots(figsize=(15, 15))
    sc_xy = ax_xy.scatter(x_normalized, y_normalized, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_xy.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xy.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    cbar = plt.colorbar(sc_xy, ax=ax_xy, label='Z [μm]', shrink=0.7)
    cbar.set_label(label='Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_xy, ratio=xy_aspect_ratio)
    fig_xy.savefig(os.path.join(full_output_directory, f'{base_name}_xy_view.png'), dpi=dpi, transparent=True)

    fig_xz, ax_xz = plt.subplots(figsize=(15, 6))
    sc_xz = ax_xz.scatter(x_normalized, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_xz.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xz.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_xz, ratio=xz_aspect_ratio)
    fig_xz.savefig(os.path.join(full_output_directory, f'{base_name}_xz_view.png'), dpi=dpi, transparent=True)

    fig_yz, ax_yz = plt.subplots(figsize=(15, 6))
    sc_yz = ax_yz.scatter(y_normalized, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_yz.set_xlabel('Y [μm]', fontsize=fontsize_axis_title)
    ax_yz.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_yz, ratio=yz_aspect_ratio)
    fig_yz.savefig(os.path.join(full_output_directory, f'{base_name}_yz_view.png'), dpi=dpi, transparent=True)

    fig_3d = plt.figure(figsize=(15, 15))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    sc_3d = ax_3d.scatter(x_normalized, y_normalized, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_3d.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_3d.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    ax_3d.set_box_aspect(xyz_aspect_ratio)
    ax_3d.zaxis.set_ticks([])
    ax_3d.zaxis.set_ticklabels([])
    fig_3d.savefig(os.path.join(full_output_directory, f'{base_name}_3d_plot.png'), dpi=dpi, transparent=True)
    
    print("Plots saved successfully.")

output_directory = '/Users/path/to/output/directory'

plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=sphere_radius2, roi_center=roi_center, roi_diameter=roi_diameter, target_range=target_range, filtering_radius=filtering_radius, filtering_min_neighbours=filtering_min_neighbours)
print("Processing finished")
