import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import os
from scipy.spatial import cKDTree

xy_aspect_ratio = 1
yz_aspect_ratio = 2
xz_aspect_ratio = 2
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 300
fontsize_axis_title = 18

input_file_path = '/Users/path/to/data/file.csv/'
filtering_radius = 20
filtering_min_neighbours = 0.2 
yz_aspect_ratio = 2.5
xz_aspect_ratio = 2.5



def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name

def set_aspect_ratio(ax, ratio = 1):
    """Set aspect ratio of the plot."""
    ax.set_aspect(ratio)

def filter_points(x, y, z, radius = filtering_radius, min_neighbors = filtering_min_neighbours):
    """Filter points based on the density of neighboring points."""
    print(f"Starting filtering points with radius = {radius} and min_neighbors = {min_neighbors}")
    points = np.column_stack((x, y, z))
    tree = cKDTree(points)
    
    neighbor_counts = tree.query_ball_point(points, r=radius)
    num_neighbors = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])
    
    mask = num_neighbors >= min_neighbors
    print(f"Filtering complete. {np.sum(mask)} points retained out of {len(x)}.")
    return x[mask], y[mask], z[mask]

def plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius = 0.05, filter_radius = filtering_radius, min_neighbours = filtering_min_neighbours):
    print(f"Processing file: {input_file_path}")
    base_input_directory = '/Users/path/to/input/directory'
    input_directory = os.path.dirname(input_file_path)
    relative_path = os.path.relpath(input_directory, start = base_input_directory)
    full_output_directory = os.path.join(output_directory, relative_path)
    
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)
        print(f"Created output directory: {full_output_directory}")
    
    print("Loading data...")
    df = pd.read_csv(input_file_path, chunksize = 100000)
    df = pd.concat(df)
    print(f"Data loaded. {len(df)} points found.")
    
    x = df.iloc[:, 2].values / 1000
    y = df.iloc[:, 3].values / 1000
    z = df.iloc[:, 4].values / 1000
    
    x, y, z = filter_points(x, y, z, radius = filter_radius, min_neighbors = min_neighbors)
    
    norm = Normalize(vmin = z.min(), vmax = z.max())
    cmap = plt.get_cmap('viridis')
    sphere_area = np.pi * (sphere_radius * 1000) ** 2
    
    base_name = get_base_name(input_file_path)
    
    print("Generating plots...")
    fig = plt.figure(figsize = (35, 20))
    spec = fig.add_gridspec(2, 2, height_ratios = [5, 1])
    
    # XY Plane View (top left)
    ax1 = fig.add_subplot(spec[0, 0])
    sc1 = ax1.scatter(x, y, c = z, cmap = cmap, s = sphere_area, edgecolor = 'none', linewidth = 0)
    ax1.set_xlabel('X [μm]', fontsize = fontsize_axis_title)
    ax1.set_ylabel('Y [μm]', fontsize = fontsize_axis_title)
    cbar = plt.colorbar(sc1, ax = ax1, label = 'Z [μm]', shrink = 0.7)
    cbar.set_label(label = 'Z [μm]', fontsize = fontsize_axis_title)
    set_aspect_ratio(ax1, ratio = xy_aspect_ratio)
    
    ax2 = fig.add_subplot(spec[0, 1], projection = '3d')
    sc2 = ax2.scatter(x, y, z, c = z, cmap = cmap, s = sphere_area, edgecolor = 'none', linewidth = 0)
    ax2.set_xlabel('X [μm]', fontsize = fontsize_axis_title)
    ax2.set_ylabel('Y [μm]', fontsize = fontsize_axis_title)
    ax2.set_box_aspect(xyz_aspect_ratio)
    ax2.zaxis.set_ticks([])
    ax2.zaxis.set_ticklabels([])
    
    ax3 = fig.add_subplot(spec[1, 0])
    sc3 = ax3.scatter(x, z, c = z, cmap = cmap, s = sphere_area, edgecolor = 'none', linewidth = 0)
    ax3.set_xlabel('X [μm]', fontsize = fontsize_axis_title)
    ax3.set_ylabel('Z [μm]', fontsize = fontsize_axis_title)
    set_aspect_ratio(ax3, ratio = xz_aspect_ratio)

    ax4 = fig.add_subplot(spec[1, 1])
    sc4 = ax4.scatter(y, z, c = z, cmap = cmap, s = sphere_area, edgecolor = 'none', linewidth = 0)
    ax4.set_xlabel('Y [μm]', fontsize = fontsize_axis_title)
    ax4.set_ylabel('Z [μm]', fontsize = fontsize_axis_title)
    set_aspect_ratio(ax4, ratio = yz_aspect_ratio)
    
    plt.tight_layout()
    fig.savefig(os.path.join(full_output_directory, f'{base_name}_combined_views.png'), dpi = dpi, transparent = True)
    print("Saved combined views plot.")
    
    fig_xy, ax_xy = plt.subplots(figsize = (15, 15))
    sc_xy = ax_xy.scatter(x, y, c = z, cmap = cmap, s = sphere_area, edgecolor = 'none', linewidth = 0)
    ax_xy.set_xlabel('X [μm]', fontsize = fontsize_axis_title)
    ax_xy.set_ylabel('Y [μm]', fontsize = fontsize_axis_title)
    cbar = plt.colorbar(sc_xy, ax = ax_xy, label = 'Z [μm]', shrink = 0.7)
    cbar.set_label(label = 'Z [μm]', fontsize = fontsize_axis_title)
    set_aspect_ratio(ax_xy, ratio = xy_aspect_ratio)
    fig_xy.savefig(os.path.join(full_output_directory, f'{base_name}_xy_view.png'), dpi = dpi, transparent = True)
    print("Saved XY view plot.")

    fig_xz, ax_xz = plt.subplots(figsize = (15, 6))
    sc_xz = ax_xz.scatter(x, z, c = z, cmap = cmap, s = sphere_area, edgecolor = 'none', linewidth = 0)
    ax_xz.set_xlabel('X [μm]', fontsize = fontsize_axis_title)
    ax_xz.set_ylabel('Z [μm]', fontsize = fontsize_axis_title)
    set_aspect_ratio(ax_xz, ratio = xz_aspect_ratio)
    fig_xz.savefig(os.path.join(full_output_directory, f'{base_name}_xz_view.png'), dpi = dpi, transparent = True)
    print("Saved XZ view plot.")

    fig_yz, ax_yz = plt.subplots(figsize = (15, 6))
    sc_yz = ax_yz.scatter(y, z, c = z, cmap = cmap, s = sphere_area, edgecolor = 'none', linewidth = 0)
    ax_yz.set_xlabel('Y [μm]', fontsize = fontsize_axis_title)
    ax_yz.set_ylabel('Z [μm]', fontsize = fontsize_axis_title)
    set_aspect_ratio(ax_yz, ratio = yz_aspect_ratio)
    fig_yz.savefig(os.path.join(full_output_directory, f'{base_name}_yz_view.png'), dpi = dpi, transparent = True)
    print("Saved YZ view plot.")

    fig_3d = plt.figure(figsize = (15, 12))
    ax_3d = fig_3d.add_subplot(111, projection = '3d')
    sc_3d = ax_3d.scatter(x, y, z, c = z, cmap = cmap, s = sphere_area, edgecolor = 'none', linewidth = 0)
    ax_3d.set_xlabel('X [μm]', fontsize = fontsize_axis_title)
    ax_3d.set_ylabel('Y [μm]', fontsize = fontsize_axis_title)
    ax_3d.set_box_aspect(xyz_aspect_ratio)
    ax_3d.zaxis.set_ticks([])
    ax_3d.zaxis.set_ticklabels([])
    fig_3d.savefig(os.path.join(full_output_directory, f'{base_name}_3d_plot.png'), dpi = dpi, transparent = True)
    print("Saved 3D plot.")
    
    print("Processing finished")

output_directory = '/Users/path/to/output/directory'
plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius = 0.0006, filter_radius = filtering_radius, min_neighbors = filtering_min_neighbours)
