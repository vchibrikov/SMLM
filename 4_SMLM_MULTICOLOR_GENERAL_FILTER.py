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

file1_path = '/Users/path/to/the/first/data/file.csv'
file2_path = '/Users/path/to/the/second/data/file.csv'
filtering_radius_1 = 0.2
filtering_min_neighbours_1 = 10
filtering_radius_2 = 0.2
filtering_min_neighbours_2 = 40
tag1 = 'Tag1'
tag2 = 'Tag2'
file2_corr_x = 0.0
file2_corr_y = 0.0


def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name

def set_aspect_ratio(ax, ratio=1):
    """Set aspect ratio of the plot."""
    ax.set_aspect(ratio)

def filter_points(x, y, z, radius, min_neighbors):
    """Filter points based on the density of neighboring points."""
    print(f"Starting filtering points with radius = {radius} and min_neighbors = {min_neighbors}")
    points = np.column_stack((x, y, z))
    tree = cKDTree(points)
    
    neighbor_counts = tree.query_ball_point(points, r=radius)
    num_neighbors = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])
    
    # Filter points based on the minimum number of neighbors
    mask = num_neighbors >= min_neighbors
    print(f"Filtering complete. {np.sum(mask)} points retained out of {len(x)}.")
    return x[mask], y[mask], z[mask]

def plot_3d_scatter_from_csv(file1_path, file2_path=None, output_directory=None,
                             sphere_radius=0.05, 
                             filter_radius1=filtering_radius_1, min_neighbors1=filtering_min_neighbours_1,
                             filter_radius2=filtering_radius_2, min_neighbors2=filtering_min_neighbours_2):
    print(f"Processing file 1: {file1_path}")
    base_input_directory = '/Users/path/to/input/directory'
    
    # Create output directory if it doesn't exist
    input_directory1 = os.path.dirname(file1_path)
    relative_path1 = os.path.relpath(input_directory1, start=base_input_directory)
    full_output_directory = os.path.join(output_directory, relative_path1)
    
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)
        print(f"Created output directory: {full_output_directory}")
    
    print("Loading data from file 1...")
    df1 = pd.read_csv(file1_path, chunksize=100000)
    df1 = pd.concat(df1)
    print(f"Data loaded from file 1. {len(df1)} points found.")
    
    x1 = df1.iloc[:, 2].values / 1000
    y1 = df1.iloc[:, 3].values / 1000
    z1 = df1.iloc[:, 4].values / 1000
    
    x1, y1, z1 = filter_points(x1, y1, z1, radius=filter_radius1, min_neighbors=min_neighbors1)
    
    if file2_path:
        print(f"Processing file 2: {file2_path}")

        print("Loading data from file 2...")
        df2 = pd.read_csv(file2_path, chunksize=100000)
        df2 = pd.concat(df2)
        print(f"Data loaded from file 2. {len(df2)} points found.")
        
        x2 = df2.iloc[:, 2].values / 1000 + file2_corr_x
        y2 = df2.iloc[:, 3].values / 1000 + file2_corr_y
        z2 = df2.iloc[:, 4].values / 1000
        
        x2, y2, z2 = filter_points(x2, y2, z2, radius=filter_radius2, min_neighbors=min_neighbors2)
    
    norm1 = Normalize(vmin=z1.min(), vmax=z1.max())
    norm2 = Normalize(vmin=z2.min(), vmax=z2.max())
    
    cmap1 = plt.get_cmap('viridis')
    cmap2 = plt.get_cmap('Reds')
    
    sphere_area = np.pi * (sphere_radius * 1000) ** 2
    
    base_name1 = get_base_name(file1_path)
    base_name2 = get_base_name(file2_path) if file2_path else "NoFile2"
    
    print("Generating plots...")
    fig = plt.figure(figsize=(35, 20))
    spec = fig.add_gridspec(2, 2, height_ratios=[5, 1])
    
    ax1 = fig.add_subplot(spec[0, 0])
    sc1 = ax1.scatter(x1, y1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 1 ({base_name1})')
    if file2_path:
        sc2 = ax1.scatter(x2, y2, c=z2, cmap=cmap2, norm=norm2, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 2 ({base_name2})')
    ax1.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax1.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    cbar1 = plt.colorbar(sc1, ax=ax1, label='Z [μm]', shrink=0.7)
    cbar1.set_label(label=f'Z [μm] ({tag1})', fontsize=fontsize_axis_title)
    if file2_path:
        cbar2 = plt.colorbar(sc2, ax=ax1, label='Z [μm]', shrink=0.7)
        cbar2.set_label(label=f'Z [μm] ({tag2})', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax1, ratio=xy_aspect_ratio)
    
    ax2 = fig.add_subplot(spec[0, 1], projection='3d')
    sc3d1 = ax2.scatter(x1, y1, z1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 1 ({base_name1})')
    if file2_path:
        sc3d2 = ax2.scatter(x2, y2, z2, c=z2, cmap=cmap2, norm=norm2, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 2 ({base_name2})')
    ax2.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax2.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    ax2.set_box_aspect(xyz_aspect_ratio)
    ax2.zaxis.set_ticks([])
    ax2.zaxis.set_ticklabels([])
    
    ax3 = fig.add_subplot(spec[1, 0])
    sc4 = ax3.scatter(x1, z1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 1 ({base_name1})')
    if file2_path:
        sc5 = ax3.scatter(x2, z2, c=z2, cmap=cmap2, norm=norm2, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 2 ({base_name2})')
    ax3.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax3.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax3, ratio=xz_aspect_ratio)

    ax4 = fig.add_subplot(spec[1, 1])
    sc6 = ax4.scatter(y1, z1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 1 ({base_name1})')
    if file2_path:
        sc7 = ax4.scatter(y2, z2, c=z2, cmap=cmap2, norm=norm2, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 2 ({base_name2})')
    ax4.set_xlabel('Y [μm]', fontsize=fontsize_axis_title)
    ax4.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax4, ratio=yz_aspect_ratio)
    
    plt.tight_layout()
    fig.savefig(os.path.join(full_output_directory, f'{base_name1}_combined_views.png'), dpi=dpi, transparent=True)
    print("Saved combined views plot.")
    
    fig_xy, ax_xy = plt.subplots(figsize=(15, 15))
    sc_xy1 = ax_xy.scatter(x1, y1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 1 ({base_name1})')
    if file2_path:
        sc_xy2 = ax_xy.scatter(x2, y2, c=z2, cmap=cmap2, norm=norm2, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 2 ({base_name2})')
    ax_xy.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xy.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    cbar_xy1 = plt.colorbar(sc_xy1, ax=ax_xy, label='Z [μm]', shrink=0.7)
    cbar_xy1.set_label(label=f'Z [μm] ({tag1})', fontsize=fontsize_axis_title)
    if file2_path:
        cbar_xy2 = plt.colorbar(sc_xy2, ax=ax_xy, label='Z [μm]', shrink=0.7)
        cbar_xy2.set_label(label=f'Z [μm] ({tag2})', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_xy, ratio=xy_aspect_ratio)
    fig_xy.savefig(os.path.join(full_output_directory, f'{base_name1}_xy_view.png'), dpi=dpi, transparent=True)
    print("Saved XY view plot.")

    fig_xz, ax_xz = plt.subplots(figsize=(15, 6))
    sc_xz1 = ax_xz.scatter(x1, z1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 1 ({base_name1})')
    if file2_path:
        sc_xz2 = ax_xz.scatter(x2, z2, c=z2, cmap=cmap2, norm=norm2, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 2 ({base_name2})')
    ax_xz.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xz.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_xz, ratio=xz_aspect_ratio)
    fig_xz.savefig(os.path.join(full_output_directory, f'{base_name1}_xz_view.png'), dpi=dpi, transparent=True)
    print("Saved XZ view plot.")

    fig_yz, ax_yz = plt.subplots(figsize=(15, 6))
    sc_yz1 = ax_yz.scatter(y1, z1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 1 ({base_name1})')
    if file2_path:
        sc_yz2 = ax_yz.scatter(y2, z2, c=z2, cmap=cmap2, norm=norm2, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 2 ({base_name2})')
    ax_yz.set_xlabel('Y [μm]', fontsize=fontsize_axis_title)
    ax_yz.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_yz, ratio=yz_aspect_ratio)
    fig_yz.savefig(os.path.join(full_output_directory, f'{base_name1}_yz_view.png'), dpi=dpi, transparent=True)
    print("Saved YZ view plot.")

    fig_3d = plt.figure(figsize=(15, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    sc_3d1 = ax_3d.scatter(x1, y1, z1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 1 ({base_name1})')
    if file2_path:
        sc_3d2 = ax_3d.scatter(x2, y2, z2, c=z2, cmap=cmap2, norm=norm2, s=sphere_area, edgecolor='none', linewidth=0, label=f'File 2 ({base_name2})')
    ax_3d.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_3d.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    ax_3d.set_box_aspect(xyz_aspect_ratio)
    ax_3d.zaxis.set_ticks([])
    ax_3d.zaxis.set_ticklabels([])
    fig_3d.savefig(os.path.join(full_output_directory, f'{base_name1}_3d_plot.png'), dpi=dpi, transparent=True)
    print("Saved 3D plot.")
    
    print("Processing finished")

output_directory = '/Users/path/to/output/directory'
plot_3d_scatter_from_csv(file1_path, file2_path=file2_path, output_directory=output_directory,
                         sphere_radius=0.0006, filter_radius1=filtering_radius_1, min_neighbors1=filtering_min_neighbours_1,
                         filter_radius2=filtering_radius_2, min_neighbors2=filtering_min_neighbours_2)