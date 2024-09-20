# SMLM_SINGLECOLOR_GENERAL_FILTER.py
This Python script reads 3D point cloud data from a .csv file, filters the points based on density, and generates a series of scatterplots (xy, xz, yz, and 3D views). The filtered data is visualized in high-quality images and saved to an output directory.

- Visual Studio Code release used: 1.93.1
- Python release used: 3.12.4. 64-bit
> Warning! There are no guaranties this code will run on your machine.

## Features
- Point filtering: filters points based on density using a specified radius and minimum number of neighbors.
- 2D and 3D visualization: generates 3D scatterplots as well as xy, xz, and yz projections.
- Automatic output directory creation: creates the necessary directory structure for saving the plots based on the input file’s location.
- Customizable plot settings: adjustable aspect ratios, sphere radius for plot points, and other visual parameters.

## Dependencies
The following Python packages are required:
- numpy
- pandas
- matplotlib
- scipy

## Parameters
- input_file_path: path to the .csv file containing 3D point cloud data.
- output_directory: directory where the generated plots will be saved.
- sphere_radius: radius of the points in the scatter plot (default is 0.05).
- filtering_radius: radius used for filtering points based on their density.
- filtering_min_neighbours: minimum number of neighbors required for a point to be kept after filtering.
- xy_aspect_ratio: aspect ratio for XY view.
- xz_aspect_ratio: aspect ratio for XZ view.
- yz_aspect_ratio: aspect ratio for YZ view.
- xyz_aspect_ratio: aspect ratio for 3D plot (box aspect).

## Description
Following script consist of several principle blocks of the code, which are explained below.

### Library imports and global variables
This section imports the necessary libraries, including pandas for data handling, numpy for numerical operations, and matplotlib for visualization. Some global variables are also defined, such as aspect ratios for plots, DPI settings for image resolution, and font sizes for axis titles. It also defines parameters like the path to the .csv input file, filtering radius, and minimum neighbours.
 ```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import os
from scipy.spatial import cKDTree

xy_aspect_ratio = 1
yz_aspect_ratio = 2.5
xz_aspect_ratio = 2.5
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 300
fontsize_axis_title = 18

input_file_path = '/Users/path/to/data/file.csv/'
filtering_radius = 20
filtering_min_neighbours = 0.2
 ```

### Helper function to generate base name
This function generates a base name for saving output files. It uses the last three directories of the input file path and the file name (without extension) to create a unique identifier.
 ```
def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name
 ```

### Set aspect ratio function
A utility function to set the aspect ratio of a given axis ax in the plot. The default aspect ratio is 1, but it can be customized as needed.
```
def set_aspect_ratio(ax, ratio=1):
    """Set aspect ratio of the plot."""
    ax.set_aspect(ratio)
 ```

### Filtering points based on neighbor density
This function filters points based on the density of neighbouring points using a cKDTree for efficient nearest-neighbour queries. The points are kept if they have more than a specified number of neighbours within a given radius.
```
def filter_points(x, y, z, radius=filtering_radius, min_neighbors=filtering_min_neighbours):
    """Filter points based on the density of neighboring points."""
    print(f"Starting filtering points with radius = {radius} and min_neighbors = {min_neighbors}")
    points = np.column_stack((x, y, z))
    tree = cKDTree(points)
    
    neighbor_counts = tree.query_ball_point(points, r=radius)
    num_neighbors = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])
    
    mask = num_neighbors >= min_neighbors
    print(f"Filtering complete. {np.sum(mask)} points retained out of {len(x)}.")
    return x[mask], y[mask], z[mask]
 ```

### Function to generate 3D scatterplots
This function reads the .csv file containing 3D point data, filters the points using filter_points, and sets up the base structure for generating scatter plots. It also handles creating the appropriate output directory structure.
```
def plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=0.05, filter_radius=filtering_radius, min_neighbours=filtering_min_neighbours):
    print(f"Processing file: {input_file_path}")
    base_input_directory = '/Users/path/to/input/directory'
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
    
    x = df.iloc[:, 2].values / 1000 # or any column number with x coordinates in nm
    y = df.iloc[:, 3].values / 1000 # or any column number with y coordinates in nm
    z = df.iloc[:, 4].values / 1000 # or any column number with z coordinates in nm
    
    x, y, z = filter_points(x, y, z, radius=filter_radius, min_neighbors=min_neighbours)
    
    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = plt.get_cmap('viridis')
    sphere_area = np.pi * (sphere_radius * 1000) ** 2
    
    base_name = get_base_name(input_file_path)
 ```

### Generating xy, xz, yz, and 3D plots
This block generates four plots: xy view, 3D scatter view, xz view, and yz view. It organizes the plots in a 2x2 grid and saves the combined output as a single image.
```
    print("Generating plots...")
    fig = plt.figure(figsize=(35, 20))
    spec = fig.add_gridspec(2, 2, height_ratios=[5, 1])
    
    ax1 = fig.add_subplot(spec[0, 0])
    sc1 = ax1.scatter(x, y, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax1.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax1.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    cbar = plt.colorbar(sc1, ax=ax1, label='Z [μm]', shrink=0.7)
    cbar.set_label(label='Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax1, ratio=xy_aspect_ratio)
    
    ax2 = fig.add_subplot(spec[0, 1], projection='3d')
    sc2 = ax2.scatter(x, y, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax2.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax2.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    ax2.set_box_aspect(xyz_aspect_ratio)
    ax2.zaxis.set_ticks([])
    ax2.zaxis.set_ticklabels([])
    
    ax3 = fig.add_subplot(spec[1, 0])
    sc3 = ax3.scatter(x, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax3.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax3.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax3, ratio=xz_aspect_ratio)

    ax4 = fig.add_subplot(spec[1, 1])
    sc4 = ax4.scatter(y, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax4.set_xlabel('Y [μm]', fontsize=fontsize_axis_title)
    ax4.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax4, ratio=yz_aspect_ratio)
    
    plt.tight_layout()
    fig.savefig(os.path.join(full_output_directory, f'{base_name}_combined_views.png'), dpi=dpi, transparent=True)
    print("Saved combined views plot.")
 ```

### Generating and saving individual views
This block saves individual images for the xy, xz, yz, and 3D views of the scatterplots, each in their respective files with transparent backgrounds.
```
    fig_xy, ax_xy = plt.subplots(figsize=(15, 15))
    sc_xy = ax_xy.scatter(x, y, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_xy.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xy.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    cbar = plt.colorbar(sc_xy, ax=ax_xy, label='Z [μm]', shrink=0.7)
    cbar.set_label(label='Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_xy, ratio=xy_aspect_ratio)
    fig_xy.savefig(os.path.join(full_output_directory, f'{base_name}_xy_view.png'), dpi=dpi, transparent=True)
    print("Saved XY view plot.")

    fig_xz, ax_xz = plt.subplots(figsize=(15, 6))
    sc_xz = ax_xz.scatter(x, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_xz.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_xz.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_xz, ratio=xz_aspect_ratio)
    fig_xz.savefig(os.path.join(full_output_directory, f'{base_name}_xz_view.png'), dpi=dpi, transparent=True)
    print("Saved XZ view plot.")

    fig_yz, ax_yz = plt.subplots(figsize=(15, 6))
    sc_yz = ax_yz.scatter(y, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_yz.set_xlabel('Y [μm]', fontsize=fontsize_axis_title)
    ax_yz.set_ylabel('Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax_yz, ratio=yz_aspect_ratio)
    fig_yz.savefig(os.path.join(full_output_directory, f'{base_name}_yz_view.png'), dpi=dpi, transparent=True)
    print("Saved YZ view plot.")
    
    fig_3d = plt.figure(figsize=(15, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    sc_3d = ax_3d.scatter(x, y, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax_3d.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax_3d.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    ax_3d.set_box_aspect(xyz_aspect_ratio)
    ax_3d.zaxis.set_ticks([])
    ax_3d.zaxis.set_ticklabels([])
    fig_3d.savefig(os.path.join(full_output_directory, f'{base_name}_3d_plot.png'), dpi=dpi, transparent=True)
    print("Saved 3D plot.")
 ```

### Main call to process the file
This is the main function call that triggers the execution of the entire script. It processes the .csv file, generates the 3D scatterplots, and saves them to the specified output directory.
```
   output_directory = '/Users/path/to/output/directory'
plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=0.0006, filter_radius=filtering_radius, min_neighbors=filtering_min_neighbours)
 ```
 
## License
This project is licensed under the MIT License. See LICENSE file.
