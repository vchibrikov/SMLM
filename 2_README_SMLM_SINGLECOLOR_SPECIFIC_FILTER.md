# SMLM_SINGLECOLOR_SPECIFIC_FILTER.py
This Python script processes a .csv file containing 3D coordinate data and generates multiple 2D and 3D scatter plots, visualizing the data from different perspectives. The script allows for filtering of points based on neighborhood density and normalizes the data for display within a specific target range. The aspect ratios of the plots can also be customized, and the outputs are saved as high-resolution images.

- Visual Studio Code release used: 1.93.1
- Python release used: 3.12.4. 64-bit
> Warning! There are no guaranties this code will run on your machine.

## Features
- Data input: reads large .csv files in chunks for efficient memory usage.
- Region of interest (ROI): selects data points from a specified circular region of interest on the xy plane.
- Filtering: filters points based on the number of neighbors within a specified radius using a KDTree spatial algorithm.
- Aspect ratio customization: custom aspect ratios can be set for the generated plots.
- Plot types: combined 2D and 3D views; separate 2D views of xy, xz, and yz planes; 3D scatter plots with customizable viewing angles.
- Output: high-quality, transparent .png images.

## Dependencies
- Python 3.6+
- pandas: for reading and processing large .csv files.
- numpy: for efficient numerical operations.
- matplotlib: For generating 2D and 3D plots.
- scipy: for KDTree filtering of points based on proximity.
- mpl_toolkits.mplot3d: for 3D plotting.

## Parameters
- input_file_path: path to the input .csv file.
- output_directory: directory to save the output images.
- roi_center: center of the region of interest in the xy plane.
- roi_diameter: diameter of the region of interest.
- target_range: range to which the x and y coordinates are normalized.
- filtering_radius: radius within which neighbors are counted for filtering points.
- filtering_min_neighbours: minimum number of neighbors a point must have to be included in the plot.
- sphere_radius: radius of the plotted points in micrometers (adjusted for scaling).
- xy_aspect_ratio = 1
- yz_aspect_ratio = 0.5
- xz_aspect_ratio = 0.5
- xyz_aspect_ratio = [1, 1, 0.01]
- dpi: resolution for saved images.
- fontsize_axis_title: font size for axis titles.

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
 ```

### Base name extraction
This function extracts the base name for saving files. It constructs a unique name using the last three directories of the input path and the .csv file name (without its extension). This is used for naming output files.
```
def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name
 ```

### Set aspect ratio for 2D plot
This function sets the aspect ratio of a given axis (ax) in a 2D plot. It’s a utility to control how the axes are scaled visually.
```
def set_aspect_ratio(ax, ratio=1):
    """Set aspect ratio of the plot."""
    ax.set_aspect(ratio)
 ```

###  Filter points by density of neighbors
This function filters points based on the density of neighboring points within a specified radius (filtering_radius) using a KDTree for fast neighbor searching. It returns only the points that have at least filtering_min_neighbours neighbors within the radius.
```
def filter_points(x, y, z, filtering_radius, filtering_min_neighbours):
    points = np.vstack((x, y, z)).T
    tree = KDTree(points)
    neighbor_counts = tree.query_ball_point(points, r=filtering_radius)
    mask = np.array([len(neighbors) >= filtering_min_neighbours for neighbors in neighbor_counts])
    return x[mask], y[mask], z[mask]
 ```

###  Main plotting function
```
def plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=sphere_radius1, roi_center=(0, 0), roi_diameter=1, 
                             target_range=(0, 5), filtering_radius=filtering_radius, filtering_min_neighbours=filtering_min_neighbours):
 ```

###  Extract base name and prepare output directory
This block sets up the output directory structure based on the input file’s relative path. If the necessary directories do not exist, it creates them.
```
    base_input_directory = '/Users/path/to/input/directory'
    input_directory = os.path.dirname(input_file_path)
    relative_path = os.path.relpath(input_directory, start=base_input_directory)
    full_output_directory = os.path.join(output_directory, relative_path)
    
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)
 ```

###  Read and filter the data
The code reads the input .csv file into a pandas DataFrame, possibly in chunks (for large files), and concatenates them. It extracts the x, y, and z coordinates (assumed to be in the 3rd, 4th, and 5th columns, respectively) and scales them down to nanometers (by dividing by 1000).
 ```
    df = pd.read_csv(input_file_path, chunksize=100000)
    df = pd.concat(df)
    
    x = df.iloc[:, 2].values / 1000
    y = df.iloc[:, 3].values / 1000
    z = df.iloc[:, 4].values / 1000
 ```

###  Normalize z-values and define ROI
The z-values are normalized for color mapping using the normalize function. cmap is set to the 'viridis' colormap. Region of interest (ROI): The code defines a square region of interest (ROI) based on the roi_center and roi_diameter. It creates a mask to filter out points that fall within this ROI.
```
    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = plt.get_cmap('viridis')
    sphere_area = np.pi * (sphere_radius * 1000) ** 2
    
    base_name = get_base_name(input_file_path)
    
    center_x, center_y = roi_center
    diameter = roi_diameter
    half_side = diameter / 2
    roi = [center_x - half_side, center_x + half_side, center_y - half_side, center_y + half_side]
    
    x_masked = (x >= roi[0]) & (x <= roi[1])
    y_masked = (y >= roi[2]) & (y <= roi[3])
    mask = x_masked & y_masked
    
    x, y, z = x[mask], y[mask], z[mask]
 ```

### Apply neighbor filtering and normalize coordinates
The filter_points function is applied to further filter points based on neighbor density. The x and y coordinates are normalized to fall within the specified target_range (default: 0 to 5).
```
    x, y, z = filter_points(x, y, z, filtering_radius, filtering_min_neighbours)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_normalized = ((x - x_min) / (x_max - x_min)) * (target_range[1] - target_range[0]) + target_range[0]
    y_normalized = ((y - y_min) / (y_max - y_min)) * (target_range[1] - target_range[0]) + target_range[0]
 ```

### Generate combined 2D and 3D views
This part creates the main figure and subplots layout using a grid specification. The first plot (ax1) is a 2D scatterplot for the xy plane, with colors representing the z-values.
```
    fig = plt.figure(figsize=(35, 20))
    spec = fig.add_gridspec(2, 2, height_ratios=[6, 3])
    
    ax1 = fig.add_subplot(spec[0, 0])
    sc1 = ax1.scatter(x_normalized, y_normalized, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax1.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax1.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    cbar = plt.colorbar(sc1, ax=ax1, label='Z [μm]', shrink=0.7)
    cbar.set_label(label='Z [μm]', fontsize=fontsize_axis_title)
    set_aspect_ratio(ax1, ratio=xy_aspect_ratio)
 ```

### Generate 3D scatterplot
This subplot generates a 3D scatterplot of the points in the x, y, and z coordinates, with z-values dictating the color.
```
    ax2 = fig.add_subplot(spec[0, 1], projection='3d')
    sc2 = ax2.scatter(x_normalized, y_normalized, z, c=z, cmap=cmap, s=sphere_area, edgecolor='none', linewidth=0)
    ax2.set_xlabel('X [μm]', fontsize=fontsize_axis_title)
    ax2.set_ylabel('Y [μm]', fontsize=fontsize_axis_title)
    ax2.set_box_aspect(xyz_aspect_ratio)
    ax2.zaxis.set_ticks([])
    ax2.zaxis.set_ticklabels([])
 ```

### Other projections (xz and yz)
These two subplots display projections in the xz and yz planes, respectively.
```
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
 ```

### Save the combined plot
The combined figure is saved as a .png file in the specified output directory.
```
    plt.tight_layout()
    fig.savefig(os.path.join(full_output_directory, f'{base_name}_combined_views.png'), dpi=dpi, transparent=True)
 ```

### Generate and save individual view plots
This block creates and saves individual views of the xy, xz, yz, and 3D projections as separate .png files.
```
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
 ```

### Final call to execute
This block creates and saves individual views of the zy, xz, yz, and 3D projections as separate .png files.
```
output_directory = '/Users/path/to/output/directory'
plot_3d_scatter_from_csv(input_file_path, output_directory, sphere_radius=sphere_radius2, roi_center=roi_center, roi_diameter=roi_diameter, target_range=target_range, filtering_radius=filtering_radius, filtering_min_neighbours=filtering_min_neighbours)
print("Processing finished")
 ```

## License
This project is licensed under the MIT License. See LICENSE file.
