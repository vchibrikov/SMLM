# SMLM_MULTICOLOR_SPECIFIC_FILTER.py
This Python script loads two  .csv files, processes the data points by filtering them based on neighborhood density, and generates several 2D and 3D scatterplots. The plots visualize the relationship between x, y, and z coordinates across the dataset, with optional comparison between two datasets.

- Visual Studio Code release used: 1.93.1
- Python release used: 3.12.4. 64-bit
> Warning! There are no guaranties this code will run on your machine.

## Features
- Script handles large .csv files in chunks to optimize memory usage.
- Script filters points based on their neighborhood density using the cKDTree from scipy.spatial.
- Script limits the displayed data points to a user-defined ROI.
- Script generates multiple 2D views (xy, xz, yz) and a 3D plot.
- Script allows adjustment of filtering thresholds, plot size, color maps, and aspect ratios.
- Script supports two datasets processing: compares data from two .csv files, plotting them with different color schemes.

## Dependencies
- pandas: for reading and processing large .csv files.
- numpy: for efficient numerical operations.
- matplotlib: For generating 2D and 3D plots.
- scipy: for KDTree filtering of points based on proximity.
- mpl_toolkits.mplot3d: for 3D plotting.

## Parameters
- file1_path: Path to the first CSV file.
- file2_path: (Optional) Path to the second CSV file.
- output_directory: Directory where the output images will be saved.
- sphere_radius: The radius of each point in the scatter plot (default: 0.0001).
- filter_radius1 and filter_radius2: Radius within which to count neighbors for filtering data points.
- min_neighbors1 and min_neighbors2: Minimum number of neighbors required for a point to be kept.
- roi_center_x, roi_center_y: Coordinates for the center of the Region of Interest (ROI).
- roi_diameter: Diameter of the ROI.

## Description
Following script consist of several principle blocks of the code, which are explained below.

### Library imports
Libraries for data manipulation (pandas), numerical operations (numpy), plotting (matplotlib), handling 3D plots, normalization, and filesystem operations (os). Additionally, cKDTree from scipy.spatial is used for fast spatial queries and nearest-neighbor searches.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import os
from scipy.spatial import cKDTree
```

### Global parameters configuration
Libraries for data manipulation (pandas), numerical operations (numpy), plotting (matplotlib), handling 3D plots, normalization, and filesystem operations (os). Additionally, cKDTree from scipy.spatial is used for fast spatial queries and nearest-neighbor searches.
```
xy_aspect_ratio = 1
yz_aspect_ratio = 0.6
xz_aspect_ratio = 0.6
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 300
fontsize_axis_title = 18

filtering_radius_1 = 0.2
filtering_min_neighbours_1 = 10
filtering_radius_2 = 0.2
filtering_min_neighbours_2 = 40
file1_path = '/Users/path/to/the/first/data/file.csv'
file2_path = '/Users/path/to/the/second/data/file.csv'
tag1 = 'Tag1'
tag2 = 'Tag2'
file2_corr_x = 0.0
file2_corr_y = 0.0
roi_center_x = 5
roi_center_y = 5
roi_diameter = 5.0
sphere_radius1 = 0.0001
sphere_radius2 = 0.0001
```

### Global parameters configuration
- Parameter definitions: Configures global parameters for plotting (aspect ratios, DPI for saving images, font sizes) and filtering criteria (radius, minimum neighbors).
- File paths: Paths to the data files to be processed.
- Other variables: Correction factors, region-of-interest (ROI) center and diameter, and sphere radii for visualization.
```
xy_aspect_ratio = 1
yz_aspect_ratio = 0.6
xz_aspect_ratio = 0.6
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 300
fontsize_axis_title = 18

filtering_radius_1 = 0.2
filtering_min_neighbours_1 = 10
filtering_radius_2 = 0.2
filtering_min_neighbours_2 = 40
file1_path = '/Users/path/to/the/first/data/file.csv'
file2_path = '/Users/path/to/the/second/data/file.csv'
tag1 = 'Tag1'
tag2 = 'Tag2'
file2_corr_x = 0.0
file2_corr_y = 0.0
roi_center_x = 5
roi_center_y = 5
roi_diameter = 5.0
sphere_radius1 = 0.0001
sphere_radius2 = 0.0001
```

### Helper function - Extract file name and directory
Extracts the base file name and constructs a base name by including parts of the directory structure for use in naming output files.
```
def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name
```

### Helper function - Set plot aspect ratio
Configures the aspect ratio of a given axis to make plots visually proportional.
```
def set_aspect_ratio(ax, ratio=1):
    ax.set_aspect(ratio)
```

### Helper function - Filter points using nearest neighbors
Uses cKDTree for spatial indexing of 3D points. Filters out points that do not have enough neighboring points within a specified radius (min_neighbors). Returns only the points that meet the criteria.
```
def filter_points(x, y, z, radius, min_neighbors):
    print(f"Starting filtering points with radius = {radius} and min_neighbors = {min_neighbors}")
    points = np.column_stack((x, y, z))
    tree = cKDTree(points)
    
    neighbor_counts = tree.query_ball_point(points, r=radius)
    num_neighbors = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])
    
    mask = num_neighbors >= min_neighbors
    print(f"Filtering complete. {np.sum(mask)} points retained out of {len(x)}.")
    return x[mask], y[mask], z[mask]
```

### Main plotting function - File 1 processing
- Data Loading: loads data file in chunks and concatenates them into a single dataframe.
- Normalization: scales the data from micrometers to nanometers.
- Filtering: filters points based on density (using filter_points), then masks data within a defined region of interest (ROI).
 ```
def plot_3d_scatter_from_csv(file1_path, file2_path=None, output_directory=None, ...):
    print(f"Processing file 1: {file1_path}")
    
    input_directory1 = os.path.dirname(file1_path)
    full_output_directory = os.path.join(output_directory, relative_path1)

    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)
    
    df1 = pd.read_csv(file1_path, chunksize=100000)
    df1 = pd.concat(df1)
    x1 = df1.iloc[:, 2].values / 1000
    y1 = df1.iloc[:, 3].values / 1000
    z1 = df1.iloc[:, 4].values / 1000

    x1, y1, z1 = filter_points(x1, y1, z1, radius=filter_radius1, min_neighbors=min_neighbors1)

    mask_x = (x1 > roi_center_x - roi_diameter / 2) & (x1 < roi_center_x + roi_diameter / 2)
    mask_y = (y1 > roi_center_y - roi_diameter / 2) & (y1 < roi_center_y + roi_diameter / 2)
    
    combined_mask = mask_x & mask_y
    x1, y1, z1 = x1[combined_mask], y1[combined_mask], z1[combined_mask]
```

### Main plotting function - File 2 processing (optional)
- File 2 (optional): loads, filters, and masks data from a second file if provided.
 ```
    if file2_path:
        df2 = pd.read_csv(file2_path, chunksize=100000)
        df2 = pd.concat(df2)

        x2 = df2.iloc[:, 2].values / 1000 + file2_corr_x
        y2 = df2.iloc[:, 3].values / 1000 + file2_corr_y
        z2 = df2.iloc[:, 4].values / 1000

        x2, y2, z2 = filter_points(x2, y2, z2, radius=filter_radius2, min_neighbors=min_neighbors2)

        mask_x = (x2 > roi_center_x - roi_diameter / 2) & (x2 < roi_center_x + roi_diameter / 2)
        mask_y = (y2 > roi_center_y - roi_diameter / 2) & (y2 < roi_center_y + roi_diameter / 2)

        combined_mask = mask_x & mask_y
        x2, y2, z2 = x2[combined_mask], y2[combined_mask], z2[combined_mask]
```

### Plotting - 2D and 3D scatterplots
2D and 3D scatterplots: the function creates multiple scatter plots (2D and 3D) of the points, using color to represent the z-coordinate.
```
    fig = plt.figure(figsize=(35, 20))
    ax1 = fig.add_subplot(spec[0, 0])
    sc1 = ax1.scatter(x1, y1, c=z1, cmap=cmap1, norm=norm1, s=sphere_area, edgecolor='none')
    ax1.set_xlabel('X [μm]')
    ax1.set_ylabel('Y [μm]')
    plt.colorbar(sc1)
```

### Saving figures
File saving: each plot is saved as an image file (.png) in the output directory.
```
    fig.savefig(os.path.join(full_output_directory, f'{base_name1}_combined_views.png'), dpi=dpi, transparent=True)
```

### Final call
Function invocation: the script concludes by calling the plot_3d_scatter_from_csv function with specified parameters, paths, and output directories.
```
output_directory = '/Users/path/to/output/directory'
plot_3d_scatter_from_csv(file1_path, file2_path=file2_path, output_directory=output_directory, ...)
```

## License
This project is licensed under the MIT License. See LICENSE file.
