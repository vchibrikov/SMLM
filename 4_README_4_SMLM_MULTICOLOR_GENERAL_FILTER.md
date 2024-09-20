# SMLM_MULTICOLOR_GENERAL_FILTER.py
This Python project reads .csv data files, filters the data points based on neighborhood density, and generates a series of 2D and 3D scatterplots. It is designed to handle large datasets efficiently by processing data in chunks, using spatial trees for neighborhood queries, and saving visualizations in multiple formats.

- Visual Studio Code release used: 1.93.1
- Python release used: 3.12.4. 64-bit
> Warning! There are no guaranties this code will run on your machine.

## Features
- Reads .csv files containing spatial data (x, y, z coordinates).
- Filters data points based on the number of neighboring points within a specified radius.
- Generates and saves scatterplots in 2D and 3D formats.
- Supports filtering and visualizing two datasets together with customizable corrections for alignment.
- Outputs visualizations in multiple views (xy, xz, yz, 3D) as .png images.

## Dependencies
The following Python packages are required:
  - pandas
  - numpy
  - matplotlib
  - scipy

## Parameters
- xy_aspect_ratio = 1; ratio of x to y axes for 2D visualization.
- yz_aspect_ratio = 2; ratio of y to z axes for 2D visualization.
- xz_aspect_ratio = 2; ratio of x to z axes for 2D visualization.
- xyz_aspect_ratio = [1, 1, 0.01]; ratio of x to y and z axes for 3D visualization.
- dpi = 300; image quality parameter.
- fontsize_axis_title = 18; axis title fontsize.
- file1_path = '/Users/path/to/the/first/data/file.csv'; path to data file 1.
- file2_path = '/Users/path/to/the/second/data/file.csv'; path to data file 2.
- filtering_radius_1 = 0.2; radius used for filtering points of data file 1 based on their density.
- filtering_min_neighbours_1 = 10; minimum number of neighbors in data file 1 required for a point to be kept after filtering.
- filtering_radius_2 = 0.2; radius used for filtering points of data file 2 based on their density.
- filtering_min_neighbours_2 = 40; minimum number of neighbors in data file 2 required for a point to be kept after filtering.
- tag1 = 'Tag1'; axis title of file data 1.
- tag2 = 'Tag2'; axis title of file data 2.
- file2_corr_x = 0.0; x axis correction parameter of file data 2.
- file2_corr_y = 0.0; y axis correction parameter of file data 2.

## Description
Following script consist of several principle blocks of the code, which are explained below.

### Library imports and global variables
- pandas and numpy are used for data manipulation and numerical operations.
- matplotlib and mpl_toolkits.mplot3d are for creating 2D and 3D plots.
- Normalize from matplotlib.colors is used to normalize color scales in plots.
- os handles file and directory operations.
- cKDTree from scipy.spatial is a data structure for fast spatial searching, used for filtering points based on neighbors.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import os
from scipy.spatial import cKDTree
```

### Set plotting parameters
These variables define:
- Aspect ratios for different plot views (xy, yz, xz).
- DPI (dots per inch) for output image resolution.
- Font size for axis titles.
```
xy_aspect_ratio = 1
yz_aspect_ratio = 2
xz_aspect_ratio = 2
xyz_aspect_ratio = [1, 1, 0.01]
dpi = 300
fontsize_axis_title = 18
```

### Define filepaths and filtering parameters
- File paths and filtering parameters are set for two datasets (file1, file2).
- filtering_radius_1/2 and filtering_min_neighbours_1/2 control point filtering by distance and neighbor count.
- file2_corr_x/y are correction offsets for aligning the second dataset.
```
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
```

### Helper function to extract base name
This function extracts a base name from a filepath, combining parts of the directory and the file name.
```
def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name
```

### Set aspect ratio of a plot
Sets the aspect ratio for 2D plots by modifying the aspect of the axes (ax).
```
def set_aspect_ratio(ax, ratio=1):
    ax.set_aspect(ratio)
```

### Filter points based on neighbor density
Filters 3D points based on the density of neighboring points:
- Uses cKDTree to find neighboring points within a given radius.
- Filters out points with fewer neighbors than the min_neighbors threshold.
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

### Plotting and processing data
This is the main function that processes .csv files and generates various 2D and 3D scatter plots.
```
def plot_3d_scatter_from_csv(file1_path, file2_path=None, output_directory=None,
                             sphere_radius=0.05, 
                             filter_radius1=filtering_radius_1, min_neighbors1=filtering_min_neighbours_1,
                             filter_radius2=filtering_radius_2, min_neighbors2=filtering_min_neighbours_2):
```

### Directory setup and file loading
- Sets up the output directory structure.
- Loads .csv data from file1_path into a pandas dataframe (df1).
```
    base_input_directory = '/Users/path/to/input/directory'
    input_directory1 = os.path.dirname(file1_path)
    relative_path1 = os.path.relpath(input_directory1, start=base_input_directory)
    full_output_directory = os.path.join(output_directory, relative_path1)
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)

    df1 = pd.read_csv(file1_path, chunksize=100000)
    df1 = pd.concat(df1)
```

### Data preprocessing and filtering
- Extracts x, y, z coordinates from columns in the dataframe and scales them.
- Filters points using the filter_points function based on the radius and neighbor threshold.
```
    x1 = df1.iloc[:, 2].values / 1000
    y1 = df1.iloc[:, 3].values / 1000
    z1 = df1.iloc[:, 4].values / 1000
    
    x1, y1, z1 = filter_points(x1, y1, z1, radius=filter_radius1, min_neighbors=min_neighbors1)
```

###  Loading and filtering second dataset
If a second file is provided, loads and processes it similarly, applying any necessary x/y corrections.
```
    if file2_path:
        df2 = pd.read_csv(file2_path, chunksize=100000)
        df2 = pd.concat(df2)
        
        x2 = df2.iloc[:, 2].values / 1000 + file2_corr_x
        y2 = df2.iloc[:, 3].values / 1000 + file2_corr_y
        z2 = df2.iloc[:, 4].values / 1000
        
        x2, y2, z2 = filter_points(x2, y2, z2, radius=filter_radius2, min_neighbors=min_neighbors2)
```

###  Plotting
- Sets up the color normalization for z-values.
- Creates a figure and multiple subplots with 2D and 3D scatter plots, using plt.scatter() for each view (xy, xz, yz).
```
    norm1 = Normalize(vmin=z1.min(), vmax=z1.max())
    norm2 = Normalize(vmin=z2.min(), vmax=z2.max())
    
    sphere_area = np.pi * (sphere_radius * 1000) ** 2
    fig = plt.figure(figsize=(35, 20))
    spec = fig.add_gridspec(2, 2, height_ratios=[5, 1])
    ...
```

###  Saving the plots
Saves each plot to the output directory in different views (combined, xy, xz, yz, and 3D).
```
    fig.savefig(os.path.join(full_output_directory, f'{base_name1}_combined_views.png'), dpi=dpi, transparent=True)
```

###  Calling the function
Calls the plot_3d_scatter_from_csv function to generate and save the plots for the two datasets.
```
output_directory = '/Users/path/to/output/directory'
plot_3d_scatter_from_csv(file1_path, file2_path=file2_path, output_directory=output_directory,
                         sphere_radius=0.0006, filter_radius1=filtering_radius_1, min_neighbors1=filtering_min_neighbours_1,
                         filter_radius2=filtering_radius_2, min_neighbors2=filtering_min_neighbours_2)
```

## License
This project is licensed under the MIT License. See LICENSE file.
