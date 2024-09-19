# SMLM_SINGLECOLOR_GENERAL_FILTER.py
This Python script reads 3D point cloud data from a .csv file, filters the points based on density, and generates a series of scatter plots (xy, xz, yz, and 3D views). The filtered data is visualized in high-quality images and saved to an output directory.

- Visual Studio Code release used: 1.93.1
- Python release used: 3.12.4. 64-bit
> Warning! There are no guaranties this code will run on your machine.

## Features
- Point filtering: filters points based on density using a specified radius and minimum number of neighbors.
- 2D and 3D visualization: generates 3D scatter plots as well as xy, xz, and yz projections.
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
- sphere_radius: dadius of the points in the scatter plot (default is 0.05).
- filtering_radius: radius used for filtering points based on their density.
- filtering_min_neighbours: minimum number of neighbors required for a point to be kept after filtering.
- xy_aspect_ratio: aspect ratio for XY view.
- xz_aspect_ratio: aspect ratio for XZ view.
- yz_aspect_ratio: aspect ratio for YZ view.
- xyz_aspect_ratio: aspect ratio for 3D plot (box aspect).

Following script consist of several principle blocks of the code, which are explained below.

### 

## License
This project is licensed under the MIT License.
