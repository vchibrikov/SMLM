# SMLM_SINGLECOLOR_GENERAL_FILTER.py**
This Python script processes 3D data from a .csv file, applies spatial filtering to remove outliers based on the density of neighboring points, and generates interactive scatterplots. It features a Lasso selection tool for segmenting points within user-defined paths and visualizes profiles based on the selected data. The final plots are saved to a specified output directory.

- Visual Studio Code release used: 1.93.1
- Python release used: 3.12.4. 64-bit
> Warning! There are no guaranties this code will run on your machine.

## Features
- 3D scatter plot visualization: Displays points in a 2D scatter plot with z-values as color and allows for interaction.
- Lasso selection: Allows manual selection of a subset of points on the scatter plot using a lasso tool.
- Profile generation: Plots the number of points within each segment of the lasso selection.
- Point filtering: Removes points that have fewer than a specified number of neighbors within a given radius.
- Automatic saving: Saves plots and profiles in a well-structured output directory.

## Dependencies
The following Python packages are required:
- pandas
- numpy
- matplotlib
- scipy
- mpl_toolkits
- os

## Parameters
- xy_aspect_ratio = 1
- yz_aspect_ratio = 2
- xz_aspect_ratio = 2
- xyz_aspect_ratio = [1, 1, 0.01]
- dpi: quality parameter of an images.
- fontsize_axis_title: font of axes titles.
- num_segment: number of segments to split profile on.
- fig_len: length parameter of profile image.
- fig_wid: width parameter of profile image.
- filtering_radius: radius used for filtering points based on their density.
- filtering_min_neighbours: minimum number of neighbors required for a point to be kept after filtering.
- input_file_path: path to input file
- lasso_thickness: thickness of lasso line for data gathering.

## Description
Following script consist of several principle blocks of the code, which are explained below.

### Library imports and global variables
This block defines global parameters, including aspect ratios for plots (xy_aspect_ratio, etc.), plot styling settings (dpi, fontsize_axis_title, etc.), filtering parameter (filtering_radius, filtering_min_neighbours), file paths and thickness for the lasso tool.
 ```
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
 ```

### LassoAnalyzer class definition
Constructor (__init__) initializes the LassoAnalyzer object, sets attributes like canvas, coordinates (x, y), lasso thickness, and number of segments.
 ```
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
 ```

### Handling lasso selection (onselect)
onselect(): triggered when the user completes a lasso selection. It records the lasso coordinates, calculates its length, interpolates the line if segments are specified, and counts points near the line.
 ```
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
 ```

### Interpolating Lasso segments
interpolate_lasso_segments() splits the lasso line into a specified number of segments by interpolating points.
 ```
    def interpolate_lasso_segments(self):
        """Interpolate points along the lasso to achieve the desired number of segments."""
        if self.num_segments is None:
            return
        ...
 ```

### Calculating Lasso length
calculate_lasso_length() calculates the total length of the drawn lasso based on the sum of distances between consecutive points. 
```
    def interpolate_lasso_segments(self):
        """Interpolate points along the lasso to achieve the desired number of segments."""
        if self.num_segments is None:
            return
        ...
 ```

### Splitting the Lasso and counting points
split_lasso_and_count_points() divides the lasso into segments and counts how many data points fall within a specified thickness around each segment.
```
    def split_lasso_and_count_points(self):
        """Split the lasso into segments and calculate the number of points near each segment."""
        if self.num_segments is None or len(self.lasso_coords) < 2:
            return
        ...
 ```

### Checking point distance from Lasso
points_within_thickness() determines which points lie within a defined distance (thickness) of the lasso line.
```
    def points_within_thickness(self, segment_start, segment_end):
        """Find points within the lasso line's thickness (distance from the line)."""
        ...
 ```

### Plotting profile
plot_profile() creates a plot showing the relationship between the lasso segment length and the number of points near each segment. Saves the plot if an output path is provided.
```
    def plot_profile(self, output_path=None):
        """Plot the profile: length of the lasso segment (x-axis) vs. points per segment (y-axis)."""
        ...
 ```

### Utility functions
get_base_name() extracts the base name of the file by combining its path and name. set_aspect_ratio() sets the aspect ratio for a plot.
 ```
def get_base_name(input_file_path):
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    directory_path = os.path.dirname(input_file_path)
    directory_parts = directory_path.split(os.sep)
    base_name = '_'.join(directory_parts[-3:] + [file_name])
    return base_name

def set_aspect_ratio(ax, ratio=1):
    """Set aspect ratio of the plot."""
    ax.set_aspect(ratio)
 ```

### Filtering points
filter_points() filters points based on the density of neighboring points within a specified radius. Points with fewer than the minimum required neighbors are removed.
 ```
def filter_points(x, y, z, radius=filtering_radius, min_neighbors=filtering_min_neighbours):
    """Filter points based on the density of neighboring points."""
    ...
 ```

### Main plotting function
plot_3d_scatter_with_lasso() this is the main function for loading data, filtering points, and generating a 2D scatter plot where a lasso tool allows for manual selection of regions. It also generates a profile plot for the selected region.
```
def plot_3d_scatter_with_lasso(input_file_path, output_directory, sphere_radius=0.05, filter_radius=0.1, min_neighbors=5, lasso_thickness=lasso_thickness):
    ...
 ```

### Running the script
This block calls the plot_3d_scatter_with_lasso() function, specifying input file paths, output directory, and parameters like sphere_radius, filter_radius, and lasso_thickness. This executes the entire process of loading, filtering, plotting, and analyzing the data.
```
output_directory = '/Users/path/to/output/directory'
plot_3d_scatter_with_lasso(input_file_path, output_directory, sphere_radius=0.0006, filter_radius=filtering_radius, min_neighbors=filtering_min_neighbours, lasso_thickness=lasso_thickness)
 ```

## License
This project is licensed under the MIT License. See LICENSE file.
