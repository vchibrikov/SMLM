import numpy as np  # For numerical operations, including array manipulations and distance calculations
import pandas as pd  # For handling data, including reading CSV files and writing Excel files
import matplotlib.pyplot as plt  # For creating plots and visualizations
from matplotlib.widgets import PolygonSelector  # For interactive region selection on plots
from matplotlib.path import Path  # For handling polygon path definitions
from scipy.spatial import cKDTree  # For fast spatial queries and filtering by proximity
from scipy.spatial.distance import cdist  # For calculating pairwise distances between points
from scipy.optimize import curve_fit  # For fitting curves to data
import random  # For generating random numbers to create unique filenames

fontsize_axis_title = 18  # Font size for axis titles in plots

# Define an exponential decay function for curve fitting
def exponential_decay(r, a, b):
    return a * np.exp(-b * r)

# Filtering function based on minimum neighbors
def filter_by_min_neighbors(points, radius, min_neighbors):
    """
    Filters points based on the number of neighbors within a specified radius.
    """
    tree = cKDTree(points)  # Build a KD-tree for efficient spatial queries
    counts = np.array([len(tree.query_ball_point(p, radius)) for p in points])  # Count the number of neighbors within the radius
    return points[counts >= min_neighbors]  # Return points that have at least 'min_neighbors' neighbors

# Load datasets
# Replace dataset paths with actual paths to your data files
dataset1_file = '/path/to/data/dataset1.csv'  # Path to first dataset (e.g., CBM3a)
dataset2_file = '/path/to/data/dataset2.csv'  # Path to second dataset (e.g., LM19)

# Read the datasets into pandas DataFrames
data1 = pd.read_csv(dataset1_file, header=None)
data2 = pd.read_csv(dataset2_file, header=None)

# Extract X, Y coordinates for both datasets and convert from nm to μm (micrometers)
x1 = (data1.iloc[:, 0].values / 1000)  # Dataset 1, X coordinates
y1 = (data1.iloc[:, 1].values / 1000)  # Dataset 1, Y coordinates
x2 = data2.iloc[:, 0].values / 1000  # Dataset 2, X coordinates
y2 = data2.iloc[:, 1].values / 1000  # Dataset 2, Y coordinates
points1 = np.column_stack((x1, y1))  # Combine X and Y coordinates into a 2D array for dataset 1
points2 = np.column_stack((x2, y2))  # Combine X and Y coordinates into a 2D array for dataset 2

# Apply Min Neighbors Filter
radius1, min_neighbors1 = 0.2, 5  # Filtering parameters for Dataset 1
radius2, min_neighbors2 = 0.2, 5  # Filtering parameters for Dataset 2

# Filter points in both datasets based on the number of neighbors within the specified radius
filtered_points1 = filter_by_min_neighbors(points1, radius1, min_neighbors1)
filtered_points2 = filter_by_min_neighbors(points2, radius2, min_neighbors2)

# Global variables for region selection
selected_region = None  # Stores the selected polygon region
selected_points1 = None  # Stores the points from dataset 1 that fall within the selected region
selected_points2 = None  # Stores the points from dataset 2 that fall within the selected region

def select_region(ax, points1, points2):
    """
    Allows the user to interactively draw a polygon on the plot and select points within it for both datasets.
    """
    def onselect(verts):
        global selected_region, selected_points1, selected_points2

        # Create a path from the drawn polygon
        path = Path(verts)
        selected_region = verts  # Store the selected region

        # Filter points within the polygon for both datasets
        mask1 = path.contains_points(points1)
        mask2 = path.contains_points(points2)
        selected_points1 = points1[mask1]  # Points from dataset 1 inside the polygon
        selected_points2 = points2[mask2]  # Points from dataset 2 inside the polygon

        # Highlight selected points on the plot
        ax.scatter(selected_points1[:, 0], selected_points1[:, 1], color='green', label='Selected Points (Dataset 1)')
        ax.scatter(selected_points2[:, 0], selected_points2[:, 1], color='red', label='Selected Points (Dataset 2)')
        ax.legend()
        plt.draw()  # Redraw the plot with the highlighted points

        # Disconnect the selector after the selection is made
        selector.disconnect_events()
        print("Region selected, and points filtered.")

    # Use PolygonSelector for drawing the region interactively
    selector = PolygonSelector(ax, onselect, useblit=True)
    plt.show()  # Display the plot and wait for user interaction

def calculate_pcf(points1, points2, output_file, plot_file, dpi=600):
    """
    Calculate and plot the pair correlation function (PCF) between points from two datasets.
    Save the results to a .xlsx file and plot as .png.
    """
    # Compute pairwise distances between points from the two datasets
    distances = cdist(points1, points2, metric='euclidean')
    distances_flat = distances.flatten()  # Flatten distances array for histogram calculation

    # Create histogram of distances with 200 bins
    hist, bin_edges = np.histogram(distances_flat, bins=200)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers

    # Create fixed bin centers with a step of 0.001
    step = 0.001
    bin_centers_fixed = np.arange(start=np.ceil(bin_centers[0] / step) * step, 
                                  stop=np.floor(bin_centers[-1] / step) * step + step, 
                                  step=step)
    frequencies_fixed = np.interp(bin_centers_fixed, bin_centers, hist)  # Interpolate frequencies for fixed bins

    # Fit the exponential decay function to the data
    popt, pcov = curve_fit(exponential_decay, bin_centers_fixed, frequencies_fixed, p0=(np.max(frequencies_fixed), 0.1))

    # Save the data to an Excel file
    df = pd.DataFrame({
        'Bin Centers': bin_centers_fixed,
        'Frequency': frequencies_fixed,
        'Fitted Curve': exponential_decay(bin_centers_fixed, *popt)  # Calculate fitted curve values
    })
    df.to_excel(output_file, index=False)  # Save the DataFrame to an Excel file

    # Create a plot of the histogram and fitted curve
    plt.figure(figsize=(6, 6))
    plt.bar(bin_centers_fixed, frequencies_fixed, width=step, color='blue', alpha=0.7)
    plt.xlabel('Distance [μm]', fontsize=fontsize_axis_title)
    plt.ylabel('Frequency', fontsize=fontsize_axis_title)
    plt.grid(True)

    # Save the plot as a .png file with transparent background and high DPI
    plt.savefig(plot_file, dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close()  # Close the plot to free up resources

# Main code to execute the steps
fig, ax = plt.subplots(figsize=(10, 8))  # Create figure and axis for plotting
# Plot filtered points from both datasets
ax.scatter(filtered_points1[:, 0], filtered_points1[:, 1], s=5, color='green', label='Filtered Points (Dataset 1)')
ax.scatter(filtered_points2[:, 0], filtered_points2[:, 1], s=5, color='red', label='Filtered Points (Dataset 2)')
ax.set_title("Select Region with Polygon Tool")  # Set plot title
ax.set_xlabel("X [μm]")  # Set X-axis label
ax.set_ylabel("Y [μm]")  # Set Y-axis label
ax.legend()  # Display legend

# Step 1: Select Region (interactive region selection)
select_region(ax, filtered_points1, filtered_points2)

# Step 2: Calculate Pair Correlation Function (PCF) for Selected Region
if selected_points1 is not None and selected_points2 is not None:
    # Generate a random number for unique filenames
    random_number = random.randint(100000, 999999)

    # Define output file paths with random number appended to filenames
    output_file = f'/path/to/output/pcf_result_{random_number}.xlsx'  # Path to save the PCF result data
    plot_file = f'/path/to/output/pcf_plot_{random_number}.png'  # Path to save the PCF plot

    # Calculate PCF and save data and plot
    calculate_pcf(selected_points1, selected_points2, output_file, plot_file)
    print(f"Results saved to {output_file} and plot saved as {plot_file}")
else:
    print("No region selected. PCF calculation skipped.")
