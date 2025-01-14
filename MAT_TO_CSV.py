import scipy.io
import csv

# This script reads data from a .mat file, extracts a specific data array associated with a given key,
# formats the data into rows with a fixed number of columns, and writes the formatted data to a .csv file.

def read_mat_and_write_csv(mat_file_path, csv_file_path):
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)  # Reads the .mat file into a dictionary
    
    # Specify the key you are interested in
    target_key = "your_target_key"  # Replace with the actual key you are interested in

    # Check if the key exists in the .mat file
    if target_key not in mat_data:
        print(f"Key '{target_key}' not found in the .mat file.")
        return
    
    # Extract the data associated with the target key
    data = mat_data[target_key].flatten()  # Flattens the data to a 1D array for easier processing
    
    # Specify the number of columns for the CSV file
    num_columns = 8  # Adjust as per your requirements
    
    # Write the extracted data to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Divide the data into rows with `num_columns` columns
        for i in range(0, len(data), num_columns):
            row = data[i:i + num_columns]
            writer.writerow(row)
    
    print(f"Data under '{target_key}' written to {csv_file_path}")

# Example usage with placeholder paths
mat_file = "path/to/your/input_file.mat"  # Replace with the actual path to the .mat file
csv_file = "path/to/your/output_file.csv"  # Replace with the desired path for the output .csv file

# Call the function with the specified paths
read_mat_and_write_csv(mat_file, csv_file)
