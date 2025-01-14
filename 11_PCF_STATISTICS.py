import os  # For interacting with the file system and reading directories
import pandas as pd  # For reading and processing Excel files
import numpy as np  # For numerical calculations like mean and standard deviation

def process_xlsx_files(folder_path, output_file):
    # List to store values from the first column of the row with the highest value in the second column
    first_column_values = []
    
    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Process only .xlsx files
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)  # Construct full file path
            
            # Read the .xlsx file into a pandas DataFrame
            df = pd.read_excel(file_path, header=None)
            
            # Check if the second column exists
            if df.shape[1] > 1:
                # Convert the second column to numeric, coercing any errors to NaN
                df[1] = pd.to_numeric(df[1], errors='coerce')
                
                # Find the row with the maximum value in the second column
                max_value_row = df.iloc[df[1].idxmax()]
                
                # Extract the value from the first column of that row
                first_column_values.append(max_value_row[0])  # Append the value to the list
    
    # Calculate the mean and standard deviation of the extracted values
    if first_column_values:
        mean_value = np.mean(first_column_values)  # Calculate the mean
        std_dev_value = np.std(first_column_values)  # Calculate the standard deviation
        
        # Write the results to the output .txt file
        with open(output_file, 'w') as f:
            f.write(f"Mean of the first column values: {mean_value}\n")
            f.write(f"Standard Deviation of the first column values: {std_dev_value}\n")
        print(f"Results saved to {output_file}")
    else:
        print("No data found in the specified folder.")  # If no data was found

# Define folder path and output file
folder_path = '/path/to/your/data/folder'  # Replace with actual folder path containing .xlsx files
output_file = '/path/to/output/statistics_output.txt'  # Replace with path for saving the statistics output

# Example alternative paths, commented out
# folder_path = '/path/to/another/data/folder'
# output_file = '/path/to/output/another_output.txt'

# Call the function to process the files and save the results
process_xlsx_files(folder_path, output_file)

