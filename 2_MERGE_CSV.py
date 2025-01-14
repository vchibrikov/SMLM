import pandas as pd

# This script merges two CSV files, adds a new column based on the file names to indicate file-specific data,
# and saves the merged data into a new CSV file without headers.

def merge_csv_and_add_column(csv_file1, csv_file2, merged_file):
    # Read both CSV files without headers
    df1 = pd.read_csv(csv_file1, header=None)  # Load the first CSV into a DataFrame
    df2 = pd.read_csv(csv_file2, header=None)  # Load the second CSV into a DataFrame
    
    # Add Column 9 with values based on the file names
    if "CBM3a" in csv_file1:  # Check for a specific pattern in the first file name
        df1['Column 9'] = 0  # Assign a value of 0 if the pattern is found
    else:
        df1['Column 9'] = 1  # Assign a value of 1 otherwise
    
    if "CBM3a" in csv_file2:  # Check for a specific pattern in the second file name
        df2['Column 9'] = 0  # Assign a value of 0 if the pattern is found
    else:
        df2['Column 9'] = 1  # Assign a value of 1 otherwise
    
    # Concatenate the two DataFrames (without column names)
    merged_df = pd.concat([df1, df2], ignore_index=True, axis=0)  # Merge both DataFrames row-wise
    
    # Save the merged data to a new CSV file (excluding column names)
    merged_df.to_csv(merged_file, index=False, header=False)  # Write the DataFrame to a file without headers
    
    print(f"Merged data saved to {merged_file}")

# Example usage with placeholder paths
csv_file1 = "path/to/your/first_file.csv"  # Replace with the actual path to the first CSV file
csv_file2 = "path/to/your/second_file.csv"  # Replace with the actual path to the second CSV file
merged_file = "path/to/your/merged_file.csv"  # Replace with the desired path for the output CSV file

# Call the function with the specified paths
merge_csv_and_add_column(csv_file1, csv_file2, merged_file)
