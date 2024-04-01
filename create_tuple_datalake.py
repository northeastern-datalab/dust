import glob
import pandas as pd
import os
import utilities as utl

# Define the input and output folders
input_folder = 'data/ugen_benchmark/datalake'
output_folder = 'data/ugen_benchmark_tuple_level'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all CSV files in the input folder
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

# Process each CSV file
for csv_file in csv_files:
    try:
        # Read the CSV file into a DataFrame
        # df = pd.read_csv(csv_file)
        df = utl.read_csv_file(csv_file)
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            try:
                # Create a new DataFrame with only the header and the current row
                new_df = pd.DataFrame([row], columns=df.columns)

                # Save the new DataFrame as a CSV file in the output folder
                output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(csv_file))[0]}_{index + 1}.csv")
                new_df.to_csv(output_file, index=False)

            except Exception as e:
                print(f"Error processing row {index + 1} in file {csv_file}: {e}")

    except Exception as e:
        print(f"Error reading file {csv_file}: {e}")

print("Tables generated successfully.")
