'''
Deprecated earlier attempt at correcting for drift using logs.
'''

# A script for reading in data from the calibration logs.)
# Ultimately the goal is to reduce error in the recorded values by accounting for drift.

import os
import pandas as pd
import chardet

# Specify the relative path to the directory containing the CSV files
directory = '../data/calibrationlogs'  # Replace with your relative path, e.g., './data' or '../csv_files'

# Define the prefix you want to filter by
prefix = 'Calibration File Export -'  # Replace with your specific prefix

# Initialize an empty list to hold each DataFrame
dataframes = []

# Loop through each file in the specified directory
for filename in os.listdir(directory):
    # Process only files that start with the specified prefix and end with .csv
    if filename.startswith(prefix) and filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)

        # Check the encoding of the file so it will be parsed correctly:
        with open(file_path, 'rb') as file:
            # Read a portion of the file to detect encoding
            raw_data = file.read(10000)  # Reading the first 10,000 bytes
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        # Read the CSV file into a DataFrame, using the detected encoding and a multi-character separator
        df = pd.read_csv(file_path, encoding=encoding, sep='=,', skiprows=2, index_col=False, engine='python')
        df['source_file'] = filename[26:-4]  # Add a column with the unique portion of the filename for sorting
        df.reset_index()

        # Standardize column names from each file
        new_column_names = df.columns.tolist()  # Get existing column names as a list
        new_column_names[0] = 'Parameter'
        new_column_names[1] = 'Value'
        new_column_names[2] = 'SourceFile'
        df.columns = new_column_names

        # Append the DataFrame to the list
        dataframes.append(df)



# Concatenate all DataFrames in the list into a single DataFrame
# print('Files: ', len(dataframes))
combined_df = pd.concat(dataframes, ignore_index=False)

# Display the combined DataFrame
print(combined_df)

# Display unique values in the first column and their counts
unique_counts = combined_df['Parameter'].value_counts()
print('Parameters:', unique_counts)

# unique_counts = combined_df['SourceFile'].value_counts()
# print('Source Files:', unique_counts)

breaker = '----------'
label = 'Parameter Type'
# breaks = combined_df.loc[combined_df['Parameter'] == breaker, 'Value'].iloc[:]
indices = combined_df.loc[combined_df['Parameter'] == breaker].index
sensor = combined_df.loc[combined_df['Parameter'] == label, 'Value'].iloc[:]

print('Indices:', indices)
print('Sensors:', sensor)


# Define the string to identify table types
identifier_column = 'col2'
identifier_string = 'B'  # Example string to identify table type

# Split dataframe into individual tables
tables = []
current_table = []

for index, row in combined_df.iterrows():
    if pd.isna(row['col1']):
        if current_table:
            tables.append(pd.DataFrame(current_table))
            current_table = []
    else:
        current_table.append(row)

# Add the last table if it exists
if current_table:
    tables.append(pd.DataFrame(current_table))

# Label tables based on the identifier string
labeled_tables = {}
for table in tables:
    label = None
    for _, row in table.iterrows():
        if row[identifier_column] == identifier_string:
            label = identifier_string
            break
    if label:
        if label not in labeled_tables:
            labeled_tables[label] = []
        labeled_tables[label].append(table)

# Display labeled tables
for label, group in labeled_tables.items():
    print(f"Tables with label '{label}':")
    for table in group:
        print(table)
        print()

