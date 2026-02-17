# ## Identify GPU parameters for e.g. running code with tensor functions efficiently
# import torch
# print(torch.cuda.get_device_name(0))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

## Get column names from a CSV file formatted as a list of strings
import pandas as pd

# 1. Read the CSV file, loading only the header (first row)
# 'nrows=0' tells Pandas to read no data, just the column names.
df = pd.read_csv('../data/input/sensors/Eurofins.csv', nrows=0, sep=';', decimal='.')

# 2. Extract the column names (the header) as a list of strings
column_names_list = df.columns.tolist()

print(column_names_list)
