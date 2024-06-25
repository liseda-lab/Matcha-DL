import os
import re
import pandas as pd

# Define the base folder structure
base_folders = ['sup_05', 'unsup_05', 'sup_09', 'unsup_09']
sub_folder_path = 'alignment_results/snomed-ncit.pharm'
results_filename = 'results.txt'

# Regular expression pattern to extract the values from results.txt content
pattern = re.compile(r"(Unsupervised|Semi-supervised) Global Matching Results: {'P': (?P<P>[\d.]+), 'R': (?P<R>[\d.]+), 'F1': (?P<F1>[\d.]+)}")

# List to hold the extracted data
data = []

# Iterate over each base folder
for folder in base_folders:
    # Define the full path to the results.txt file
    file_path = os.path.join(os.getcwd(), folder, results_filename)
    print("Looking for", file_path)
    # Check if the results.txt file exists
    if os.path.exists(file_path):
        print("Found results.txt")
        with open(file_path, 'r') as file:
            content = file.read()
            # Extract the values using the regular expression
            match = pattern.search(content)
            if match:
                P = float(match.group('P'))
                R = float(match.group('R'))
                F1 = float(match.group('F1'))
                # Extract the supervision type and threshold from the folder name
                supervision = folder.split('_')[0]
                threshold = folder.split('_')[1]
                # Append the data to the list
                data.append({
                    'Supervision': supervision,
                    'Threshold': threshold,
                    'P': P,
                    'R': R,
                    'F1': F1
                })
                print({
                    'Supervision': supervision,
                    'Threshold': threshold,
                    'P': P,
                    'R': R,
                    'F1': F1
                })

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Write the DataFrame to an Excel file
output_file = 'alignment_results.xlsx'
df.to_excel(output_file, index=False)

print(f"Results have been written to {output_file}")
