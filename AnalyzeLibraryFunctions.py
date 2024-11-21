import json
import matplotlib.pyplot as plt
from collections import Counter
import os

# Path to the JSON file
json_file_path = 'Midio/MBPP_transformed_code_examples/sanitized-MBPP-midio.json'

# Ensure the file exists
if not os.path.exists(json_file_path):
    print(f"The file '{json_file_path}' does not exist.")
else:
    # Step 1: Read the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Step 2: Filter the samples with task_id from 1 to 22
    filtered_data = [item for item in data if 1 <= item.get('task_id', 0) <= 22]

    # Step 3: Count the occurrences of each library_function
    library_function_counts = Counter()
    all_library_functions = set()
    for item in filtered_data:
        library_functions = item.get('library_functions', [])
        # Remove 'root.std.' from library function names
        library_functions = [func.replace('root.std.', '') for func in library_functions]
        library_function_counts.update(library_functions)
        all_library_functions.update(library_functions)

    # Count the total number of unique functions used
    total_unique_functions = len(all_library_functions)
    print(f"Total number of unique library functions used: {total_unique_functions}")

    # Check if any library functions were found
    if not library_function_counts:
        print("No library functions found in the specified task IDs.")
    else:
        # Step 4: Create the bar chart
        library_functions = list(library_function_counts.keys())
        counts = list(library_function_counts.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(library_functions, counts, color='skyblue')
        plt.xlabel('Library Functions')
        plt.ylabel('Count')

        # Add total unique functions to the plot title
        plt.title(f'Count of samples using Library Functions (Unique Functions: {total_unique_functions})\nSamples with Task ID 1 to 22')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Optionally, add text annotation inside the plot
        plt.text(0.95, 0.95, f'Total Unique Functions: {total_unique_functions}',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.5))
         # Add the counts inside the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,  # Position text in the middle of the bar
                f'{int(height)}',
                ha='center',
                va='center',
                color='white',
                fontsize=9
            )
        plt.show()