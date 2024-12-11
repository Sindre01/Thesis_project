import json
import matplotlib.pyplot as plt
from collections import Counter
import os

# Path to the JSON file
json_file_path = '../MBPP_transformed_code_examples/sanitized-MBPP-midio.json'

# Ensure the file exists
if not os.path.exists(json_file_path):
    print(f"The file '{json_file_path}' does not exist.")
else:
    # Step 1: Read the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Step 2: Filter the samples with task_id from 1 to 22
    filtered_data = [item for item in data if 1 <= item.get('task_id', 0) <= 22]

    # Step 3: Count the occurrences of each textual_instance_type
    textual_instance_counts = Counter()
    all_textual_instance_types = set()
    for item in filtered_data:
        textual_instance_types = item.get('textual_instance_types', [])
        # Remove any prefixes if necessary (adjust as needed)
        textual_instance_types = [instance_type.replace('root.std.', '') for instance_type in textual_instance_types]
        textual_instance_counts.update(textual_instance_types)
        all_textual_instance_types.update(textual_instance_types)

    # Count the total number of unique textual instance types used
    total_unique_textual_instances = len(all_textual_instance_types)
    print(f"Total number of unique textual instance types used: {total_unique_textual_instances}")

    # Check if any textual instance types were found
    if not textual_instance_counts:
        print("No textual instance types found in the specified task IDs.")
    else:
        # Step 4: Create the bar chart
        textual_instance_types = list(textual_instance_counts.keys())
        counts = list(textual_instance_counts.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(textual_instance_types, counts, color='skyblue')
        plt.xlabel('Textual Instance Types')
        plt.ylabel('Count')

        # Add total unique textual instance types to the plot title
        plt.title(f'Count of Textual Instance Types Used (Unique Types: {total_unique_textual_instances})\nSamples with Task ID 1 to 22')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Add text annotation inside the plot
        plt.text(0.95, 0.95, f'Total Unique Types: {total_unique_textual_instances}',
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