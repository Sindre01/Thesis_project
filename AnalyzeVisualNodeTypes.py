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

    # Step 3: Count the occurrences of each visual_node_type
    visual_node_counts = Counter()
    all_visual_node_types = set()
    for item in filtered_data:
        visual_node_types = item.get('visual_node_types', [])
        # Remove any prefixes if necessary (adjust as needed)
        visual_node_types = [node_type.replace('root.std.', '') for node_type in visual_node_types]
        visual_node_counts.update(visual_node_types)
        all_visual_node_types.update(visual_node_types)

    # Count the total number of unique visual node types used
    total_unique_visual_nodes = len(all_visual_node_types)
    print(f"Total number of unique visual node types used: {total_unique_visual_nodes}")

    # Check if any visual node types were found
    if not visual_node_counts:
        print("No visual node types found in the specified task IDs.")
    else:
        # Step 4: Create the bar chart
        visual_node_types = list(visual_node_counts.keys())
        counts = list(visual_node_counts.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(visual_node_types, counts, color='skyblue')
        plt.xlabel('Visual Node Types')
        plt.ylabel('Count')

        # Add total unique visual node types to the plot title
        plt.title(f'Count of Visual Node Types Used (Unique Types: {total_unique_visual_nodes})\nSamples with Task ID 1 to 22')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Add text annotation inside the plot
        plt.text(0.95, 0.95, f'Total Unique Types: {total_unique_visual_nodes}',
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