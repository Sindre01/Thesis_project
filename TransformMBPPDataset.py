import json
import os

def select_objects_by_task_id(dafny_file, sanitized_file, output_file):
    # Step 1: Read the Dafny file and get the task_ids
    with open(dafny_file, 'r') as dafny_f:
        dafny_data = json.load(dafny_f)
        # task_ids are keys in the dictionary, convert them to integers
        task_ids = set(int(task_id) for task_id in dafny_data.keys())

    # Step 2: Read the sanitized file
    with open(sanitized_file, 'r') as sanitized_f:
        sanitized_data = json.load(sanitized_f)

    # Step 3: Select objects with matching task_ids
    selected_objects = [item for item in sanitized_data if item['task_id'] in task_ids]

    # Step 4: Remove 'source_file' and 'test_imports' from each object
    for item in selected_objects:
        item.pop('source_file', None)
        item.pop('test_imports', None)

    # Step 5: Check if the output file exists and create a new one if it does
    base_output_file = output_file
    file_number = 1
    while os.path.exists(output_file):
        name, ext = os.path.splitext(base_output_file)
        output_file = f"{name}_{file_number}{ext}"
        file_number += 1

    # Step 6: Write selected objects to the output file
    with open(output_file, 'w') as output_f:
        json.dump(selected_objects, output_f, indent=4)

    print(f"Selected objects written to {output_file}")

# Example usage
dafny_file = './MBPP/Dafny/mbpp-dfy-50-examples-db.json'
sanitized_file = './MBPP/sanitized-mbpp.json'
output_file = './MBPP-san-midio.json'

select_objects_by_task_id(dafny_file, sanitized_file, output_file)