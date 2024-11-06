import json

def select_objects_by_task_id(dafny_file, sanitized_file, output_file):
    # Read the Dafny file to get the task_ids
    with open(dafny_file, 'r') as dafny_f:
        dafny_data = json.load(dafny_f)
        
        # Check if dafny_data is a list or a single dictionary
        if isinstance(dafny_data, dict):
            dafny_data = [dafny_data]
        
        # Debugging: Print the structure of dafny_data
        print("Dafny Data:", dafny_data)
        
        # Extract task_ids
        task_ids = []
        for item in dafny_data:
            if 'task_id' in item:
                task_ids.append(item['task_id'])
            else:
                print(f"Warning: 'task_id' not found in item: {item}")

    # Read the sanitized file and select objects with matching task_ids
    with open(sanitized_file, 'r') as sanitized_f:
        sanitized_data = json.load(sanitized_f)
        selected_objects = [item for item in sanitized_data if item['task_id'] in task_ids]

    # Write the selected objects to a new JSON file
    with open(output_file, 'w') as output_f:
        json.dump(selected_objects, output_f, indent=4)

# Example usage
dafny_file = './MBPP/Dafny/mbpp-dfy-50-examples-db.json'
sanitized_file = './MBPP/sanitized-mbpp.json'
output_file = './MBPP/selected_objects.json'

select_objects_by_task_id(dafny_file, sanitized_file, output_file)