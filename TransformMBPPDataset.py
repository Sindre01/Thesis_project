import json
import os

def transform_json_objects(input_file, output_file):
    # Step 1: Read the existing JSON data
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    # Step 2: Transform each JSON object
    transformed_data = []
    for item in data:
        transformed_item = {
            "prompts": item.get("prompts", []),
            "task_id": item.get("task_id", -1),
            "specification": item.get("specification", {
                "function_signature": "",
                "preconditions": "",
                "postconditions": ""
            }),
            "MBPP_task_id": item.get("MBPP_task_id", -1),
            "library_functions": item.get("library_functions", []),
            "visual_node_types": item.get("visual_node_types", []),
            "textual_instance_types": item.get("textual_instance_types", []),
            "testing": {
                "library_functions": item.get("testing_library_functions", []),
                "visual_node_types": item.get("testing_visual_node_types", []),
                "textual_instance_types": item.get("testing_textual_instance_types", [])
            }
        }

        # Remove fields that are no longer needed
        fields_to_remove = [
            "code",
            "testing_library_functions",
            "testing_visual_node_types",
            "testing_textual_instance_types",
            "library_functions_secondary",
            "visual_node_types_secondary",
            "textual_instance_types_secondary"
        ]
        for field in fields_to_remove:
            transformed_item.pop(field, None)

        # Append the transformed item
        transformed_data.append(transformed_item)

    # Step 3: Write the transformed data to a new JSON file
    # Check if the output file exists and is non-empty
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        # Create a new file to avoid overwriting
        base_output_file = output_file
        file_number = 1
        while True:
            name, ext = os.path.splitext(base_output_file)
            new_output_file = f"{name}_{file_number}{ext}"
            if not os.path.exists(new_output_file) or os.path.getsize(new_output_file) == 0:
                output_file = new_output_file
                break
            file_number += 1

    # Write the transformed data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)

    print(f"Transformed data written to {output_file}")

# Example usage
input_file = './midio/sanitized-MBPP-midio.json'
output_file = 'sanitized-MBPP-midio-transformed.json'

# transform_json_objects(input_file, output_file)



def merge_datasets(midio_file, mbpp_file, output_file):
    # Step 1: Read the Midio dataset
    with open(midio_file, 'r') as midio_f:
        midio_data = json.load(midio_f)
    
    # Step 2: Read the MBPP dataset
    with open(mbpp_file, 'r') as mbpp_f:
        mbpp_data = json.load(mbpp_f)
        # Create a dictionary indexed by task_id for quick lookup
        mbpp_dict = {item['task_id']: item for item in mbpp_data}

    # Step 3: Merge the datasets
    for midio_item in midio_data:
        mbpp_task_id = midio_item.get('MBPP_task_id', -1)
        # Skip if MBPP_task_id is -1 or not an integer
        if not isinstance(mbpp_task_id, int) or mbpp_task_id == -1:
            continue
        # Find the corresponding MBPP item
        mbpp_item = mbpp_dict.get(mbpp_task_id)
        if mbpp_item:
            # Get the test_list from MBPP item
            test_list = mbpp_item.get('test_list', [])
            # Add test_list to the testing attribute of Midio item
            if 'testing' not in midio_item:
                midio_item['testing'] = {}

            midio_item['testing']['test_list'] = test_list
            original_prompt = mbpp_item.get('prompt', "")
            if original_prompt != "":
                midio_item['prompts'][0] = original_prompt

    # Step 4: Write the merged dataset to a new file
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        # Create a new file to avoid overwriting
        base_output_file = output_file
        file_number = 1
        while True:
            name, ext = os.path.splitext(base_output_file)
            new_output_file = f"{name}_{file_number}{ext}"
            if not os.path.exists(new_output_file) or os.path.getsize(new_output_file) == 0:
                output_file = new_output_file
                break
            file_number += 1

    with open(output_file, 'w') as outfile:
        json.dump(midio_data, outfile, indent=4)
    
    print(f"Merged dataset written to {output_file}")

# Example usage
midio_file = './Midio/MBPP_transformed_code_examples/sanitized-MBPP-midio.json'
mbpp_file = './MBPP/sanitized-mbpp.json'
output_file = './Midio/MBPP_transformed_code_examples/merged_dataset.json'

merge_datasets(midio_file, mbpp_file, output_file)