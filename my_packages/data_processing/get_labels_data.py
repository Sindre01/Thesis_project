import json
from my_packages.data_processing.midio_libraries_processing import extract_midio_functions

def used_libraries_from_dataset(data):

    # Collect all used library functions from sanitized-MBPP-midio.json
    used_library_functions = set()
    for sample in data:
        library_functions = sample.get('library_functions', [])
        # Remove 'root.std.' prefix if present
        clean_functions = [func.replace('root.std.', '') for func in library_functions]
        used_library_functions.update(clean_functions)

    std_library_data = extract_midio_functions("../../midio_libraries/src/std_library")
    http_package_data = extract_midio_functions("../../midio_libraries/src/http_package")

    combined_library_data = std_library_data + http_package_data

    # Filter the combined dataset to include only functions used in sanitized-MBPP-midio
    filtered_functions = [func for func in combined_library_data if func['function_name'] in used_library_functions]

    # Print the number of used library functions included in the dataset
    print(f"Library functions included in the dataset: {len(filtered_functions)}")

    # Find functions in used_library_functions that are not in filtered_functions
    filtered_function_names = set(func['function_name'] for func in filtered_functions)
    # functions migt missing from the dataset, in case some functions in the provided dataset where not found in libraries
    missing_functions = used_library_functions - filtered_function_names
    if missing_functions:
        print(f"These functions could not be found in libraries: {missing_functions}")

    return filtered_functions

def used_libraries_to_string(data):
    name_doc_string = ""
    for func in data:
        name_doc_string += f"Function node name: {func['function_name']}\n Documentation: {func['doc']}\n\n"
    return name_doc_string

# Example usage
if __name__ == "__main__":

    with open( '../../data/few_shot/train_5_shot.json', 'r') as file:
        data = json.load(file)
    
    used_libraries = used_libraries_from_dataset(data)
    
    # Save the filtered functions into one file
    with open('../../data/few_shot/libraries_5_shot.json', 'w') as f:
        json.dump(used_libraries, f, indent=4)