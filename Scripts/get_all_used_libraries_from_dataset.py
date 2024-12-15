import json

def get_all_used_libraries_from_dataset(mbpp_path, std_library_path, http_package_path, output_path):
    # Load the sanitized-MBPP-midio dataset
    with open(mbpp_path, 'r') as f:
        mbpp_data = json.load(f)

    # Collect all used library functions from sanitized-MBPP-midio.json
    used_library_functions = set()
    for sample in mbpp_data:
        library_functions = sample.get('library_functions', [])
        # Remove 'root.std.' prefix if present
        clean_functions = [func.replace('root.std.', '') for func in library_functions]
        used_library_functions.update(clean_functions)

    # Load the StdLibrary dataset
    with open(std_library_path, 'r') as f:
        std_library_data = json.load(f)

    # Load the HttpPackage dataset
    with open(http_package_path, 'r') as f:
        http_package_data = json.load(f)

    # Combine StdLibrary and HttpPackage datasets
    combined_library_data = std_library_data + http_package_data

    # Filter the combined dataset to include only functions used in sanitized-MBPP-midio
    filtered_functions = [func for func in combined_library_data if func['function_name'] in used_library_functions]

    # Find functions in used_library_functions that are not in filtered_functions
    filtered_function_names = set(func['function_name'] for func in filtered_functions)

    # Save the filtered functions into one file
    with open(output_path, 'w') as f:
        json.dump(filtered_functions, f, indent=4)

    # Print the number of used library functions included in the dataset
    print(f"Number of used library functions included in the dataset: {len(filtered_functions)}")

    # Print the functions missing from the dataset
    missing_functions = used_library_functions - filtered_function_names
    print(f"Difference between original dataset and new used libraries dataset: {missing_functions}")

# Example usage
if __name__ == "__main__":
    dataset_path = '../Data/Few-shot/train_10_shot.json'
    std_library_path = '../Midio_libraries/StdLibrary_dataset.json'
    http_package_path = '../Midio_libraries/HttpPackage_dataset.json'
    output_path = '../Data/Few-shot/libraries_10_shot.json'
    
    get_all_used_libraries_from_dataset(dataset_path, std_library_path, http_package_path, output_path)