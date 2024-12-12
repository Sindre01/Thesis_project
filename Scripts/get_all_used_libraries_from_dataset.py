import json

# Load the sanitized-MBPP-midio dataset
with open('../Data/MBPP_transformed_code_examples/sanitized-MBPP-midio.json', 'r') as f:
    mbpp_data = json.load(f)

# Collect all used library functions from sanitized-MBPP-midio.json
used_library_functions = set()
for sample in mbpp_data:
    library_functions = sample.get('library_functions', [])
    # Remove 'root.std.' prefix if present
    clean_functions = [func.replace('root.std.', '') for func in library_functions]
    used_library_functions.update(clean_functions)

# Load the StdLibrary dataset
with open('../Midio_libraries/StdLibrary_dataset.json', 'r') as f:
    std_library_data = json.load(f)

# Load the HttpPackage dataset
with open('../Midio_libraries/HttpPackage_dataset.json', 'r') as f:
    http_package_data = json.load(f)

# Combine StdLibrary and HttpPackage datasets
combined_library_data = std_library_data + http_package_data

# Filter the combined dataset to include only functions used in sanitized-MBPP-midio
filtered_functions = [func for func in combined_library_data if func['function_name'] in used_library_functions]

# Find functions in used_library_functions that are not in filtered_functions
filtered_function_names = set(func['function_name'] for func in filtered_functions)

# Save the filtered functions into one file
with open('../Data/used_libraries_in_datasets.json', 'w') as f:
    json.dump(filtered_functions, f, indent=4)

# print(f"Number of used library functions: {len(used_library_functions)}")

print(f"Number of used library functions included in the dataset: {len(filtered_functions)}")

missing_functions = used_library_functions - filtered_function_names
print(f"Functions missing from: {missing_functions}")