import json
from my_packages.data_processing.midio_libraries_processing import extract_midio_functions
import re
import json
import ast

def used_functions_from_dataset(data: list[dict]):
    """Extracts the used external functions from a dataset of sanitized MBPP-midio samples."""
    # Collect all used external functions from sanitized-MBPP-midio.json
    used_external_functions = set()
    for sample in data:
        external_functions = sample.get('external_functions', [])
        # Remove 'root.std.' prefix if present
        clean_functions = [func.replace('root.std.', '') for func in external_functions]
        used_external_functions.update(clean_functions)

    std_library_data = extract_midio_functions("../../midio_libraries/src/std_library")
    http_package_data = extract_midio_functions("../../midio_libraries/src/http_package")

    combined_library_data = std_library_data + http_package_data

    # Filter the combined dataset to include only functions used in sanitized-MBPP-midio
    filtered_functions = [func for func in combined_library_data if func['function_name'] in used_external_functions]

    # Print the number of used library functions included in the dataset
    print(f"Library functions included in the dataset: {len(filtered_functions)}")

    # Find functions in used_external_functions that are not in filtered_functions
    filtered_function_names = set(func['function_name'] for func in filtered_functions)
    # functions migt missing from the dataset, in case some functions in the provided dataset where not found in libraries
    missing_functions = used_external_functions - filtered_function_names
    if missing_functions:
        print(f"These functions could not be found in libraries: {missing_functions}")

    return filtered_functions

def used_functions_to_string(data: list[dict]):
    """Converts a list of function dictionaries to an explainable string format. (e.g. to use in SYSTEM_PROMPT)"""
    name_doc_string = ""
    for func in data:
        name_doc_string += f"Function node name: {func['function_name']}\n Documentation: {func['doc']}\n\n"
    return name_doc_string

    
def parse_assertion(assertion):
    """
    Parses an assertion string like 'assert smallest_num([10, 20, 1, 45, 99]) == 1'
    and extracts function inputs and expected output.
    """
    # Regex to extract function name, inputs, and expected output
    match = re.match(r'assert\s+([\w_]+)\((.*?)\)\s*==\s*(True|False|\d+|\[.*?\]|".*?")', assertion)
    
    if not match:
        return None

    function_name, input_str, expected_output = match.groups()
    
    # Convert expected output to proper JSON types
    if expected_output.lower() == "true":
        expected_output = True
    elif expected_output.lower() == "false":
        expected_output = False
    elif expected_output.isdigit():
        expected_output = int(expected_output)
    else:
        try:
            expected_output = ast.literal_eval(expected_output)  # Safely evaluate lists or other literals
        except (ValueError, SyntaxError):
            expected_output = expected_output.strip('"')

    # Convert input parameters to a list safely
    try:
        inputs = ast.literal_eval(f"({input_str})")  # Safely handle lists, tuples, and single values
    except (ValueError, SyntaxError):
        inputs = [input_str.strip()]

    return {"input": inputs, "expected_output": expected_output}

def transform_test_list(test_list):
    """
    Transforms a list of assertions into structured test cases.
    """
    test_cases = []
    for assertion in test_list:
        parsed_case = parse_assertion(assertion)
        if parsed_case:
            test_cases.append(parsed_case)
    
    return {"test_cases": test_cases}