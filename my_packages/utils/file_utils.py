import json
import os
from my_packages.common.classes import CodeEvaluationResult
from my_packages.data_processing.code_files import extract_tests_module

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def extract_all_code_and_write_to_file():
    """ Extracts code wthout tests module from the files in includes_files folder and writes to files only_files folder"""
    for i in range(50):
        code = read_test_code_file(i+1)
        print(f"Code {i+1}: {code}")
        test_module = get_test_module_from_file(i+1)
        removed_module= code.replace(test_module, "")
        print(f"Removed module: {removed_module}")
        write_code_file(i+1, removed_module)
        
def read_file(_file: str) -> str:
    with open(_file) as reader:
        return reader.read()
    
def write_json_file(root_file_path: str, content: list[dict]):
    file_path = os.path.join(project_root, root_file_path)
    with open(file_path, 'w') as writer:
        json.dump(content, writer, indent=4)

def write_directly_json_file(file_path: str, content: list[dict]):
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as writer:
        json.dump(content, writer, indent=4)

    print(f"✅ File written successfully: {file_path}")

# def append_file(root_file_path: str, obj: dict):
#     file_path = os.path.join(project_root, root_file_path)
#     with open(file_path, 'a') as writer:
#         json.dump(obj, writer, indent=4)

def append_to_ndjson(result_obj, filename="results.ndjson"):
    """Efficiently appends result objects to a newline-delimited JSON file."""
    with open(filename, "a") as f:
        f.write(json.dumps(result_obj) + "\n")

def read_code_file(task_id: int) -> str:
    """Reads the code file from MBPP_Midio_50/only_files/ folder."""
    file_path = os.path.join(project_root, f'data/MBPP_Midio_50/only_files/task_id_{task_id}.midio')

    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error: {e}"

def write_code_file(task_id: int, code: str):
    file_path = os.path.join(project_root, f'data/MBPP_Midio_50/only_files/task_id_{task_id}.midio')
    with open(file_path, 'w') as f:
        f.write(code)
    
def read_test_code_file(task_id):
    """Reads the code file with tests from MBPP_Midio_50/includes_tests/ folder."""
    file_path = os.path.join(project_root, f'data/MBPP_Midio_50/includes_tests/task_id_{task_id}_tests.midio')

    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error: {e}"
    
def get_test_module_from_file(task_id: int) -> str:
    """Reads the test code from the file."""
    code_file = read_test_code_file(task_id)
    module_tests = extract_tests_module(code_file)
    if not module_tests:
        print(f"Did NOT found tests module block for task {task_id}!!")
    return module_tests


def read_dataset_to_json(file_path: str) -> dict:
    """Reads the dataset from the file and returns it as a json object."""
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def save_results_to_file(test_results: dict[int, list[CodeEvaluationResult]], filename: str):
    """
    Saves test results to a JSON file properly formatted as a valid JSON array.
    
    - Converts CodeEvaluationResult objects to dictionaries.
    - Reads the existing JSON file (if any) and appends new results.
    - Writes the updated JSON data back as an array.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    file_path = os.path.join(project_root, f"notebooks/few-shot/logs/{filename}")

    # Ensure the file exists with an empty array if it doesn't
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([], f)  # Initialize with an empty array

    # Load existing data
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):  # Ensure it's a list
                existing_data = []
        except json.JSONDecodeError:  # Handle corrupted/malformed JSON
            existing_data = []

    # Convert new results to a dictionary and append them
    new_results = []
    for task_id, results in test_results.items():
        for result in results:
            new_results.append({
                "task_id": task_id,
                "candidate_id": result.candidate_id,
                "metric": result.metric,
                "passed": result.passed,
                "error_type": result.error_type,
                "error_msg": result.error_msg,
                "test_result": result.test_result,
                "code": result.code
            })

    # Append new results to existing data
    existing_data.extend(new_results)

    # Write back as a valid JSON array
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Results saved to {file_path} as a valid JSON array.")

def save_results_as_string(test_results: dict[int, list[CodeEvaluationResult]], filename: str):
    """
    Saves the string representation of CodeEvaluationResult objects to a file incrementally.
    """
    file_path = os.path.join(project_root, f'notebooks/few-shot/logs/{filename}')

    with open(file_path, "a", encoding="utf-8") as f:
        for task_id, results in test_results.items():
            for result in results:
                f.write(str(result))  # Call __str__() and write to file
                f.write("\n")  # Newline for readability
