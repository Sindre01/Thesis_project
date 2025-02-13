import json
import os
import re
from my_packages.data_processing.code_files import extract_tests_module

def read_file(_file: str) -> str:
    with open(_file) as reader:
        return reader.read()

def read_code_file(task_id: int) -> str:
    """Reads the code file from MBPP_Midio_50/only_files/ folder."""
    script_path = os.path.dirname(os.getcwd())
    file_path = os.path.join(script_path, f'../data/MBPP_Midio_50/only_files/task_id_{task_id}.midio')
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error: {e}"

def extract_func_signature(code_str: str) -> str:
    """Extracts the function signature from the code string."""
    # Regex explanation:
    # - Start with "func(" (including its parameters) until the opening brace '{'
    # - Then match lazily any text until encountering a line with "in("
    # - Continue matching lazily until encountering a line with "out("
    # - Continue until the first closing '}' that marks the end of the function block.
    pattern = re.compile(
        r'(func\(.*?\{.*?in\(.*?out\(.*?\})',
        re.DOTALL
    )

    match = pattern.search(code_str)
    if match:
        extracted_block = match.group(1)
        print("Extracted block:")
        print(extracted_block)
        return extracted_block
    else:
        print("No match found.")
        return "Not found"
    
def read_test_code_file(task_id):
    """Reads the code file with tests from MBPP_Midio_50/includes_tests/ folder."""
    script_path = os.path.dirname(os.getcwd())
    file_path = os.path.join(script_path, f'../data/MBPP_Midio_50/includes_tests/task_id_{task_id}_tests.midio')
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


def read_dataset_to_json(file_path: str):
    """Reads the dataset from the file and returns it as a json object."""
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    return dataset

