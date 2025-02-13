import json
import os
import re
from my_packages.data_processing.code_files import extract_tests_module

def read_file(_file: str) -> str:
    with open(_file) as reader:
        return reader.read()


def read_code_file(task_id: int) -> str:
    """Reads the code file from MBPP_Midio_50/only_files/ folder."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    file_path = os.path.join(project_root, f'data/MBPP_Midio_50/only_files/task_id_{task_id}.midio')

    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error: {e}"

def write_code_file(task_id: int, code: str):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    file_path = os.path.join(project_root, f'data/MBPP_Midio_50/only_files/task_id_{task_id}.midio')
    with open(file_path, 'w') as f:
        f.write(code)
    
def extract_func_signature(code_str: str) -> str:
    """
    Extracts a function block starting with 'func(' and ending with a closing brace,
    and returns a string that contains only:
      - the header line (first line),
      - lines inside the function that start with 'in(' or 'out(' (ignoring leading whitespace),
      - the closing brace line if it exists.
    """
    # First, use a regex to extract the whole function block.
    # This pattern captures from "func(" up to the first occurrence of a closing brace on its own line.
    pattern = re.compile(r'(func\(.*?\{.*?\n\})', re.DOTALL)
    match = pattern.search(code_str)
    if not match:
        print("No function block found.")
        return "Not found"
    
    block = match.group(1)
    # Split the block into individual lines.
    lines = block.splitlines()
    if not lines:
        return ""
    
    # Assume the first line is the header.
    header = lines[0]
    
    # Determine if the last line is just the closing brace.
    closing = ""
    if lines[-1].strip() == "}":
        closing = lines[-1]
        inner_lines = lines[1:-1]
    else:
        inner_lines = lines[1:]
    
    # Filter inner lines: only keep lines starting with "in(" or "out(" (ignoring leading whitespace).
    filtered_inner = [line for line in inner_lines if line.lstrip().startswith("in(") or line.lstrip().startswith("out(")]
    
    # Reconstruct the block.
    result_lines = [header] + filtered_inner
    if closing:
        result_lines.append(closing)


    return "\n".join(result_lines)
    
def read_test_code_file(task_id):
    """Reads the code file with tests from MBPP_Midio_50/includes_tests/ folder."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
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


def read_dataset_to_json(file_path: str):
    """Reads the dataset from the file and returns it as a json object."""
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    return dataset

