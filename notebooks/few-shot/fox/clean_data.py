import os
import sys
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append('../../../')
from my_packages.evaluation.midio_compiler import compile_code
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file
from my_packages.evaluation.code_evaluation import extract_code
def is_compile_ready(code: str) -> bool:
    """
    Check if the code is testable by checking if the first word contains Midio specifics.
    To avoid compiling dangerous code and save execution time.
    """
    node_modules = [ #Not critical if not updated. Code wil fail either way if it starts with thes. Just for error messages to be applied.
        "Url", 
        "Std", 
        "Http", 
        "Strings", 
        "Time", 
        "Testing", 
        "Data", 
        "Json", 
        "CSV", 
        "List", 
        "Map",
        "Iteration", 
        "Math", 
        "LinearAlgebra", 
        "Logic",
        "Scheduling",
        "Net",
        "Image",
        "File",
        "Env",
        "Buffer",
        "Sets",
        "Process",
        "Base64",
        "Hashing"
    ]
    correct_starts = ["import", "func", "module"]
    other_keywords = ["instance", "data_instance", "getter", "setter", "in", "out"]

    first_word = code.split()[0] if code.strip() else ""

    if any(kw in first_word for kw in (correct_starts+node_modules+other_keywords)):
        return True
    return False
# dir = f'{script_dir}/validation_runs/' 
dir = f'{script_dir}/testing_runs/'

for root, dirs, files in os.walk(dir):
    for file in files:
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(root, file)
        data = read_dataset_to_json(file_path)
        count = 0
        not_compile_ready = []
        for d in data:
            tasks = d["task_candidates"]
            for task_id, code_candiates in tasks.items():
                for i, code in enumerate(code_candiates):
                    
                    after_clean = extract_code(code)
                    after_clean = after_clean.replace("No Midio code found in response!", "")
                    after_clean = after_clean.lstrip("\n")

                    if code != after_clean:
                        count += 1
                        # print("Removed this part from code: " + code.replace(after_clean, ''))
                    d["task_candidates"][task_id][i] = after_clean
                    if not is_compile_ready(after_clean):
                        # print(f"Code in task: {task_id}, for file {file} is not compile ready")
                        not_compile_ready.append(after_clean.join(after_clean.split()[:1]))
        if count > 0:
            print(f"{file}: Cleaned {count} code snippets")
        print(f"{file}: not compilable list: {not_compile_ready}")
        write_json_file(file_path, data)
# print(compile_code("module", type="build", flag=""))
                