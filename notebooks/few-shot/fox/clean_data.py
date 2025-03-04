import os
import sys
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append('../../../')
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file
from my_packages.evaluation.code_evaluation import extract_code
dir = f'{script_dir}/validation_runs/coverage/regular/'

for file in os.listdir(dir):
    print(file)
    file_path = dir + file
    data = read_dataset_to_json(file_path)

    for d in data:
        tasks = d["task_candidates"]
        for task_id, code_candiates in tasks.items():
            for i, code in enumerate(code_candiates):
                
                after_clean = extract_code(code)
                if code != after_clean:
                    print("Removed this part from code: " + code.replace(after_clean, ''))
                d["task_candidates"][task_id][i] = after_clean
    write_json_file(file_path, data)
                
      