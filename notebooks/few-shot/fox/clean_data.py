import os
import sys
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append('../../../')
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file
from my_packages.evaluation.code_evaluation import extract_code
dir = f'{script_dir}/validation_runs/coverage/regular/'
dir = f'{script_dir}/validation_runs/coverage/signature/'
dir = f'{script_dir}/validation_runs/similarity/regular/'
dir = f'{script_dir}/validation_runs/coverage/signature/'

for file in [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]:
    # if "1_shot_qwq:32b-fp16.json" not in file:
    #     continue
    print(file)
    file_path = dir + file
    data = read_dataset_to_json(file_path)
    count = 0
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
    if count > 0:
        print(f"Cleaned {count} code snippets")
    write_json_file(file_path, data)
                
      