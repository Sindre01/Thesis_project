import os
import sys

sys.path.append('../../../')
from my_packages.utils.file_utils import read_dataset_to_json
from my_packages.evaluation.code_evaluation import extract_code
dir = './validation_runs/similarity/signature/'
for file in os.listdir(dir):
    print(file)
    data = read_dataset_to_json(dir + file)
    for d in data:
        tasks = d["task_candidates"]
        for task_id, code_candiates in tasks.items():
            for code in code_candiates:
                
                after_clean = extract_code(code)
                if code != after_clean:
                    print("Removed this part from code: " + code.replace(after_clean, ''))
                # else:
                #     print("No change in code")
                
      