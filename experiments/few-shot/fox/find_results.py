import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
experiment_dir = os.path.abspath(f"{script_dir}/..")

sys.path.append(project_dir)
os.environ['EXPERIMENT_DB_NAME'] = "few_shot_experiments"
from my_packages.common.classes import Phase
from my_packages.evaluation.find_results import find_results

if __name__ == "__main__":
    find_results( 
        experiment_folder="few-shot",
        env="prod",
        eval_method="3_fold",
        experiment_types=["similarity"],
        prompt_types=["signature"],
        shots=[1, 5, 10],
        metrics=["syntax", "semantic", "tests", "visual"],
        ks=[1, 2, 3, 5, 10],
        use_threads=True,
        model="llama3.2:3b-instruct-fp16",
        phase=Phase.TESTING,
    )

