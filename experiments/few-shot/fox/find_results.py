import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
experiment_dir = os.path.abspath(f"{script_dir}/..")

sys.path.append(project_dir)
from my_packages.common.classes import Phase
from my_packages.evaluation.find_results import find_results

if __name__ == "__main__":
    find_results( 
        experiment_folder="few-shot",
        env="prod",
        eval_method="3_fold",
        example_selector_types=["coverage", "similarity"],
        experiment_types=["regular", "signature"],
        shots=[1, 5, 10],
        metrics=["syntax", "semantic", "tests"],
        ks=[1, 2, 3, 5, 10],
        use_threads=True,
        model="",
        phase=Phase.TESTING,
    )

