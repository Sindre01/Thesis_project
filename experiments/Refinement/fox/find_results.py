import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
experiment_dir = os.path.abspath(f"{script_dir}/..")
os.environ['EXPERIMENT_DB_NAME'] = "refinement_experiments" # before importing find_results and other db service functions
sys.path.append(project_dir)
from my_packages.common.classes import Phase
from my_packages.evaluation.find_results import find_results

if __name__ == "__main__":
    find_results( 
        experiment_folder="Refinement",
        env="prod",
        eval_method="3_fold",
        experiment_types=["RAG"],
        prompt_types=["signature"],
        shots=[5],
        metrics=["syntax", "semantic", "tests", "nodes"],
        ks=[1, 2, 3, 5, 10],
        use_threads=True,
        model="llama3.2:3b-instruct-fp16",
        phase=Phase.TESTING,
    )