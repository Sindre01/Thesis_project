from datetime import datetime
import os
import sys
from zoneinfo import ZoneInfo
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
experiment_dir = os.path.abspath(f"{script_dir}/..")
results_dir = f"{project_dir}/notebooks/few-shot/fox/testing_runs"
sys.path.append(project_dir)

from my_packages.db_service.data_processing import flatten_metric_results
from my_packages.db_service.results_service import save_results_to_db

from my_packages.common import Run
from my_packages.db_service.best_params_service import save_best_params_to_db
from my_packages.evaluation.code_evaluation import calculate_final_result, evaluate_code
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file

def evaluate_testing_runs(
        file_path: str,
        env: str,
        metrics: str,
        ks: list[int],
):
    results = []
    runs_json = read_dataset_to_json(file_path)
    for run in runs_json: # each seed
        experiment_name = run["experiment_name"]
    
        metric_results_lists = evaluate_code (
            run["task_candidates"],
            ks=ks,
            evaluation_metric=metrics,
            experiment_name=experiment_name,
            model_name=run["model"],
            env=env,
            hyperparams={
                "seed": run["seed"], 
                "temperature": run["temperature"], 
                "top_p": run["top_p"], 
                "top_k": run["top_k"]
            },
            phase="testing"
        )
        results.append(Run(
            phase="testing",
            temperature=run["temperature"],
            top_p=run["top_p"],
            top_k=run["top_k"],
            metric_results=
            { # pass@k for each metric. E.g. pass@k syntax, pass@k semantic and pass@k tests
                # e.g. {"pass@k_syntax": {pass@1: 0.1}, "pass@k semantic": {pass@1: 0.1}}
                f"pass@k_{metrics[i]}": metric_results # result is a dictionary of pass@k scores for each k value. 
                for i, metric_results in enumerate(metric_results_lists)
            },
            seed=run["seed"],
            metadata={"largest_prompt_size": run["largest_context"]}
        ))
    final_result = calculate_final_result(results)
    return results, final_result


if __name__ == "__main__":
    ks = [1, 2, 5, 10]
    phase = "testing"
    metrics = ["syntax", "semantic", "tests"]
    env = ""
    runs_folder = f"{project_dir}/notebooks/few-shot/fox/{phase}_runs"
    
    all_results = {}
    for file_name in os.listdir(runs_folder):
        file_path = os.path.join(runs_folder, file_name)
        print(f"Processing file: {file_path}")
        model_name = file_name.split("_")[-1].split(".")[0]
        experiment_name = "_".join(file_name.split("_")[:-1])
        print(f"Processing experiment: '{experiment_name}' with model: '{model_name}'")  

        results, final_results = evaluate_testing_runs(
            file_path, 
            env,
            metrics,
            ks=ks
        )
        seeds=[run.seed for run in results]
        if env == "prod":
            save_results_to_db(
                experiment_name,
                model_name,
                seeds=seeds,
                ks=ks,
                metrics=metrics,
                result=final_results
            )
        flattened_metrics = flatten_metric_results(final_results.metric_results)
        all_results.setdefault(experiment_name, []).append(
            {
            "model_name": model_name,
            "metrics": metrics,
            "seed": seeds,
            "temperature": final_results.temperature,
            "top_p": final_results.top_p,
            "top_k": final_results.top_k,
            "ks": ks,
            "created_at": datetime.now(ZoneInfo("Europe/Oslo")).isoformat(),
            **flattened_metrics,
        }
        )
        print(f"Results for {experiment_name} with model {model_name} is: {flattened_metrics}")

    ##Write results for model on each experiement to files
    for experiment_name, results in all_results.items():
        write_json_file(f"{project_dir}/notebooks/few-shot/fox/results/{experiment_name}.json", results)     
    
    