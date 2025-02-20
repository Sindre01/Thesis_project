from datetime import datetime
import os
import sys
from zoneinfo import ZoneInfo

root_dir = os.getcwd()
results_dir = f"{root_dir}/notebooks/few-shot/fox/validation_runs"
sys.path.append(root_dir)
from my_packages.db_service.data_processing import flatten_metric_results

from my_packages.common import Run
from my_packages.db_service.best_params_service import save_best_params_to_db
from my_packages.evaluation.code_evaluation import evaluate_code
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file

def evaluate_valiation_runs(
        file_path: str,
        env: str,
        optimizer_metric: str,
        ks: list[int],
):
    val_best_metric = 0.0
    best_run = None

    runs_json = read_dataset_to_json(file_path)
    for run in runs_json:
        experiment_name = run["experiment_name"]
    
        metric_results_lists = evaluate_code (
            run["task_candidates"],
            ks=ks,
            evaluation_metric=[optimizer_metric],
            experiment_name=experiment_name,
            model_name=run["model"],
            env=env,
            hyperparams={
                "seed": run["seed"], 
                "temperature": run["temperature"], 
                "top_p": run["top_p"], 
                "top_k": run["top_k"]
            },
            phase=phase
        )
        ## Optimizing for the first k in the ks list
        pass_at_k_dict = metric_results_lists[0]
        val_metric = pass_at_k_dict[f"pass@{ks[0]}"]
        print(f"Validation with temp={run['temperature']}, top_k={run['top_k']} and top_p={run['top_p']}. Gave pass@{ks[0]}={val_metric} and pass@ks={pass_at_k_dict}")

        #Optimize for the best pass@ks[0] for the provided metric
        if val_metric > val_best_metric or best_run is None:
            print(f"New best pass@{ks[0]} found, {val_metric}")
            val_best_metric = val_metric
            best_run = Run(
                phase="validation",
                temperature=run["temperature"],
                top_p=run["top_p"],
                top_k=run["top_k"],
                metric_results={f"pass@k_{optimizer_metric}": pass_at_k_dict},
                seed=run["seed"],
                metadata={"largest_prompt_size": run["largest_context"]}
            )
    return best_run


if __name__ == "__main__":
    phase = "validation"
    optimizer_metric = "semantic"
    env = ""
    runs_folder = f"{root_dir}/notebooks/few-shot/fox/{phase}_runs"
    
    all_best_params = {}
    for file_name in os.listdir(runs_folder):
        file_path = os.path.join(runs_folder, file_name)
        print(f"Processing file: {file_path}")
        model_name = file_name.split("_")[-1].split(".")[0]
        experiment_name = "_".join(file_name.split("_")[:-1])
        print(f"Processing experiment: '{experiment_name}' with model: '{model_name}'")  

        best_run_result = evaluate_valiation_runs(
            file_path, 
            env,
            optimizer_metric,
            ks=[1]
        )
        if env == "prod":
            save_best_params_to_db(
                experiment_name, 
                model_name, 
                optimizer_metric, 
                best_run_result
            )
        flattened_metrics = flatten_metric_results(best_run_result.metric_results)
        all_best_params.setdefault(experiment_name, []).append({
            "model_name": model_name,
            "optimizer_metric": optimizer_metric,
            "temperature": best_run_result.temperature,
            "top_p": best_run_result.top_p,
            "top_k": best_run_result.top_k,
            "seed": best_run_result.seed,
            "created_at": datetime.now(ZoneInfo("Europe/Oslo")).isoformat(),
            **flattened_metrics,
        })
        print(f"Best hyperparameters for {experiment_name} with model {model_name} is: temp = {best_run_result.temperature}, top_p = {best_run_result.top_p}, top_k = {best_run_result.top_k}, seed = {best_run_result.seed}, {flattened_metrics}")
        
    ##Write best params for model on each experiment to files
    for experiment_name, best_params in all_best_params.items():
        write_json_file(f"{root_dir}/notebooks/few-shot/fox/best_params/{experiment_name}.json", best_params)     
        
        