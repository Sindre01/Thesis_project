from datetime import datetime
import os
import sys
from zoneinfo import ZoneInfo
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
experiment_dir = os.path.abspath(f"{script_dir}/..")
results_dir = f"{project_dir}/notebooks/few-shot/fox/validation_runs"
sys.path.append(project_dir)

from my_packages.db_service.experiment_service import confirm_validation_rerun, experiment_exists, pretty_print_experiment_collections, run_experiment_quality_checks, setup_experiment_collection
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
    print(f"🔍 Evaluating validation runs with metric {optimizer_metric}")
    val_best_metric = 0.0
    best_run = Run("validation", -1, -1, -1, {f"pass@k_{optimizer_metric}": {"pass@1": 0.0}}, -1, {})

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
            phase="validation"
        )
        ## Optimizing for the first k in the ks list
        pass_at_k_dict = metric_results_lists[0]
        val_metric = pass_at_k_dict[f"pass@{ks[0]}"]
        print(f"Validation with temp={run['temperature']}, top_k={run['top_k']} and top_p={run['top_p']}. Gave {optimizer_metric}@{ks[0]}={val_metric}")

        #Optimize for the best pass@ks[0] for the provided metric
        if val_metric > val_best_metric:
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
    env = ""
    optimizer_metrics = ["syntax", "semantic", "tests"] # find best parameters seperately for each metric
    example_selector_types = ["coverage", "similarity"]
    
    all_best_params = {}
    for example_selector_type in example_selector_types:
        print(f"🔍 Finding best hyperparameters for {example_selector_type} examples")
        runs_folder = f"{project_dir}/notebooks/few-shot/fox/validation_runs/{example_selector_type}"
        for file_name in os.listdir(runs_folder):
            file_path = os.path.join(runs_folder, file_name)
            print(f"Processing file: {file_path}")
            experiment_name = "_".join(file_name.split("_")[:-1])
            model_name = file_name.split("_")[-1].split(".")[0]
            print(f"Processing experiment: '{experiment_name}' with model: '{model_name}'")  

            if experiment_exists(experiment_name):
                # delete_experiment(experiment_name)
                print(f"📂 Experiemnt '{experiment_name}' already exists.")
                pretty_print_experiment_collections(
                    experiment_name,
                    exclude_columns=["stderr", "stdout", "code_candidate", "test_result", "error_msg"]
                )
                if not run_experiment_quality_checks(experiment_name):
                    print("\n❌ Experiment quality checks failed. Exiting.")
                    raise Exception("Experiment quality checks failed.")
            else:
                setup_experiment_collection(experiment_name)

            if not confirm_validation_rerun(experiment_name, model_name):
                print(f"🚫 Skipping validation for {experiment_name} with model {model_name}")
                continue

            for optimizer_metric in optimizer_metrics: # optimizing for each metric seperately
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

                ## Save best params to dict for later storing in file. For use in Fox
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
                print(f"Best hyperparameters for {experiment_name} with model {model_name} is: temp = {best_run_result.temperature}, top_p = {best_run_result.top_p}, top_k = {best_run_result.top_k}, seed = {best_run_result.seed}, {flattened_metrics}\n\n")
            
    ##Write best params for model on each experiment to files
    for experiment_name, best_params in all_best_params.items():
        write_json_file(f"{project_dir}/notebooks/few-shot/fox/best_params/{experiment_name}.json", best_params)     
        
        