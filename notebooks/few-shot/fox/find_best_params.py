from datetime import datetime
import os
import sys
from zoneinfo import ZoneInfo
import concurrent.futures

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
experiment_dir = os.path.abspath(f"{script_dir}/..")
results_dir = f"{project_dir}/notebooks/few-shot/fox/validation_runs"
sys.path.append(project_dir)

from my_packages.db_service.experiment_service import (
    confirm_rerun,
    experiment_exists,
    pretty_print_experiment_collections,
    run_experiment_quality_checks,
    setup_experiment_collection,
)
from my_packages.db_service.data_processing import flatten_metric_results
from my_packages.common.classes import Run
from my_packages.db_service.best_params_service import get_best_params, save_best_params_to_db
from my_packages.evaluation.code_evaluation import evaluate_code
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file
from my_packages.db_service import get_db_connection


def evaluate_valiation_runs(
        file_path: str, 
        env: str, 
        metrics: list[str], 
        ks: list[int],
        db_connection=None
    ):
    """Evaluate runs from one file for a given optimizer metric and return the best run."""
    print(f"🔍 Evaluating validation runs with metric {metrics} on file {file_path}")
    results = []
    runs_json = read_dataset_to_json(file_path)
    for run in runs_json:
        experiment_name = run["experiment_name"]

        metric_results_lists = evaluate_code(
            run["task_candidates"],
            ks=ks,
            evaluation_metrics=[metrics],
            experiment_name=experiment_name,
            model_name=run["model"],
            env=env,
            hyperparams={
                "seed": run["seed"],
                "temperature": run["temperature"],
                "top_p": run["top_p"],
                "top_k": run["top_k"],
            },
            phase="validation",
            db_connection=db_connection,
        )

        results.append(Run(
            phase="validation",
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

    best_run = calculate_best_run(results)
    return results, best_run

def calculate_best_run(
        validation_runs: list[Run],
):
    """Calculate the best run from the validation runs"""
    
    val_best_metric = 0.0
    best_run = Run("validation", 0.2, 0.6, 10, {f"pass@k_{optimizer_metric}": {"pass@1": 0.0}}, 9, {})
    for run in validation_runs:
        
        if len(run.metric_results) != 1:
            raise ValueError(f"Expected one metric result, got {len(run.metric_results)}")
        
        pass_at_k_dict = run.metric_results[0]
        val_metric = pass_at_k_dict[f"pass@{ks[0]}"]
        # (Optional: remove or lower verbosity of print statements)
        # print(f"Validation with temp={run['temperature']}, top_k={run['top_k']}, top_p={run['top_p']} -> {optimizer_metric}@{ks[0]}={val_metric}")

        if val_metric > val_best_metric:
            # print(f"New best pass@{ks[0]} found: {val_metric}")
            val_best_metric = val_metric
            best_run = Run(
                phase="validation",
                temperature=run["temperature"],
                top_p=run["top_p"],
                top_k=run["top_k"],
                metric_results={f"pass@k_{optimizer_metric}": pass_at_k_dict},
                seed=run["seed"],
                metadata={"largest_prompt_size": run["largest_context"]},
            )


def get_db_best_params(experiment_name, model_name, optimizer_metrics, eval_method: str):
    """Get best parameters for a given experiment and model from the database."""
    best_params = []
    for optimizer_metric in optimizer_metrics:
        best_param = get_best_params(experiment_name, model_name, optimizer_metric, 1, eval_method=eval_method)
        if best_param is not None:
            flattened_metrics = flatten_metric_results(best_param.metric_results)
            params = {
                "model_name": model_name,
                "optimizer_metric": optimizer_metric,
                "seed": best_param.seed,
                "temperature": best_param.temperature,
                "top_p": best_param.top_p,
                "top_k": best_param.top_k,
                "eval_method": eval_method,
                "created_at": best_param.created_at.isoformat(),
                **flattened_metrics,
            }
            best_params.append(params)
        else:
            raise Exception(f"Best parameters not found in DB for {experiment_name} and {model_name}")
    return best_params

def extract_experiment_and_model_name(file_name: str):
    """Extract experiment and model name from a file name and dir name in this format:
        - "regular_coverage_1_shot_llama3.3:70b-instruct-fp16.json"
        - "regular_coverage_1_shot_llama3.3:70b-instruct-fp16"
    
    """
    file_base = os.path.splitext(file_name)[0]
    parts = file_base.split("_")
    experiment_name = "_".join(parts[:-1])
    model_name = parts[-1]
    
    return experiment_name, model_name

def extract_models_from_files(files: list[str]):
    return [extract_experiment_and_model_name(file)[1] for file in files]

def process_file(
        file_name: str, 
        runs_folder: str, 
        env: str, 
        optimizer_metrics: list[str], 
        eval_method: bool = False,
        ks = [1],
    ):
    """
    Process a single file with validation runs, for an experiemnet and model.
    Returns a tuple: (experiment_name, [list of best parameter dictionaries])
    """
    db = get_db_connection()
    # Extract experiment_name and model_name from file name:
    experiment_name, model_name = extract_experiment_and_model_name(file_name)
    

    file_results = []
    if eval_method == "3_fold":
        # file_path = os.path.join(runs_folder, file_name, eval_method)
        # for fold_idx in range(3):
        #     fold_file = f"{file_name}/fold_{fold_idx}.json"
        #     file_path = os.path.join(runs_folder, fold_file)

        #     if not os.path.exists(file_path):
        #         print(f"⚠️ Skipping missing fold file: {file_path}")
        #         continue

        print(f"Using k-fold validation")
        print("Not implemented yet!")
        raise NotImplementedError
    elif eval_method == "hold_out":
        file_path = os.path.join(runs_folder, file_name)
        print(f"Using hold-out validation")

        for optimizer_metric in optimizer_metrics:
            best_run_result = evaluate_valiation_runs(
                file_path, 
                env, 
                optimizer_metric, 
                ks=ks, 
                db_connection=db
            )

            if env == "prod":
                save_best_params_to_db(
                    experiment=experiment_name,
                    model_name=model_name, 
                    optimizer_metric=optimizer_metric, 
                    best_params=best_run_result, 
                    db_connection=db,
                    eval_method=eval_method
                )

            flattened_metrics = flatten_metric_results(best_run_result.metric_results)
            file_results.append({
                "model_name": model_name,
                "optimizer_metric": optimizer_metric,
                "seed": best_run_result.seed,
                "temperature": best_run_result.temperature,
                "top_p": best_run_result.top_p,
                "top_k": best_run_result.top_k,
                "created_at": datetime.now(ZoneInfo("Europe/Oslo")).isoformat(),
                **flattened_metrics,
            })
            print(f"Best hyperparameters for {experiment_name} with model {model_name}: temp={best_run_result.temperature}, top_p={best_run_result.top_p}, top_k={best_run_result.top_k}, seed={best_run_result.seed}")
    else:
        raise ValueError(f"Invalid eval_method: {eval_method}")
    
    return experiment_name, file_results

def main(
        env: str,
        example_selector_types: list[str],
        experiment_types: list[str],
        shots: list[int],
        optimizer_metrics: list[str],
        use_threads: bool,
        eval_method: str
):
    """Find best hyperparameters for each experiment and model."""
    all_best_params = {}
    for example_selector_type in example_selector_types:
        for experiment_type in experiment_types:
            for shot in shots:
                start_time = datetime.now()
                experiment_name = f"{experiment_type}_{example_selector_type}_{shot}_shot"
                print(f"\n🔍 Finding best hyperparameters for experiment: {experiment_name}")
                runs_folder = f"{project_dir}/notebooks/few-shot/fox/validation_runs/{example_selector_type}/{experiment_type}/{eval_method}"
                
                print(f"Processing experiment '{experiment_name}'")

                if experiment_exists(experiment_name):
                    print(f"📂 Experiment '{experiment_name}' already exists.")
                    pretty_print_experiment_collections(
                        experiment_name,
                        exclude_columns=["stderr", "stdout", "code_candidate", "test_result", "error_msg"],
                        filter={"eval_method": eval_method}
                    )
                    if not run_experiment_quality_checks(experiment_name):
                        raise Exception("Experiment quality checks failed.")
                else:
                    setup_experiment_collection(experiment_name)
                
                # Only files for current shot
                names_in_folder = [f for f in os.listdir(runs_folder) if os.path.isdir(os.path.join(runs_folder, f))]
                candidate_names = [f for f in names_in_folder if f"{shot}_shot" in f]
                print(f"Found {len(candidate_names)} candidate names in {runs_folder}")
                models = extract_models_from_files(candidate_names)

                skip_models = []
                print(f"Models to process: {models}")
                for model_name in models:
                    if not confirm_rerun(experiment_name, model_name, eval_method, phase="validation"):
                        print(f"🚫 Skipping testing for {experiment_name} with model {model_name}")
                        best_params = get_db_best_params(
                            experiment_name, 
                            model_name, 
                            optimizer_metrics,
                            eval_method=eval_method
                        )
                        all_best_params.setdefault(experiment_name, []).extend(best_params)
                        skip_models.append(model_name)
                
                # Remove files for models that are skipped
                chosen_names = [file for file in chosen_names if extract_experiment_and_model_name(file)[1] not in skip_models]
                
                if use_threads:
                    # Process candidate files concurrently
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        futures = [
                            executor.submit(
                                process_file, 
                                file_name, 
                                runs_folder, 
                                env, 
                                optimizer_metrics
                            )
                            for file_name in chosen_names
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if result is None:
                                continue
                            experiment_name, best_params = result
                            all_best_params.setdefault(experiment_name, []).extend(best_params)
                else:
                    # Process candidate files sequentially
                    for file_name in chosen_names:
                        experiment_name, best_params = process_file(file_name, runs_folder, env, optimizer_metrics)
                        all_best_params.setdefault(experiment_name, []).extend(best_params)

                end_time = datetime.now()
                print(f"🕒 Time taken for {shot}-shot {example_selector_type} on {experiment_type}: {end_time - start_time}")

                # Write best parameters for each experiment to file
                output_dir = f"{project_dir}/notebooks/few-shot/fox/best_params/{example_selector_type}/{experiment_type}/{eval_method}/"
                os.makedirs(output_dir, exist_ok=True)

                for experiment_name, results in all_best_params.items():
                    print(f"writes results for {experiment_name} to file")
                    write_json_file(f"{output_dir}/{experiment_name}.json", results)     

if __name__ == "__main__":
    env = "prod"
    example_selector_types = ["similarity"]  # ["coverage", "similarity"]
    experiment_types = ["signature"]  # ["regular", "signature", "cot"]
    shots = [1, 5, 10]
    optimizer_metrics = ["syntax", "semantic", "tests"]  # Separate metric evaluations
    use_threads = True
    eval_method = "hold_out" # Not working yet
    # print(extract_experiment_and_model_name("regular_similarity_1_shot_qwen2.5-coder:32b-instruct-fp16.json"))

    main(
        env=env,
        example_selector_types=example_selector_types,
        experiment_types=experiment_types,
        shots=shots,
        optimizer_metrics=optimizer_metrics,
        use_threads=use_threads
    )
