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
    confirm_validation_rerun,
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
        optimizer_metric: str, 
        ks: list[int],
        db_connection=None
    ):
    """Evaluate runs from one file for a given optimizer metric and return the best run."""
    print(f"🔍 Evaluating validation runs with metric {optimizer_metric} on file {file_path}")
    val_best_metric = 0.0
    best_run = Run("validation", 0.2, 0.6, 10, {f"pass@k_{optimizer_metric}": {"pass@1": 0.0}}, 9, {})

    runs_json = read_dataset_to_json(file_path)
    for run in runs_json:
        experiment_name = run["experiment_name"]

        metric_results_lists = evaluate_code(
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
                "top_k": run["top_k"],
            },
            phase="validation",
            db_connection=db_connection,
        )

        pass_at_k_dict = metric_results_lists[0]
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
    return best_run

def get_db_best_params(experiment_name, model_name, optimizer_metrics):
    """Get best parameters for a given experiment and model from the database."""
    best_params = []
    for optimizer_metric in optimizer_metrics:
        best_param = get_best_params(experiment_name, model_name, optimizer_metric, 1)
        if best_param is not None:
            flattened_metrics = flatten_metric_results(best_param.metric_results)
            params = {
                "model_name": model_name,
                "optimizer_metric": optimizer_metric,
                "seed": best_param.seed,
                "temperature": best_param.temperature,
                "top_p": best_param.top_p,
                "top_k": best_param.top_k,
                "created_at": best_param.created_at.isoformat(),
                **flattened_metrics,
            }
            best_params.append(params)
        else:
            raise Exception(f"Best parameters not found in DB for {experiment_name} and {model_name}")
    return best_params

def extract_experiment_and_model_name(file_name: str):
    experiment_name = "_".join(file_name.split("_")[:-1])
    model_name = ".".join(file_name.split("_")[-1].split(".")[:-1])  # remove .json
    return experiment_name, model_name

def extract_models_from_files(files: list[str]):
    return [extract_experiment_and_model_name(file)[1] for file in files]

def process_file(file_name: str, runs_folder: str, env: str, optimizer_metrics: list[str]):
    """
    Process a single file with validation runs, for an experiemnet and model.
    Returns a tuple: (experiment_name, [list of best parameter dictionaries])
    """
    file_path = os.path.join(runs_folder, file_name)
    # Extract experiment_name and model_name from file name:
    experiment_name, model_name = extract_experiment_and_model_name(file_name)
    db = get_db_connection()

    file_results = []
    for optimizer_metric in optimizer_metrics:
        best_run_result = evaluate_valiation_runs(file_path, env, optimizer_metric, ks=[1], db_connection=db)

        if env == "prod":
            save_best_params_to_db(experiment_name, model_name, optimizer_metric, best_run_result, db_connection=db)

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
    return experiment_name, file_results

def main(
        env: str,
        example_selector_types: list[str],
        experiment_types: list[str],
        shots: list[int],
        optimizer_metrics: list[str],
        use_threads: bool,
):
    """Find best hyperparameters for each experiment and model."""
    all_best_params = {}
    for example_selector_type in example_selector_types:
        for experiment_type in experiment_types:
            for shot in shots:
                start_time = datetime.now()
                experiment_name = f"{experiment_type}_{example_selector_type}_{shot}_shot"
                print(f"\n🔍 Finding best hyperparameters for experiment: {experiment_name}")
                runs_folder = f"{project_dir}/notebooks/few-shot/fox/validation_runs/{example_selector_type}/{experiment_type}"

                # Only files for current shot
                files_in_folder = [f for f in os.listdir(runs_folder) if os.path.isfile(os.path.join(runs_folder, f))]
                candidate_files = [file for file in files_in_folder if f"{shot}_shot" in file]
                print(f"Found {len(candidate_files)} candidate files in {runs_folder}")
                
                print(f"Processing experiment '{experiment_name}'")

                if experiment_exists(experiment_name):
                    print(f"📂 Experiment '{experiment_name}' already exists.")
                    pretty_print_experiment_collections(
                        experiment_name,
                        exclude_columns=["stderr", "stdout", "code_candidate", "test_result", "error_msg"],
                    )
                    if not run_experiment_quality_checks(experiment_name):
                        raise Exception("Experiment quality checks failed.")
                else:
                    setup_experiment_collection(experiment_name)

                models = [extract_experiment_and_model_name(file)[1] for file in candidate_files]
                print(f"Models to process: {models}")
                skip_models = []
                for model_name in models:
                    if not confirm_validation_rerun(experiment_name, model_name):
                        best_params = get_db_best_params(experiment_name, model_name, optimizer_metrics)
                        all_best_params.setdefault(experiment_name, []).extend(best_params)
                        skip_models.append(model_name)
                
                # Remove files for models that are skipped
                candidate_files = [file for file in candidate_files if extract_experiment_and_model_name(file)[1] not in skip_models]
                
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
                            for file_name in candidate_files
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if result is None:
                                continue
                            experiment_name, best_params = result
                            all_best_params.setdefault(experiment_name, []).extend(best_params)
                else:
                    # Process candidate files sequentially
                    for file_name in candidate_files:
                        experiment_name, best_params = process_file(file_name, runs_folder, env, optimizer_metrics)
                        all_best_params.setdefault(experiment_name, []).extend(best_params)

                end_time = datetime.now()
                print(f"🕒 Time taken for {shot}-shot {example_selector_type} on {experiment_type}: {end_time - start_time}")

                # Write best parameters for each experiment to file
                output_dir = f"{project_dir}/notebooks/few-shot/fox/best_params/{example_selector_type}/{experiment_type}"
                os.makedirs(output_dir, exist_ok=True)
                
                for experiment_name, best_params in all_best_params.items():
                    write_json_file(f"{output_dir}/{experiment_name}.json", best_params)


if __name__ == "__main__":
    env = "prod"
    example_selector_types = ["similarity"]  # ["coverage", "similarity"]
    experiment_types = ["signature"]  # ["regular", "signature", "cot"]
    shots = [1, 5, 10]
    optimizer_metrics = ["syntax", "semantic", "tests"]  # Separate metric evaluations
    use_threads = True
    # print(extract_experiment_and_model_name("regular_similarity_1_shot_qwen2.5-coder:32b-instruct-fp16.json"))

    main(
        env=env,
        example_selector_types=example_selector_types,
        experiment_types=experiment_types,
        shots=shots,
        optimizer_metrics=optimizer_metrics,
        use_threads=use_threads
    )
