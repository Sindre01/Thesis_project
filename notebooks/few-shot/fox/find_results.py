from datetime import datetime
import os
import sys
from zoneinfo import ZoneInfo
import concurrent.futures

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
experiment_dir = os.path.abspath(f"{script_dir}/..")
results_dir = f"{project_dir}/notebooks/few-shot/fox/testing_runs"
sys.path.append(project_dir)

from my_packages.db_service.experiment_service import (
    confirm_testing_rerun,
    experiment_exists,
    pretty_print_experiment_collections,
    run_experiment_quality_checks,
    setup_experiment_collection,
)
from my_packages.db_service.data_processing import flatten_metric_results
from my_packages.common import Run
from my_packages.db_service.results_service import get_db_results, save_results_to_db
from my_packages.evaluation.code_evaluation import calculate_final_result, evaluate_code
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file
from my_packages.db_service import get_db_connection

def evaluate_testing_runs(
        file_path: str,
        env: str,
        metrics: str,
        ks: list[int],
        db_connection=None
):
    """Evaluate runs from one file for metrics and return the runs and best run."""
    print(f"🔍 Evaluating testing runs with metrics {metrics} on file {file_path}")
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
            phase="testing",
            db_connection=db_connection
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

def extract_experiment_and_model_name(file_name: str):
    experiment_name = "_".join(file_name.split("_")[:-1])
    model_name = ".".join(file_name.split("_")[-1].split(".")[:-1])  # remove .json
    return experiment_name, model_name

def extract_models_from_files(files: list[str]):
    return [extract_experiment_and_model_name(file)[1] for file in files]

def process_file(file_name: str, runs_folder: str, env: str, metrics: list[str], ks: list[int]):
    """
    Process a single file with testing runs, for an experiemnet and model.
    Returns a tuple: (experiment_name, [list of best parameter dictionaries])
    """
    file_path = os.path.join(runs_folder, file_name)
    # Extract experiment_name and model_name from file name:
    experiment_name, model_name = extract_experiment_and_model_name(file_name)
    db = get_db_connection()

    file_results = []
    results, final_results = evaluate_testing_runs(
        file_path, 
        env,
        metrics,
        ks=ks,
        db_connection=db
    )
    seeds=[run.seed for run in results]
    if env == "prod":
        save_results_to_db(
            experiment_name,
            model_name,
            seeds=seeds,
            ks=ks,
            metrics=metrics,
            result=final_results,
            db_connection=db
        )
    flattened_metrics = flatten_metric_results(final_results.metric_results)
    file_results.append(
        {
            "model_name": model_name,
            "metrics": metrics,
            "seed": seeds,
            "temperature": final_results.temperature,
            "top_p": final_results.top_p,
            "top_k": final_results.top_k,
            "ks": ks,
            "created_at": final_results.created_at,
            **flattened_metrics,
    })
    print(f"Results for {experiment_name} with model {model_name} is: {flattened_metrics}")
    return experiment_name, file_results
def main(
    env: str,
    example_selector_types: list[str],
    experiment_types: list[str],
    shots: list[int],
    metrics: list[str],
    ks: list[int],
    use_threads: bool
):
    """Find results for each experiments and model."""
    all_results = {}
    for example_selector_type in example_selector_types:
        for experiment_type in experiment_types:
            for shot in shots:
                start_time = datetime.now()
                experiment_name = f"{experiment_type}_{example_selector_type}_{shot}_shot"
                print(f"\n🔍 Finding results for experiment: {experiment_name}")
                runs_folder = f"{project_dir}/notebooks/few-shot/fox/testing_runs/{example_selector_type}/{experiment_type}"

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

                # Only files for current shot
                candidate_files = [file for file in os.listdir(runs_folder) if f"{shot}_shot" in file]
                print(f"Found {len(candidate_files)} candidate files in {runs_folder}")

                models = [extract_experiment_and_model_name(file)[1] for file in candidate_files]
                print(f"Models to process: {models}")
                skip_models = []
                for model_name in models:
                    if not confirm_testing_rerun(experiment_name, model_name):
                        print(f"🚫 Skipping testing for {experiment_name} with model {model_name}")
                        results = get_db_results(experiment_name, model_name)
                        print(f"Results for {experiment_name} with model {model_name} is: {results}")
                        all_results.setdefault(experiment_name, []).extend(results)
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
                                metrics,
                                ks
                            )
                            for file_name in candidate_files
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if result is None:
                                continue
                            experiment_name, results = result
                            all_results.setdefault(experiment_name, []).extend(results)
                else:
                    # Process candidate files sequentially
                    for file_name in candidate_files:
                        experiment_name, results = process_file(file_name, runs_folder, env, metrics, ks)
                        all_results.setdefault(experiment_name, []).extend(results)
                
                end_time = datetime.now()
                print(f"🕒 Time taken for {shot}-shot {example_selector_type} examples on {experiment_type}: {end_time - start_time}")    
            ##Write results for model on each experiement to files
            output_dir = f"{project_dir}/notebooks/few-shot/fox/results/{example_selector_type}/{experiment_type}"
            os.makedirs(output_dir, exist_ok=True)
            for experiment_name, results in all_results.items():
                print(f"writes results for {experiment_name} to file")
                write_json_file(f"{output_dir}/{experiment_name}.json", results)     


if __name__ == "__main__":
    env = "prod" # if 'prod' then it will use the MongoDB database
    ks = [1, 2, 5, 10]
    example_selector_types = ["coverage"] #["coverage", "similarity", "cot"]
    experiment_types = ["signature"]  # ["regular", "signature", "cot"]
    shots = [1]
    metrics = ["syntax", "semantic", "tests"] # ["syntax", "semantic", "tests"] or ["syntax", "semantic"]
    use_threads = True

    main(
        env = env,
        example_selector_types=example_selector_types,
        experiment_types=experiment_types,
        shots=shots,
        metrics=metrics,
        ks=ks,
        use_threads = use_threads
    )


   
    
    