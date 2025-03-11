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
from my_packages.common.classes import Run
from my_packages.db_service.results_service import get_db_results, save_results_to_db
from my_packages.evaluation.code_evaluation import calculate_final_result, evaluate_code
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file
from my_packages.db_service import get_db_connection

def evaluate_testing_runs(
        file_path: str,
        env: str,
        metrics: list[str],
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
    """Extract experiment and model name from a file name OR dir name in this format:
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
        metrics: list[str], 
        ks: list[int], 
        eval_method: str
    ):
    """
    Process a single file with testing runs, for an experiment and model.
    Supports k-fold and hold-out evaluation methods.
    Returns a tuple: (experiment_name, dict with final result)
    """
    db = get_db_connection()
    experiment_name, model_name = extract_experiment_and_model_name(file_name)

    all_runs = []
    final_result = None

    if eval_method == "3_fold":
        for fold_idx in range(3):
            fold_file = f"{file_name}/fold_{fold_idx}.json"
            file_path = os.path.join(runs_folder, fold_file)

            if not os.path.exists(file_path):
                print(f"⚠️ Skipping missing fold file: {file_path}")
                continue

            _, final_result = evaluate_testing_runs(
                file_path,
                env,
                metrics,
                ks=ks,
                db_connection=db
            )
            all_runs.extend(final_result) # append the final averaged result accross the seeds
            print(f"✅ Processed fold {fold_idx} for {experiment_name}")

        if not all_runs:
            print(f"❌ No valid folds found for {experiment_name}")
            return experiment_name, {}

        final_result = calculate_final_result(all_runs)

    elif eval_method == "hold_out":
        file_path = os.path.join(runs_folder, file_name)

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return experiment_name, {}

        all_runs, final_result = evaluate_testing_runs(
            file_path,
            env,
            metrics,
            ks=ks,
            db_connection=db
        )
        print(f"✅ Processed {experiment_name} with model {model_name}")

    # Save results only once outside if/else
    if env == "prod" and final_result is not None:
        seeds = [run.seed for run in all_runs]
        save_results_to_db(
            experiment=experiment_name,
            model_name=model_name,
            seeds=seeds,
            ks=ks,
            metrics=metrics,
            result=final_result,
            db_connection=db,
            eval_method=eval_method
        )

    # Flatten final result for return
    flattened_metrics = flatten_metric_results(final_result.metric_results)
    result_dict = {
        "model_name": model_name,
        "experiment": experiment_name,
        "metrics": metrics,
        "seed": [run.seed for run in all_runs],
        "temperature": final_result.temperature,
        "top_p": final_result.top_p,
        "top_k": final_result.top_k,
        "ks": ks,
        **flattened_metrics
    }

    return experiment_name, result_dict

def main(
    env: str,
    example_selector_types: list[str],
    experiment_types: list[str],
    shots: list[int],
    metrics: list[str],
    ks: list[int],
    use_threads: bool,
    eval_method: str
):
    """Find results for each experiments and model."""
    all_results = {}
    for example_selector_type in example_selector_types:
        for experiment_type in experiment_types:
            for shot in shots:
                start_time = datetime.now()
                experiment_name = f"{experiment_type}_{example_selector_type}_{shot}_shot"
                print(f"\n🔍 Finding results for experiment: {experiment_name}")
                runs_folder = f"{project_dir}/notebooks/few-shot/fox/testing_runs/{example_selector_type}/{experiment_type}/{eval_method}"

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
                names_in_folder = [f for f in os.listdir(runs_folder) if os.path.isdir(os.path.join(runs_folder, f))]
                candidate_names = [f for f in names_in_folder if f"{shot}_shot" in f]
                print(f"Found {len(candidate_names)} candidate names in {runs_folder}")

                models = [extract_experiment_and_model_name(name)[1] for name in candidate_names]
                print(f"Models to process: {models}")
                skip_models = []
                for model_name in models:
                    if not confirm_testing_rerun(experiment_name, model_name, eval_method=eval_method):
                        print(f"🚫 Skipping testing for {experiment_name} with model {model_name}")
                        results = get_db_results(
                            experiment=experiment_name, 
                            model_name=model_name, 
                            eval_method=eval_method
                        )
                        all_results.setdefault(experiment_name, []).extend(results)
                        skip_models.append(model_name)
                
                # Remove files for models that are skipped
                chosen_names = [file for file in candidate_names if extract_experiment_and_model_name(file)[1] not in skip_models]
                
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
                                ks,
                                eval_method=eval_method
                            )
                            for file_name in chosen_names
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if result is None:
                                continue
                            experiment_name, results = result
                            all_results.setdefault(experiment_name, []).extend(results)
                else:
                    # Process candidate files sequentially
                    for file_name in chosen_names:
                        experiment_name, results = process_file(file_name, runs_folder, env, metrics, ks, eval_method=eval_method)
                        all_results.setdefault(experiment_name, []).extend(results)
                
                end_time = datetime.now()
                print(f"🕒 Time taken for {shot}-shot {example_selector_type} examples on {experiment_type}: {end_time - start_time}")    
                ##Write results for model on each experiement to files
                output_dir = f"{project_dir}/notebooks/few-shot/fox/results/{example_selector_type}/{experiment_type}/{eval_method}/"
                os.makedirs(output_dir, exist_ok=True)

                for experiment_name, results in all_results.items():
                    print(f"writes results for {experiment_name} to file")
                    write_json_file(f"{output_dir}/{experiment_name}.json", results)     


if __name__ == "__main__":
    env = "prod" # if 'prod' then it will use the MongoDB database
    ks = [1, 2, 5, 10]
    example_selector_types = ["coverage"] #["coverage", "similarity", "cot"]
    experiment_types = ["signature"]  # ["regular", "signature", "cot"]
    shots = [1, 5, 10]
    metrics = ["syntax", "semantic", "tests"] # ["syntax", "semantic", "tests"] or ["syntax", "semantic"]
    use_threads = True
    eval_method = "3_fold" # "hold_out" or "3_fold"

    main(
        env = env,
        example_selector_types=example_selector_types,
        experiment_types=experiment_types,
        shots=shots,
        metrics=metrics,
        ks=ks,
        use_threads = use_threads,
        eval_method = eval_method
    )


   
    
    