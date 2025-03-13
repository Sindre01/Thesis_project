from datetime import datetime
import os
import sys
from zoneinfo import ZoneInfo
import concurrent.futures

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
experiment_dir = os.path.abspath(f"{script_dir}/..")

sys.path.append(project_dir)

from my_packages.db_service.results_service import get_db_results, save_results_to_db
from my_packages.db_service.experiment_service import (
    confirm_rerun,
    experiment_exists,
    pretty_print_experiment_collections,
    run_experiment_quality_checks,
    setup_experiment_collection,
)
from my_packages.db_service.data_processing import flatten_metric_results
from my_packages.common.classes import Phase, Run
from my_packages.db_service.best_params_service import get_db_best_params, save_best_params_to_db
from my_packages.evaluation.code_evaluation import calculate_best_params, calculate_final_result, evaluate_code
from my_packages.utils.file_utils import read_dataset_to_json, write_json_file
from my_packages.db_service import get_db_connection

def evaluate_runs(
        file_path: str, 
        env: str, 
        metrics: list[str], 
        ks: list[int],
        db_connection=None
    ):
    """Evaluate runs from a given file"""
    print(f"🔍 Evaluating {PHASE} runs with metric {metrics} on file {file_path}")
    results = []
    runs_json = read_dataset_to_json(file_path)
    for run in runs_json:
        experiment_name = run["experiment_name"]

        metric_results_lists = evaluate_code(
            run["task_candidates"],
            ks=ks,
            evaluation_metrics=metrics,
            experiment_name=experiment_name,
            model_name=run["model"],
            env=env,
            hyperparams={
                "seed": run["seed"],
                "temperature": run["temperature"],
                "top_p": run["top_p"],
                "top_k": run["top_k"],
            },
            phase=PHASE.value,
            eval_method=eval_method,
            db_connection=db_connection,
        )

        results.append(Run(
            phase=PHASE.value,
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
    

    if PHASE == Phase.VALIDATION:
        best_params = calculate_best_params(
            validation_runs=results, 
            metrics=metrics, 
            k=ks[0]
        )
        return results, best_params
    
    elif PHASE == Phase.TESTING:
        final_results = calculate_final_result(results, only_mean=True)
        return results, final_results
    
    else:
        raise Exception(f"Invalid PHASE type. Mus be one of Enum values: {Phase.VALIDATION}, {Phase.TESTING}")
    
def extract_experiment_and_model_name(file_name: str):
    """Extract experiment and model name from a file name and dir name in this format:
        - "regular_coverage_1_shot_llama3.3:70b-instruct-fp16.json"
        - "regular_coverage_1_shot_llama3.3:70b-instruct-fp16"
    
    """
    # Remove only .json if it exists
    if file_name.endswith(".json"):
        file_base = file_name[:-5]
    else:
        file_base = file_name

    parts = file_base.split("_")

    # find shot index
    shot_idx = next((i for i, p in enumerate(parts) if "shot" in p), None)
    if shot_idx is None:
        raise ValueError(f"Could not detect shot type in filename: {file_name}")

    experiment_name = "_".join(parts[:shot_idx + 1])
    model_name = "_".join(parts[shot_idx + 1:])
    
    return experiment_name, model_name

def extract_models_from_files(files: list[str]):
    return [extract_experiment_and_model_name(file)[1] for file in files]

def evaluate_final_results(
        file_name: str, 
        runs_folder: str, 
        env: str, 
        metrics: list[str], 
        ks: list[int], 
        eval_method: str
)-> tuple[str, dict]:
    db = get_db_connection()
    experiment_name, model_name = extract_experiment_and_model_name(file_name)
    all_runs = []
    result_dict = {}
    final_result = None

    if eval_method == "3_fold":
        folds = 3
        fold_files = [f"{file_name}/fold_{fold_idx}.json" for fold_idx in range(folds)]
        # Check if all fold files exist before evaluating
        for fold_file in fold_files:
            file_path = os.path.join(runs_folder, fold_file)
            if not os.path.exists(file_path):
                print(f"❌ Missing fold file: {file_path}. Exiting K-Fold CV for {experiment_name} for {model_name}")
                return experiment_name, {}
            
        for fold_file in fold_files:
            file_path = os.path.join(runs_folder, fold_file)

            _, fold_result = evaluate_runs(
                file_path,
                env,
                metrics,
                ks=ks,
                db_connection=db
            )
            all_runs.append(fold_result) # append the final averaged result accross the seeds

            print(f"✅ Processed fold {fold_file}")

        if not all_runs:
            print(f"❌ No valid folds found for {experiment_name} for {model_name}")
            return experiment_name, {}

        final_result = calculate_final_result(all_runs)

    elif eval_method == "hold_out":
        file_path = os.path.join(runs_folder, file_name)

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return experiment_name, {}

        all_runs, final_result = evaluate_runs(
            file_path,
            env,
            metrics,
            ks=ks,
            db_connection=db
        )
        print(f"✅ Processed {experiment_name} with model {model_name}")
    else:
        raise ValueError("Invalid evaluation method. Must be one of 'hold_out' or '3_fold'.")
    
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

def evaluate_best_params(
        file_name: str, 
        runs_folder: str, 
        env: str, 
        metrics: list[str], 
        ks: list[int], 
        eval_method: str
)-> tuple[str, dict]:
    if eval_method != "hold_out":
        raise ValueError("finding BEST_PARAMS is only supported for the 'hold_out' evaluation method.")
    
    db = get_db_connection()
    experiment_name, model_name = extract_experiment_and_model_name(file_name)
    best_params = []
       
    file_path = os.path.join(runs_folder, file_name)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return experiment_name, {}
    
    for optimizer_metric in metrics:
        best_run_result = evaluate_runs(
            file_path, 
            env, 
            metrics=[optimizer_metric], 
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
        best_params.append({
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
    
    return experiment_name, best_params

def process_file(
        file_name: str, 
        runs_folder: str, 
        env: str, 
        metrics: list[str], 
        ks: list[int], 
        eval_method: str
    ) -> tuple[str, str]:
    """
    Process a single file of testing runs or validation runs, for an experiment and model.
    Supports k-fold and hold-out evaluation methods.
    Returns a tuple: (experiment_name, dict with final result)
    """
    kwargs = locals()

    file_results = []

    if PHASE == Phase.VALIDATION:
        experiment_name, file_results = evaluate_best_params(**kwargs)

    elif PHASE == Phase.TESTING:
        experiment_name, file_results = evaluate_final_results(**kwargs)
        if not file_results:
            return experiment_name, file_results
    else:
        raise Exception(f"Invalid PHASE type. Must be one of Enum values: {Phase.VALIDATION}, {Phase.TESTING}")

    return experiment_name, file_results
        
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
                print(f"\n🔍 Finding results for {eval_method} experiment: {experiment_name}")
                runs_folder = f"{project_dir}/notebooks/few-shot/fox/{PHASE.value}_runs/{example_selector_type}/{experiment_type}/{eval_method}"

                print(f"Processing experiment '{experiment_name}'")
                
                if experiment_exists(experiment_name):
                    print(f"📂 Experiment {eval_method} '{experiment_name}' already exists.")
                    pretty_print_experiment_collections(
                        experiment_name,
                        exclude_columns=["stderr", "stdout", "code_candidate", "test_result", "error_msg"],
                        filter={"eval_method": eval_method}
                    )
                    if not run_experiment_quality_checks(experiment_name, eval_method=eval_method):
                        raise Exception("Experiment quality checks failed.")
                else:
                    setup_experiment_collection(experiment_name)

                # Only files for current shot
            
                names_in_folder = [f for f in os.listdir(runs_folder)]
                    
                candidate_names = [f for f in names_in_folder if f"{shot}_shot" in f]
                print(f"Found {len(candidate_names)} candidate names in {runs_folder}")

                models = [extract_experiment_and_model_name(name)[1] for name in candidate_names]
                print(f"Models to process: {models}")
                skip_models = []
                for model_name in models:
                    
                    if not confirm_rerun(experiment_name, model_name, eval_method=eval_method, phase=PHASE.value):
                        print(f"🚫 Skipping {PHASE} for {experiment_name} with model {model_name}. On eval method: '{eval_method}'")
                        if PHASE == Phase.VALIDATION:
                            results = get_db_best_params(
                                experiment=experiment_name, 
                                model=model_name, 
                                metrics=metrics, 
                                k=ks[0],
                                eval_method=eval_method
                            )
                        elif PHASE == Phase.TESTING:
                            results = get_db_results(
                                experiment=experiment_name, 
                                model=model_name, 
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
                        experiment_name, results = process_file(
                            file_name, 
                            runs_folder, 
                            env, 
                            metrics, 
                            ks, 
                            eval_method=eval_method
                        )
                        if not results:
                            continue
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
    # eval_method = "3_fold" # "hold_out" or "3_fold"
    eval_method = "3_fold"
    PHASE = Phase.TESTING # GLOBAL VARIABLE
    # PHASE = Phase.VALIDATION # GLOBAL VARIABLE

    eval_method = "hold_out" if PHASE == Phase.VALIDATION else eval_method
    env = "prod" # if 'prod' then it will use the MongoDB database
    ks = [1, 2, 5, 10]
    example_selector_types = ["similarity"] #["coverage", "similarity", "cot"]
    experiment_types = ["signature"]  # ["regular", "signature", "cot"]
    shots = [5]
    metrics = ["tests"] # ["syntax", "semantic", "tests"] or ["syntax", "semantic"]
    use_threads = False

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

