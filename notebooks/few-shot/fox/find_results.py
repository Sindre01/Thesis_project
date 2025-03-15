from datetime import datetime
import os
import sys
from zoneinfo import ZoneInfo
import concurrent.futures
from filelock import FileLock

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
from my_packages.utils.file_utils import extract_experiment_and_model_name, read_dataset_to_json, write_json_file
from my_packages.db_service import get_db_connection

def evaluate_runs(
        file_path: str, 
        env: str, 
        metrics: list[str], 
        ks: list[int],
        db_connection=None,
        fold: int = None
    ) -> tuple[list[Run], dict]:
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
            fold=fold,
            db_connection=db_connection,
        )
        print(f"seed for run: {run['seed']}")

        results.append(Run(
            phase=PHASE.value,
            temperature=run["temperature"],
            seed=run["seed"],
            top_p=run["top_p"],
            top_k=run["top_k"],
            metric_results=
            { # pass@k for each metric. E.g. pass@k syntax, pass@k semantic and pass@k tests
                # e.g. {"pass@k_syntax": {pass@1: 0.1}, "pass@k semantic": {pass@1: 0.1}}
                f"pass@k_{metrics[i]}": metric_results # result is a dictionary of pass@k scores for each k value. 
                for i, metric_results in enumerate(metric_results_lists)
            },
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
        final_results = calculate_final_result(
            testing_runs=results, 
            only_mean=True if fold is not None else False
        )
        return results, final_results
    
    else:
        raise Exception(f"Invalid PHASE type. Mus be one of Enum values: {Phase.VALIDATION}, {Phase.TESTING}")

def evaluate_final_results(
        file_name: str, 
        runs_folder: str, 
        env: str, 
        metrics: list[str], 
        ks: list[int], 
        eval_method: str
)-> tuple[str, dict]:
    """Evaluate the final results for a given model and experiment. """

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
            
        for fold_idx, fold_file in enumerate(fold_files):
            file_path = os.path.join(runs_folder, fold_file)

            _, fold_result = evaluate_runs(
                file_path,
                env,
                metrics,
                ks=ks,
                db_connection=db,
                fold=fold_idx
            )
            print(f"✅ Processed fold {fold_file} with seed {fold_result.seed}")
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
        print(f"run seed 1: {all_runs[0].seed}")
        print(f"run seed 2: {all_runs[1].seed}")
        print(f"run seed 3: {all_runs[2].seed}")
        print(f"Storing result seeds: {final_result.seed}")

        save_results_to_db(
            experiment=experiment_name,
            model_name=model_name,
            seeds=final_result.seed,
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
            "seed": final_result.seed,
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
    """Evaluate the best hyperparameters for a given model and experiment. """

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

def process_model_file(
        file_name: str, 
        runs_folder: str, 
        env: str, 
        metrics: list[str], 
        ks: list[int], 
        eval_method: str
    ) -> tuple[str, dict]:
    """
    Process a single file of testing runs or validation runs, for an experiment and model.
    Supports k-fold and hold-out evaluation methods.
    Returns a tuple: (experiment_name, dict with final result)
    """
    kwargs = locals()

    model_result = {}

    if PHASE == Phase.VALIDATION:
        experiment_name, model_result = evaluate_best_params(**kwargs)

    elif PHASE == Phase.TESTING:
        experiment_name, model_result = evaluate_final_results(**kwargs)
    else:
        raise Exception(f"Invalid PHASE type. Must be one of Enum values: {Phase.VALIDATION}, {Phase.TESTING}")
    
    return experiment_name, model_result

def process_shot_file(
    shot: int,
    experiment: str, 
    runs_folder: str, 
    env: str, 
    metrics: list[str], 
    ks: list[int], 
    eval_method: str,
    model_files: list[str]
) -> tuple[str, list[dict]]:   
    
    experiment_name = f"{experiment}_{shot}_shot"
    shot_results: list[dict] =  []    
    start_time = datetime.now()
    print("PROCESSING SHOT FILES")
    print(f"🔍 Processing {shot}-shot {experiment} files in {runs_folder}")
    # print(f"Chosen models: {model_files}")
    if use_threads:
        # Process candidate files concurrently
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_model_file, 
                    file_name, 
                    runs_folder, 
                    env, 
                    metrics,
                    ks,
                    eval_method=eval_method
                )
                for file_name in model_files
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                experiment_name, model_result = result
                shot_results.append(model_result)
    else:
        # Process candidate files sequentially
        for file_name in model_files:
            experiment_name, model_result = process_model_file(
                file_name, 
                runs_folder, 
                env, 
                metrics, 
                ks, 
                eval_method=eval_method
            )
            if not model_result:
                continue
            shot_results.append(model_result)
    
    end_time = datetime.now()
    print(f"🕒 Time taken for {shot}-shot {experiment}: {end_time - start_time}")
    print(f"📊 Results for {shot}-shot {experiment}: {shot_results}")
    return experiment_name, shot_results     

def prepare_experiement(
    experiment_name: str,
    shot: int,
    runs_folder: str,
)-> tuple[list[dict], list[str]]:
    """Setup experiment collection if not exist, decide which ones to rerun and get rexisting result."""

    db = get_db_connection()
    shot_results: list[dict] = []

    print(f"\n🔍 Finding results for {eval_method} experiment: {experiment_name}")

    if experiment_exists(experiment_name, db):
        print(f"📂 Experiment {eval_method} '{experiment_name}' already exists.")
        if not use_threads:
            pretty_print_experiment_collections(
                experiment_name,
                exclude_columns=["stderr", "stdout", "code_candidate", "test_result", "error_msg"],
                filter={"eval_method": eval_method},
                db_connection=db
            )
        if not run_experiment_quality_checks(experiment_name, eval_method=eval_method, db_connection=db):
            raise Exception("Experiment quality checks failed.")
    else:
        setup_experiment_collection(experiment_name, db_connection=db)

    # Only files for current shot
    names_in_folder = [f for f in os.listdir(runs_folder)]
        
    candidate_names = [f for f in names_in_folder if f"{shot}_shot" in f]
    print(f"Found {len(candidate_names)} models for {shot}-shot in {runs_folder}")

    models = [extract_experiment_and_model_name(name)[1] for name in candidate_names]
    # print(f"Models to process: {models}")
    skip_models = []
    for model_name in models:
        if (MODEL != "") and (MODEL != model_name):
            print(f"Skipping model {model_name}")
            skip_models.append(model_name)
            continue
        
        if not confirm_rerun(experiment_name, model_name, eval_method=eval_method, phase=PHASE.value, db_connection=db):
            print(f"🚫 Skipping {PHASE} for {experiment_name} with model {model_name}. On eval method: '{eval_method}'")
            if PHASE == Phase.VALIDATION:
                results = get_db_best_params(
                    experiment=experiment_name, 
                    model=model_name, 
                    metrics=metrics, 
                    k=ks[0],
                    eval_method=eval_method,
                    db_connection=db
                )
            elif PHASE == Phase.TESTING:
                results = get_db_results(
                    experiment=experiment_name, 
                    model=model_name, 
                    eval_method=eval_method
                )
            shot_results.extend(results)
            skip_models.append(model_name)
    
    # Remove files for models that are skipped
    chosen_models = [file for file in candidate_names if extract_experiment_and_model_name(file)[1] not in skip_models]
   
    return shot_results, chosen_models

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
    # experiment_results: dict[str, list[dict]] = {}
    for example_selector_type in example_selector_types:
        for experiment_type in experiment_types:
            runs_folder = f"{project_dir}/notebooks/few-shot/fox/{PHASE.value}_runs/{example_selector_type}/{experiment_type}/{eval_method}"  
            experiment = f"{experiment_type}_{example_selector_type}"
            experiment_results: dict[str, list[dict]] = {}
            shot_files: dict[int, list[str]] = {} # Store filenames to process for each shot

            for shot in shots:
                experiment_name = f"{experiment}_{shot}_shot"
                print(f"\n🔍 Finding results for {eval_method} experiment: {experiment_name}")
                shot_results, chosen_models = prepare_experiement(
                    experiment_name,
                    shot,
                    runs_folder
                )
                print(f"Shot results: {shot_results}")
                experiment_results.setdefault(experiment_name, []).extend(shot_results)
                print(f"Experiment results: {experiment_results}")
                shot_files.setdefault(shot, []).extend(chosen_models)

            print(f"Chosen models: {shot_files}")
            if use_threads:
                # Process candidate files concurrently
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            process_shot_file, 
                            shot,
                            experiment, 
                            runs_folder, 
                            env, 
                            metrics,
                            ks,
                            eval_method=eval_method,
                            model_files=shot_files[shot]
                        )
                        for shot in shots
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result is None:
                            continue
                        experiment_name, shot_result = result
                        if not shot_result:
                            continue
                        experiment_results.setdefault(experiment_name, []).extend(shot_result)

            else:
                # Process candidate files sequentially
                for shot in shots:
                    experiment_name, shot_result = process_shot_file(
                        shot,
                        experiment, 
                        runs_folder, 
                        env, 
                        metrics,
                        ks,
                        eval_method=eval_method,
                        shot_files=shot_files[shot]
                    )
                    if not shot_result:
                        continue
                    experiment_results.setdefault(experiment_name, []).extend(shot_result)
            
            # Write results for model on each experiement to files
            output_dir = f"{project_dir}/notebooks/few-shot/fox/results/{example_selector_type}/{experiment_type}/{eval_method}/"
            os.makedirs(output_dir, exist_ok=True)

            for experiment_name, shot_result in experiment_results.items():
                print(f"writes results for {experiment_name} to file")
                write_json_file(f"{output_dir}/{experiment_name}.json", shot_result)
            
if __name__ == "__main__":
    eval_method = "hold_out"
    PHASE = Phase.TESTING
    # PHASE = Phase.VALIDATION
    MODEL = "phi4:14b-fp16" # if empty string, all models found in current experiment folders will be processed

    eval_method = "hold_out" if PHASE == Phase.VALIDATION else eval_method
    env = "prod" # if 'prod' then it will use the MongoDB database
    ks = [1, 2, 3, 5, 10]
    example_selector_types = ["similarity"] #["coverage", "similarity", "cot"]
    experiment_types = ["signature"]  # ["regular", "signature", "cot"]
    shots = [10, 5, 1]
    metrics = ["syntax", "semantic", "tests"] # or ["syntax", "semantic"]
    use_threads = True

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

