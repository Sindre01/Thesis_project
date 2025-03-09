import argparse
import json
import time
import subprocess
import os
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
# experiment_dir = os.path.abspath(f"{script_dir}/..")
env_path = os.path.abspath(f"{project_dir}/../../.env")
results_dir = f"{project_dir}/notebooks/few-shot/fox/testing_runs"
sys.path.append(project_dir)
print("Script is located in:", script_dir)
print("Project is located in:", project_dir)
print("Env is located in:", env_path)

from my_packages.common.few_shot import init_example_selector, model_configs
from my_packages.data_processing.split_dataset import create_kfold_splits
from my_packages.common.classes import PromptType, get_prompt_type
from my_packages.evaluation.code_evaluation import run_model
# from my_packages.experiments.few_shot import get_dataset_splits, init_example_selector, model_configs
from dotenv import load_dotenv
from pathlib import Path
from my_packages.data_processing.attributes_processing import used_functions_to_string
from my_packages.prompting.few_shot import transform_code_data
from my_packages.utils.file_utils import write_directly_json_file, write_json_file, read_dataset_to_json
from my_packages.utils.tokens_utils import get_model_code_tokens_from_file, models_not_in_file, write_models_tokens_to_file
from my_packages.utils.server_utils import server_diagnostics, is_remote_server_reachable
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from my_packages.prompting.example_selectors import get_coverage_example_selector, get_semantic_similarity_example_selector
from langchain_core.example_selectors.base import BaseExampleSelector

def get_dataset_splits(main_dataset_folder):
    train_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/train_dataset.json'))
    val_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/validation_dataset.json'))
    test_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/test_dataset.json'))

    print(f"Train data: {len(train_data)}")
    print(f"Val data: {len(val_data)}")
    print(f"Test data: {len(test_data)}")
    return train_data, val_data, test_data

def get_k_fold_splits(main_dataset_folder, k, k_folds=5):

    if os.path.exists(main_dataset_folder + f'splits/{k_folds}_fold'):
        print(f"Using existing {k_folds}-fold splits")
    else:
        print(f"Creating {k_folds}-fold splits")
        train, test = get_dataset_splits(main_dataset_folder)
        create_kfold_splits((test+train), k_folds=k_folds, write_to_file=True)

    train_data = read_dataset_to_json(main_dataset_folder + f'splits/{k_folds}_fold/train_dataset_{k}.json')
    test_data = read_dataset_to_json(main_dataset_folder + f'splits/{k_folds}_fold/test_dataset_{k}.json')

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")
    return train_data, test_data

def get_info_from_filename(file_name: str):
    experiment_name = file_name.split(".")[0]
    parts = file_name.split("_")
    prompt_type = get_prompt_type(parts[0])
    example_selector = parts[1]
    shots = int(parts[2])
    
    return {
        "experiment_name": experiment_name,
        "prompt_type": prompt_type,
        "semantic_selector": example_selector == "similarity",
        "shots": shots
    }
    
def parse_experiments(experiment_list):
    """Convert dictionary input to proper PromptType Enum where needed."""
    for exp in experiment_list:
        if "prompt_type" in exp and isinstance(exp["prompt_type"], str):
            exp["prompt_type"] = PromptType(exp["prompt_type"])  # Convert to Enum
    return experiment_list

def run_testing_experiment(
        client,
        test_data,
        available_nodes,
        experiment_name,
        file_path,
        model,
        example_pool,
        prompt_type: PromptType,
        temperature,
        top_p,
        top_k,
        n,
        best_params_optimization = None,
        seeds = [3, 75, 346],
        ollama_port = "11434"
):
    total_count = len(seeds)
    count = 0
    results = []
    for seed in seeds:
        print(f"Running with seed: {seed}")
        print(f"seeds runned: {count}/{total_count}")
        model_result, largest_context = run_model(
            client,
            model["name"],
            available_nodes,
            test_data,
            example_pool,
            model["max_tokens"],
            temperature,
            top_p,
            top_k,
            n,
            seed,
            False, 
            prompt_type,
            ollama_port=ollama_port
        )
        result_obj = {
            "experiment_name": experiment_name,
            "best_params_optimization": best_params_optimization,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "n_generations_per_task": n,
            "model": model["name"],
            "largest_context": largest_context,
            "task_candidates": model_result,
        }

        results.append(result_obj)
        ## Write to file
        write_directly_json_file(file_path, results)#Temporary viewing
        count+=1
    
    write_directly_json_file(file_path, results)


def main(train_data, test_data, fold=-1):
    """Main function to run few-shot testing experiments."""
    for ex in experiments:
        selector_type= "similarity" if ex["semantic_selector"] else "coverage"
        prompt_type = ex["prompt_type"].value
        if prompt_type == "regular":
            metrics = ["syntax", "semantic"]
        elif prompt_type == "signature":
            metrics = ["syntax", "semantic", "tests"] # ["syntax", "semantic"] or ["syntax", "semantic", "tests"]
            
        results_dir = os.path.join("/fp/homes01/u01/ec-sindrre/slurm_jobs", f"few-shot/testing/{selector_type}/{prompt_type}/runs/")
        best_params_folder = f"{project_dir}/notebooks/few-shot/fox/best_params/{selector_type}/{prompt_type}"

        for shots in ex["num_shots"]:
            selector=init_example_selector(shots, train_data, semantic_selector=ex["semantic_selector"])
            experiment_name = f"{ex['name']}_{shots}_shot"

            for model_name in models:
                file_name = f"{experiment_name}_{model_name}.json"
                if fold != -1:
                    file_name = f"{experiment_name}_{model_name}/fold_{fold}.json"

                result_runs_path = os.path.join(results_dir, file_name)
                best_params_path = os.path.join(best_params_folder, experiment_name + ".json")

                print(f"\n==== Running few-shot testing for {experiment_name} on '{model_name}' ====")  
                model = get_model_code_tokens_from_file(model_name, f'{project_dir}/notebooks/few-shot/code_max_tokens.json')
                print(f"Max generation tokens for model {model['max_tokens']}")
                best_params = read_dataset_to_json(best_params_path)

                if not best_params:
                    print(f"Skipping {model_name} since no best parameters found")
                    continue

                # Find best params for model on each metric
                best_params_for_metrics = {}
                for metric in metrics:
                    best_params_for_metrics[metric] = next(
                        (params for params in best_params 
                        if params["model_name"] == model_name and params["optimizer_metric"] == metric),
                        None 
                    )

                print(f"Best parameters for {model_name} on each metric: {best_params_for_metrics}")
                optimizer_metric = metrics[-1]  # Best params for last metric in list, e.g: 'semantic' or 'tests' metric
                best_params_model = best_params_for_metrics[optimizer_metric]
                if optimizer_metric == "tests":
                    if best_params_model["tests@1"] == 0.0:
                        optimizer_metric = "semantic"
                        best_params_model = best_params_for_metrics[optimizer_metric]
                        print("No best params for tests@1 over 0.0 found. Using semantic metric instead.")

                print(f"Best parameters for {model_name} on {metrics[-1]} metric: {best_params_model}")
                run_testing_experiment(
                        client,
                        test_data,
                        available_nodes,
                        experiment_name,
                        result_runs_path,
                        model,
                        selector,
                        ex["prompt_type"],
                        best_params_model["temperature"],
                        best_params_model["top_p"],
                        best_params_model["top_k"],
                        n_generations_per_task,
                        best_params_optimization = best_params_model["optimizer_metric"],
                        ollama_port = ollama_port,
                    )

                print(f"Testing finished for {experiment_name} on model: {model_name}")
                print(f"See run results in: {result_runs_path}")

            print(f"Testing finished for {experiment_name} on models {models}")
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\n⏱️ Total execution time: {hours}h {minutes}m {seconds}s")
            subprocess.run(["bash", f"{project_dir}/notebooks/few-shot/fox/scripts/push_runs.sh", 
                            "few-shot", 
                            "testing", 
                            selector_type, 
                            prompt_type,
                            str(hours), str(minutes), str(seconds)], check=True)
            print("✅ push_runs.sh script executed successfully!")
            
if __name__ == "__main__":
    # Parse arguments:
    parser = argparse.ArgumentParser(description="Process input.")

    parser.add_argument("--model_provider", type=str, required=True, help="Model provider")
    parser.add_argument("--models", type=str, required=True, help="JSON string for models")
    parser.add_argument("--experiments", type=str, required=True, help="JSON string for experiments")
    parser.add_argument("--ollama_port", type=str, required=True, help="ollama_port")
    parser.add_argument("--fold", type=int, required=True, help="fold")

    args = parser.parse_args()
    # DEBUG: Print arguments before decoding JSON
    print("🛠️ Debug: Received --models =", repr(args.models))
    print("🛠️ Debug: Received --experiments =", repr(args.experiments))
    print("🛠️ Debug: Received --model_provider =", repr(args.model_provider))
    print("🛠️ Debug: Received --ollama_port =", repr(args.ollama_port))
    print("🛠️ Debug: Received --fold =", repr(args.fold))
    
    model_provider = args.model_provider
    models = json.loads(args.models)
    experiments = json.loads(args.experiments)
    ollama_port = args.ollama_port
    experiments = parse_experiments(experiments)
    fold = args.fold

    print("########### Parsed arguments ###########")
    print(f"Model provider: {model_provider}") 
    print(f"Models: {models}")
    print(f"Experiments: {experiments}")
    print(f"Ollama port: {ollama_port}")
    print(f"Fold: {fold}")
    print("########################################")
    n_generations_per_task = 10
    
    start_time = time.time()
    main_dataset_folder = f'{project_dir}/data/MBPP_Midio_50/'

    print("\n==== Splits data ====")
    train_data, val_data, test_data = get_dataset_splits(main_dataset_folder)
    dataset = train_data + val_data + test_data

    if fold != -1:
        print(f"Using {fold}-fold cross-validation on merged train+test splits")
        k_fold_data = (train_data + test_data)  
        train_data, test_data = get_k_fold_splits(main_dataset_folder, fold)
    else:
        print("Using static train/test split")
    
    all_responses = [sample["response"] for sample in dataset]
    print(f"Number of all responses: {len(all_responses)}")
    
    used_functions_json = read_dataset_to_json(main_dataset_folder + "metadata/used_external_functions.json") # Used functions in the dataset
    print(f"Number of used nodes: {len(used_functions_json)}")
    available_nodes = used_functions_to_string(used_functions_json) #used for Context prompt

    print("\n==== Configures models ====")
    client, models = model_configs(all_responses, model_provider, models, ollama_port)
    print(f"Models: {models}")

    print("\n==== Running testing ====")
    start_time = time.time()

    main(train_data, test_data, fold)
    



        


