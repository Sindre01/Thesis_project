import argparse
import json
import time
import subprocess
import os
import sys

from dotenv import load_dotenv
import torch

os.environ['EXPERIMENT_DB_NAME'] = "syncode_experiments"
os.environ['HF_CACHE'] = "/cluster/work/projects/ec12/ec-sindrre/hf-models"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")

sys.path.append(project_dir)
print("Script is located in:", script_dir)
print("Project is located in:", project_dir)

from my_packages.common.few_shot import init_example_selector
from my_packages.common.rag import RagData, init_rag_data
from my_packages.common.config import model_configs
from my_packages.data_processing.split_dataset import create_kfold_splits
from my_packages.common.classes import PromptType, get_prompt_type
from my_packages.evaluation.code_evaluation import run_model, run_refinement, two_step_run
from my_packages.data_processing.attributes_processing import used_functions_to_string
from my_packages.prompting.prompt_building import transform_code_data
from my_packages.utils.file_utils import write_directly_json_file, read_dataset_to_json
from my_packages.utils.tokens_utils import get_model_code_tokens_from_file

def get_hold_out_splits(main_dataset_folder):
    train_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/hold_out/train_dataset.json'))
    val_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/hold_out/validation_dataset.json'))
    test_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/hold_out/test_dataset.json'))

    print(f"Train data: {len(train_data)}")
    print(f"Val data: {len(val_data)}")
    print(f"Test data: {len(test_data)}")
    return train_data, val_data, test_data

def get_k_fold_splits(main_dataset_folder, k, k_folds=3):

    if os.path.exists(main_dataset_folder + f'splits/{k_folds}_fold'):
        print(f"Using existing {k_folds}-fold splits")
    else:
        print(f"Creating {k_folds}-fold splits")
        train, _ ,test = get_hold_out_splits(main_dataset_folder)
        # Only use train+train for k-fold splits
        create_kfold_splits((train+test), k_folds=k_folds, write_to_file=True)

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
        dataset_nodes,
        all_nodes,
        experiment_name,
        file_path,
        model,
        example_pool,
        prompt_type: PromptType,
        temperature,
        top_p,
        top_k,
        n,
        rag_data: RagData,
        max_ctx: int,
        best_params_optimization = None,
        seeds = [3, 75, 346],
        ollama_port = "11434",
        experiment_type = "similarity",
):
    total_count = len(seeds)
    count = 0
    results = []
    for seed in seeds:
        print(f"Running with seed: {seed}")
        print(f"seeds runned: {count}/{total_count}")

        if experiment_type == "assisted-RAG":
            model_result, largest_context = two_step_run(
                client,
                model["name"],
                dataset_nodes,
                all_nodes,
                test_data,
                example_pool,
                code_max_new_tokens=model["max_tokens"],
                node_max_new_tokens=model["node_max_tokens"],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                n = n,
                seed = seed,
                debug = True,
                prompt_type = prompt_type,
                ollama_port=ollama_port,
                rag_data= rag_data,
                max_ctx=max_ctx,
                node_context_type="ONE", # One doc retrival per predicted node
                constrained_output = True
            )

        else:
            model_result, largest_context = run_model(
                client,
                model["name"],
                dataset_nodes,
                all_nodes,
                test_data,
                example_pool,
                model["max_tokens"],
                temperature,
                top_p,
                top_k,
                n = n,
                seed = seed,
                debug = True,
                prompt_type = prompt_type,
                ollama_port=ollama_port,
                rag_data=rag_data,
                max_ctx=max_ctx,
                constrained_output=True,
                refinement=True if experiment_type == "Refinement" else False,

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


def main(
        train_data,
        test_data,
        fold=-1,
        k_folds=3,
        similarity_key="task",
    ):
    """Main function to run testing experiments."""
    for ex in experiments:
        selector_type= "similarity" if ex["semantic_selector"] else "coverage"
        prompt_type = ex["prompt_type"].value

        if prompt_type == "regular":
            metrics = ["syntax", "semantic"]
        elif prompt_type == "signature":
            metrics = ["syntax", "semantic", "tests"] # ["syntax", "semantic"] or ["syntax", "semantic", "tests"]

        experiment_type = ex["name"].split("_")[1] # e.g: "RAG"
        if experiment_type == "similarity" or experiment_type == "Refinement":
            max_ctx = 16000
            #No RAG
            rag_data = None

        elif experiment_type == "RAG" or experiment_type == "assisted-RAG" or experiment_type == "all-nodes":
            max_ctx = 16000
            rag_data = init_rag_data() # None if not using RAG
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        results_dir = os.path.join("/fp/homes01/u01/ec-sindrre/slurm_jobs", f"{experiment_folder}/testing/{experiment_type}/{prompt_type}/runs/")
        best_params_folder = f"{project_dir}/experiments/{experiment_folder}/fox/best_params/{experiment_type}/{prompt_type}/hold_out"
        # best_params_folder = f"{project_dir}/experiments/{experiment_folder}/fox/best_params/test"


        for shots in ex["num_shots"]:
            selector=init_example_selector(shots, train_data, semantic_selector=ex["semantic_selector"], similarity_keys=[similarity_key])

            experiment_name = f"{ex['name']}_{shots}_shot"
            for model_name in models:

                if fold != -1: # 3-fold cross-validation
                    file_name = f"{k_folds}_fold/{experiment_name}_{model_name}/fold_{fold}.json"
                else: # Static train/val/test split
                    file_name = f"hold_out/{experiment_name}_{model_name}.json"

                result_runs_path = os.path.join(results_dir, file_name)

                print(f"\n==== Running testing for {experiment_name} on '{model_name}' ====")
                model = get_model_code_tokens_from_file(model_name, f'{project_dir}/data/max_tokens.json')
                print(f"Max generation tokens for model {model['max_tokens']}")

                best_params_path = os.path.join(best_params_folder, f"{prompt_type}_{experiment_type}_{shots}_shot" + ".json")
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
                        dataset_nodes,
                        all_nodes,
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
                        rag_data = rag_data,
                        max_ctx = max_ctx,
                        experiment_type = experiment_type,
                    )

                print(f"Testing finished for {experiment_name} on model: {model_name}")
                print(f"See run results in: {result_runs_path}")

            print(f"Testing finished for {experiment_name} on models {models}")
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\n⏱️ Total execution time: {hours}h {minutes}m {seconds}s")
            subprocess.run(["bash", f"{project_dir}/my_packages/common/push_runs.sh",
                            experiment_folder,
                            "testing",
                            selector_type,
                            prompt_type,
                            str(hours), str(minutes), str(seconds),
                            str(fold)
                            ], check=True)
            print("✅ push_runs.sh script executed successfully!")

if __name__ == "__main__":
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch sees devices:", torch.cuda.device_count())
    experiment_folder = "SynCode"
    experiment_dir = os.path.abspath(f"{script_dir}/..")
    env_path = os.path.abspath(f"{project_dir}/../../.env")
    print("Env is located in:", env_path)
    load_dotenv(env_path)
    similarity_key = "task"

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
    train_data, val_data, test_data = get_hold_out_splits(main_dataset_folder)
    dataset = train_data + val_data + test_data

    dataset_nodes = read_dataset_to_json(main_dataset_folder + "/metadata/used_external_functions.json")
    print(f"Number of nodes in datset: {len(dataset_nodes)}")

    all_nodes = read_dataset_to_json( f"{project_dir}/data/all_library_nodes.json") # All nodes
    print(f"Number all nodes: {len(all_nodes)}")

    if fold != -1:
        print(f"Using 3-fold cross-validation on merged train+test splits")
        k_fold_data = (train_data + test_data)
        train_data, test_data = get_k_fold_splits(main_dataset_folder, fold)
    else:
        print("Using static train/test split")

    all_responses = [sample["response"] for sample in dataset]
    print(f"Number of all responses: {len(all_responses)}")

    print("\n==== Configures models ====")
    client, models = model_configs(all_responses, model_provider, models, ollama_port)
    print(f"Models: {models}")

    print("\n==== Running testing ====")
    start_time = time.time()

    main(train_data, test_data, fold, similarity_key=similarity_key)







