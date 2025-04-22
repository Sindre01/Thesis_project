import json
import time
import os
import subprocess
import sys
import argparse

from dotenv import load_dotenv
from my_packages.common.rag import RagData, init_rag_data
os.environ['EXPERIMENT_DB_NAME'] = "syncode_experiments"
os.environ['HF_CACHE'] = "/cluster/work/projects/ec12/ec-sindrre/hf-models"

from my_packages.common.few_shot import init_example_selector
from my_packages.common.config import model_configs
from my_packages.data_processing.split_dataset import create_kfold_splits
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")

print("Script is located in:", script_dir)
print("Project is located in:", project_dir)

sys.path.append(project_dir)
from my_packages.common.classes import PromptType
from my_packages.evaluation.code_evaluation import run_model
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

def get_k_fold_splits(main_dataset_folder, k, k_folds=5):

    if os.path.exists(main_dataset_folder + f'splits/{k_folds}_fold'):
        print(f"Using existing {k_folds}-fold splits")
    else:
        print(f"Creating {k_folds}-fold splits")
        train, val, _ = get_hold_out_splits(main_dataset_folder)
        create_kfold_splits((val+train), k_folds=k_folds, write_to_file=True)

    train_data = read_dataset_to_json(main_dataset_folder + f'splits/{k_folds}_fold/train_dataset_{k}.json')
    val_data = read_dataset_to_json(main_dataset_folder + f'splits/{k_folds}_fold/val_dataset_{k}.json')

    print(f"Train data: {len(train_data)}")
    print(f"val data: {len(val_data)}")
    return train_data, val_data


def run_val_experiment(
        client,
        val_data,
        dataset_nodes: list[dict],
        all_nodes: list[dict],
        experiment_name,
        result_runs_path,
        model,
        example_pool,
        prompt_type: PromptType,
        rag_data: RagData,
        max_ctx: int,
        experiment_type,
        temperatures = [0.2, 0.6, 0.9],
        top_ps = [0.2, 0.6, 0.9],
        top_ks = [],
        n = 1, # Max value of array is generations per task
        seed = 9,
        debug = False,
        ollama_port = "11434",
):
    
    if model["name"] in "gpt-4o":
        print(f"Not using top_k for {model['name']} model")
        top_ks = []
        combinatios = len(temperatures) * len(top_ps)
    else:
        combinatios = len(temperatures) * len(top_ps) * len(top_ks)

    current_combination = 0
    results = []

    for temp in temperatures:
        for top_k in top_ks or [-1]: #Ensures loop runs once, when top_ks is empty
            for top_p in top_ps:
                print(f"Validating with temperature: {temp}, top_k: {top_k} and top_p: {top_p}")
                
                model_result, largest_context = run_model(
                    client,
                    model=model["name"],
                    dataset_nodes=dataset_nodes,
                    all_nodes=all_nodes,
                    data=val_data,
                    example_pool=example_pool,
                    max_new_tokens=model["max_tokens"],
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                    n = n,
                    seed = seed,
                    debug = debug, 
                    prompt_type = prompt_type,
                    ollama_port=ollama_port,
                    rag_data=rag_data,
                    max_ctx=max_ctx,
                    constrained_output=True
                )
                result_obj = {
                    "experiment_name": experiment_name,
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "seed": seed,
                    "n_generations_per_task": n,
                    "model": model["name"],
                    "largest_context": largest_context,
                    "task_candidates": model_result,
                }
                current_combination += 1
                print(f"Hyperparameter combination {current_combination}/{combinatios} finished.\n")
                results.append(result_obj)
                write_directly_json_file(result_runs_path, results) #Temporary viewing
    
    write_directly_json_file(result_runs_path, results)

def parse_experiments(experiment_list):
    """Convert dictionary input to proper PromptType Enum where needed."""
    for exp in experiment_list:
        if "prompt_type" in exp and isinstance(exp["prompt_type"], str):
            exp["prompt_type"] = PromptType(exp["prompt_type"])  # Convert to Enum
    return experiment_list

def main(train_data, val_data):
    """Run validation experiments."""
    for ex in experiments:

        selector_type= "similarity" if ex["semantic_selector"] else "coverage"
        prompt_type = ex["prompt_type"].value

        experiment_type = ex["name"].split("_")[1] # e.g: "RAG"
        if experiment_type == "similarity" or experiment_type == "Refinement":
            max_ctx = 16000
            #No RAG
            rag_data = None

        elif experiment_type == "RAG" or experiment_type == "assisted-RAG":
            max_ctx = 16000
            rag_data = init_rag_data() # None if not using RAG
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")



        experiments_dir = os.path.join("/fp/homes01/u01/ec-sindrre/slurm_jobs", f"{experiment_folder}/validation/{selector_type}/{prompt_type}/runs/")

        for shots in ex["num_shots"]:
            selector=init_example_selector(shots, train_data, semantic_selector=ex["semantic_selector"])
            experiment_name = f"{ex['name']}_{shots}_shot"

            for model_name in models:
                if fold != -1:
                    file_name = f"3_fold/{experiment_name}_{model_name}/fold_{fold}.json"
                else:
                    file_name = f"hold_out/{experiment_name}_{model_name}.json"
                result_runs_path = os.path.join(experiments_dir, file_name)
    

                print(f"\n==== Running {experiment_folder} validation for {experiment_name} on '{model_name}' ====")  
                model = get_model_code_tokens_from_file(model_name, f'{project_dir}/data/max_tokens.json')
                run_val_experiment(
                    client=client,
                    val_data=val_data,
                    dataset_nodes=dataset_nodes,
                    all_nodes=all_nodes,
                    experiment_name=experiment_name,
                    result_runs_path=result_runs_path,
                    model=model,
                    example_pool=selector,
                    prompt_type=ex["prompt_type"],
                    debug=False, 
                    ollama_port = ollama_port,
                    max_ctx=max_ctx,
                    rag_data=rag_data,
                    experiment_type=experiment_type,
                )
                print(f"Validation finished for experiment: {experiment_name}")
                print(f"See run results in: {result_runs_path}")

            print(f"Validation finished for {experiment_name} on {len(models)} models: {models}")
            print(f"See run results in: {experiments_dir}")
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\n⏱️ Total execution time: {hours}h {minutes}m {seconds}s")
            subprocess.run(["bash", f"{project_dir}/my_packages/common/push_runs.sh", 
                            experiment_folder, 
                            "validation", 
                            selector_type, 
                            prompt_type,
                            str(hours), str(minutes), str(seconds),
                            str(fold)
                            ], check=True)
            print("✅ push_runs.sh script executed successfully!")
            print("🚀 validation completed successfully!")

if __name__ == "__main__":
    experiment_folder = "SynCode"
    experiment_dir = os.path.abspath(f"{script_dir}/..")
    env_path = os.path.abspath(f"{project_dir}/../../.env")
    print("Env is located in:", env_path)
    load_dotenv(env_path)

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
    print(f"fold: {fold}")
    print("########################################")

    start_time = time.time()
    main_dataset_folder = f'{project_dir}/data/MBPP_Midio_50/'

    print("\n==== Splits data ====")
    train_data, val_data, test_data = get_hold_out_splits(main_dataset_folder)
    dataset = train_data + val_data + test_data

    if fold != -1:
        print(f"Using {fold}-fold cross-validation on merged train+test splits")
        k_fold_data = (train_data + test_data)  
        train, val = get_k_fold_splits(main_dataset_folder, fold)
    else:
        print("Using static train/validation split")
    
    all_responses = [sample["response"] for sample in dataset]
    print(f"Number of all responses: {len(all_responses)}")
    
    dataset_nodes = read_dataset_to_json(main_dataset_folder + "/metadata/used_external_functions.json")
    print(f"Number of nodes in datset: {len(dataset_nodes)}")

    all_nodes = read_dataset_to_json( f"{project_dir}/data/all_library_nodes.json") # All nodes
    print(f"Number all nodes: {len(all_nodes)}")

    print("\n==== Configures models ====")
    client, models = model_configs(all_responses, model_provider, models, ollama_port)

    print(f"Total experiments variations to run: {len(experiments) * len(models)* len(experiments[0]['num_shots'])}")
    
    print("\n==== Running validation ====")
    main(
        train_data=train_data, 
        val_data=val_data
    )
        