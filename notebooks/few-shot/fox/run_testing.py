import time
import subprocess
import os
import sys
root_dir = os.getcwd()
print(f"Root directory: {root_dir}")
results_dir = f"{root_dir}/notebooks/few-shot/fox/testing_runs"
sys.path.append(root_dir)
from my_packages.common import PromptType, get_prompt_type
from my_packages.evaluation.code_evaluation import run_model
# from my_packages.experiments.few_shot import get_dataset_splits, init_example_selector, model_configs
from dotenv import load_dotenv
from pathlib import Path
from my_packages.data_processing.attributes_processing import used_functions_to_string
from my_packages.prompting.few_shot import transform_code_data
from my_packages.utils.file_utils import write_json_file, read_dataset_to_json
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

def model_configs(all_responses, model_provider):  
    load_dotenv("../../.env")

    match model_provider:
        case 'ollama':
            host = 'http://localhost:11434'
            if is_remote_server_reachable(url = host + "/api/tags"):
                print("Server is reachable.")
            else:
                server_diagnostics()
                if not is_remote_server_reachable(url = host + "/api/tags"):
                    print("Ollama server is not reachable. Batch job might have finished. Try running bash script again.")

            client = ChatOllama
    
            models = [
                # 14b models:
                "phi4:14b-fp16", #16k context length
                # "qwen2.5:14b-instruct-fp16", #128 k

                #32b models:
                # "qwq:32b-preview-fp16", #ctx: 32,768 tokens
                # "qwen2.5-coder:32b-instruct-fp16", #32,768 tokens
    
                # #70b models:
                # "llama3.3:70b-instruct-fp16", #ctx: 130k
                # "qwen2.5:72b-instruct-fp16", #ctx: 139k
            ]
            models_not_tokenized = models_not_in_file(models, f'{root_dir}/notebooks/few-shot/code_max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{root_dir}/notebooks/few-shot/code_max_tokens.json')

        case 'openai':
            openai_token = os.getenv('OPENAI_API_KEY')
            if not openai_token:
                raise Exception("OpenAI API key not found in .env file")
            client = ChatOpenAI
            models = [
                "gpt-4o",
                # "o1-preview", 
            ]
            # few_shot_messages = create_few_shot_messages(explained_used_libraries, train_prompts, train_responses, "NODE_GENERATOR_TEMPLATE", "developer")
            models_not_tokenized = models_not_in_file(models, f'{root_dir}/notebooks/few-shot/code_max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{root_dir}/notebooks/few-shot/code_max_tokens.json')

        case 'anthropic':
            anthropic_token = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_token:
                raise Exception("Anthropic API key not found in .env file")
            client = ChatAnthropic
            # embed_client = AnthropicEmbeddings
            models = [
                "claude-3-5-sonnet-latest"
            ]
            models_not_tokenized = models_not_in_file(models, f'{root_dir}/notebooks/few-shot/code_max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{root_dir}/notebooks/few-shot/code_max_tokens.json')
        case _:
            raise Exception("Model provider not supported")
    return client, models

def init_example_selector(
        num_shots: int, 
        example_pool: dict, 
        semantic_selector: bool
    )-> BaseExampleSelector:

    example_pool.sort(key=lambda x: int(x['task_id']))
    print(f"Number of examples in the pool: {len(example_pool)}")

    if semantic_selector:
        selector = get_semantic_similarity_example_selector(
            example_pool, 
            OllamaEmbeddings(model="nomic-embed-text"),
            shots=num_shots,
            input_keys=["task"],
        )
    else:
        selector = get_coverage_example_selector(
            example_pool, 
            label = "external_functions",
            shots=num_shots,
            seed=9
        )
    return selector


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
        seeds = [3, 75, 346],
):
    results = []
    for seed in seeds:
        print(f"Running with seed: {seed}")
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
            prompt_type
        )
        result_obj = {
            "experiment_name": experiment_name,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seedFOX": seed,
            "n_generations_per_task": n,
            "model": model["name"],
            "largest_context": largest_context,
            "task_candidates": model_result,
        }
        results.append(result_obj)
        ## Write to file
        write_json_file(file_path, results) #Temporary viewing
    
    write_json_file(file_path, results)

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
    

if __name__ == "__main__":
    start_time = time.time()
    main_dataset_folder = f'{root_dir}/data/MBPP_Midio_50/'
    best_params_folder = f"{root_dir}/notebooks/few-shot/fox/best_params/"
    runs_folder = f"{root_dir}/notebooks/few-shot/fox/testing_runs"
    metrics = ["syntax", "semantic"]
    env = ""
    n_generations_per_task = 1

    print("\n==== Splits data ====")
    train_data, val_data, test_data = get_dataset_splits(main_dataset_folder)
    all_responses = [sample["response"] for sample in train_data] + [sample["response"] for sample in val_data] + [sample["response"] for sample in test_data]
    print(f"Number of all responses: {len(all_responses)}")
    used_functions_json = read_dataset_to_json(main_dataset_folder + "metadata/used_external_functions.json")
    available_nodes = used_functions_to_string(used_functions_json)
    print(f"Number of available nodes: {len(available_nodes)}")

    print("\n==== Configures models ====")
    client, models = model_configs(all_responses, 'ollama')

    print("\n==== Running testing ====")
    for file_name in os.listdir(best_params_folder):
        best_params_path = os.path.join(best_params_folder, file_name)
        print(f"Processing file: {best_params_path}")
        experiment_info = get_info_from_filename(file_name)
        experiment_name = experiment_info["experiment_name"]

        best_params = read_dataset_to_json(best_params_path)
        for params in best_params:
            model_name = params["model_name"]
            print(f"Processing experiment: '{experiment_name}' with model: '{model_name}'")
            file_path = f"{runs_folder}/{experiment_name}_{model_name}.json"
            model = get_model_code_tokens_from_file(model_name, f'{root_dir}/notebooks/few-shot/code_max_tokens.json')
            example_selector = init_example_selector(experiment_info["shots"], train_data, experiment_info["semantic_selector"])
            run_testing_experiment(
                client,
                test_data[:1],
                available_nodes,
                experiment_name,
                file_path,
                model,
                example_selector,
                experiment_info["prompt_type"],
                params["temperature"],
                params["top_p"],
                params["top_k"],
                n_generations_per_task,
            )
            print(f"Testing finished for experiment: {experiment_name}")
            print(f"See run results in: {results_dir}/{experiment_name}.json")
    print("Testing finished!")
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n⏱️ Total execution time: {hours}h {minutes}m {seconds}s")
    subprocess.run(["bash", f"{root_dir}/notebooks/few-shot/fox/scripts/push_runs.sh", "testing", str(hours), str(minutes), str(seconds)], check=True)
    print("✅ push_runs.sh script executed successfully!")



        


