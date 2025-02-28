import time
import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../../..")
# experiment_dir = os.path.abspath(f"{script_dir}/..")


print("Script is located in:", script_dir)
print("Project is located in:", project_dir)
# print("Experiments are located in:", experiment_dir)

results_dir = f"{project_dir}/notebooks/few-shot/fox/validation_runs"

sys.path.append(project_dir)
from my_packages.common import PromptType
from my_packages.evaluation.code_evaluation import run_model
from dotenv import load_dotenv
from my_packages.data_processing.attributes_processing import used_functions_to_string
from my_packages.prompting.few_shot import transform_code_data
from my_packages.utils.file_utils import write_directly_json_file, read_dataset_to_json
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

                # #32b models:
                # "qwq:32b-preview-fp16", #ctx: 32,768 tokens
                # "qwen2.5-coder:32b-instruct-fp16", #32,768 tokens
    
                # #70b models:
                # "llama3.3:70b-instruct-fp16", #ctx: 130k
                # "qwen2.5:72b-instruct-fp16", #ctx: 139k
            ]
            models_not_tokenized = models_not_in_file(models, f'{project_dir}/notebooks/few-shot/code_max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{project_dir}/notebooks/few-shot/code_max_tokens.json')

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
            models_not_tokenized = models_not_in_file(models, f'{project_dir}/notebooks/few-shot/code_max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{project_dir}/notebooks/few-shot/code_max_tokens.json')

        case 'anthropic':
            anthropic_token = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_token:
                raise Exception("Anthropic API key not found in .env file")
            client = ChatAnthropic
            # embed_client = AnthropicEmbeddings
            models = [
                "claude-3-5-sonnet-latest"
            ]
            models_not_tokenized = models_not_in_file(models, f'{project_dir}/notebooks/few-shot/code_max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{project_dir}/notebooks/few-shot/code_max_tokens.json')
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


def run_val_experiment(
        client,
        val_data,
        available_nodes,
        experiment_name,
        file_path,
        model,
        example_pool,
        prompt_type: PromptType,
        temperatures = [0.9],
        top_ps = [0.2],
        top_ks = [10],
        n = 1, # Max value of array is generations per task
        seed = 9,
        debug = False,

):
    combinatios = len(temperatures) * len(top_ps) * len(top_ks)
    current_combination = 0
    results = []
    for temp in temperatures:
        for top_k in top_ks:
            for top_p in top_ps:
                print(f"Validating with temperature: {temp}, top_k: {top_k} and top_p: {top_p}")
                
                model_result, largest_context = run_model(
                    client,
                    model["name"],
                    available_nodes,
                    val_data,
                    example_pool,
                    model["max_tokens"],
                    temp,
                    top_p,
                    top_k,
                    n,
                    seed,
                    debug, 
                    prompt_type
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
                write_directly_json_file(file_path, results) #Temporary viewing
    
    write_directly_json_file(file_path, results)


if __name__ == "__main__":
    start_time = time.time()
    main_dataset_folder = f'{project_dir}/data/MBPP_Midio_50/'

    print("\n==== Splits data ====")
    train_data, val_data, test_data = get_dataset_splits(main_dataset_folder)
    all_responses = [sample["response"] for sample in train_data] + [sample["response"] for sample in val_data] + [sample["response"] for sample in test_data]
    print(f"Number of all responses: {len(all_responses)}")
    used_functions_json = read_dataset_to_json(main_dataset_folder + "metadata/used_external_functions.json")
    print(f"Number of external functions used: {len(used_functions_json)}")
    available_nodes = used_functions_to_string(used_functions_json)

    print("\n==== Configures models ====")
    client, models = model_configs(all_responses, 'ollama')

    print("\n==== Running validation ====")
    dataset = read_dataset_to_json(main_dataset_folder + "MBPP-Midio-50.json")
    experiment_type = "coverage" # "similarity"
    experiments = [
        {
            "name": "regular_coverage",
            "prompt_prefix": "Create a function",
            "num_shots": [1],
            "prompt_type": PromptType.REGULAR,
            "semantic_selector": False,
        },

        # {
        #     "name": "signature_coverage",
        #     "prompt_prefix": "Create a function",
        #     "num_shots": [1, 5, 10],
        #     "prompt_type": PromptType.SIGNATURE,
        #     "semantic_selector": False,
        # },
        # {
        #     "name": "regular_similarity",
        #     "prompt_prefix": "Create a function",
        #     "num_shots": [1, 5, 10],
        #     "prompt_type": PromptType.REGULAR,
        #     "semantic_selector": True,
        # },
        # {
        #     "name": "signature_similarity",
        #     "prompt_prefix": "Create a function",
        #     "num_shots": [1, 5, 10],
        #     "prompt_type": PromptType.SIGNATURE,
        #     "semantic_selector": False,
        # },
        # {
        #     "name": "regular_similarity",
        #     "prompt_prefix": "Create a function",
        #     "num_shots": [1, 5, 10],
        #     "prompt_type": PromptType.REGULAR,
        #     "semantic_selector": True,
        # },
        # {
        #     "name": "signature_similarity",
        #     "prompt_prefix": "Create a function",
        #     "num_shots": [1, 5, 10],
        #     "prompt_type": PromptType.SIGNATURE,
        #     "semantic_selector": True,
        # },
        # {
        #     "name": "cot_similarity",
        #     "prompt_prefix": "Create a function",
        #     "num_shots": [1, 5, 10],
        #     "prompt_type": PromptType.COT,
        #     "semantic_selector": True,
        # },
        # {
        #     "name": "cot_coverage",
        #     "prompt_prefix": "Create a function",
        #     "num_shots": [1, 5, 10],
        #     "prompt_type": PromptType.COT,
        #     "semantic_selector": False,
        # },
       
    ]

    print(f"Total experiments variations to run: {len(experiments) * len(models)* len([1, 5, 10])}")
    
    print("\n==== Running validation ====")
    for ex in experiments:

        selector_type= "similarity" if ex["semantic_selector"] else "coverage"
        prompt_type = ex["prompt_type"].value
        experiments_dir = os.path.join("/fp/homes01/u01/ec-sindrre/slurm_jobs", f"few-shot/validation/{selector_type}/{prompt_type}/runs/")

        for shots in ex["num_shots"]:
            selector=init_example_selector(shots, train_data, semantic_selector=ex["semantic_selector"])
            experiment_name = f"{ex['name']}_{shots}_shot"

            for model_name in models:
                file_name = f"{experiment_name}_{model_name}.json"
                result_runs_path = os.path.join(experiments_dir, file_name)
    

                print(f"\n==== Running few-shot validation for {experiment_name} on '{model_name}' ====")  
                model = get_model_code_tokens_from_file(model_name, f'{project_dir}/notebooks/few-shot/code_max_tokens.json')
                run_val_experiment(
                    client,
                    val_data[:1],
                    available_nodes,
                    experiment_name,
                    result_runs_path,
                    model,
                    selector,
                    ex["prompt_type"],
                )
                print(f"Validation finished for experiment: {experiment_name}")
                print(f"See run results in: {result_runs_path}")

            print(f"Validation finished for {experiment_name} on {len(models)} models: {models}")
            print(f"See run results in: {experiments_dir}")
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\n⏱️ Total execution time: {hours}h {minutes}m {seconds}s")
            subprocess.run(["bash", f"{project_dir}/notebooks/few-shot/fox/scripts/push_runs.sh", 
                            "few-shot", 
                            "validation", 
                            selector_type, 
                            prompt_type,
                            str(hours), str(minutes), str(seconds)], check=True)
            print("✅ push_runs.sh script executed successfully!")