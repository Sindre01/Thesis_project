# %% [markdown]
# ## Install dependencies

# %%


# %pip install ollama
# %pip install transformers
# %pip install tiktoken
# %pip install huggingface-hub
# %pip install python-dotenv


# %% [markdown]
# ## Prepare data

# %%
import json
import sys
sys.path.append('../')  # Add the path to the my_packages module
from my_packages.data_processing.split_dataset import split_on_shots, split
from my_packages.data_processing.get_labels_data import used_libraries_from_dataset
from my_packages.analysis.analyze_datasets import analyze_library_distribution, analyze_instance_distribution, analyze_visual_node_types_distribution

main_dataset_folder = '../data/mbpp_transformed_code_examples/sanitized-MBPP-midio.json'

# main dataset
with open(main_dataset_folder, 'r') as file:
    dataset = json.load(file)
    
num_shot = 10 # Few-shot examples
eval_size_percentage = 0.5
train_data, val_data, test_data = split_on_shots(num_shot, eval_size_percentage, dataset, seed = 64, write_to_file=True)

def extract_prompts_and_responses(data):
    prompts = [f"{item['prompts'][0]}\n " for item in data]
    # Responses from the library_functions list
    responses = []
    for sample in data:
        file_path = f"../data/mbpp_transformed_code_examples/only_files/task_id_{sample['task_id']}.midio"
        try:
            with open(file_path, 'r') as file:
                solution_code = file.read().strip()
                responses.append(solution_code)
        except FileNotFoundError:
            responses.append("File not found")
        except Exception as e:
            responses.append(f"Error: {e}")
    
    return prompts, responses

def used_libraries_to_string(data):
    name_doc_string = ""
    for func in data:
        name_doc_string += f"Name: {func['function_name']}\nDocumentation: {func['doc']}\n\n"
    return name_doc_string
    
# Extract training, validation, and test data
train_prompts, train_responses = extract_prompts_and_responses(train_data)  # Use as examples for few-shot learning
val_prompts, val_responses = extract_prompts_and_responses(val_data)  # Validation set
test_prompts, test_responses = extract_prompts_and_responses(test_data)  # Test set

# Extract all unique nodes (library_functions) across datasets
used_libraries_json = used_libraries_from_dataset(train_data)

explained_used_libraries = used_libraries_to_string(used_libraries_json)

print(f"train set samples: {len(train_prompts)}")
print(f"Validation set samples: {len(val_prompts)}")
print(f"test set samples: {len(test_prompts)}")

#Bar chart of distribuation
analyze_library_distribution(train_data, val_data, test_data)
analyze_instance_distribution(train_data, val_data, test_data)
analyze_visual_node_types_distribution(train_data, val_data, test_data)

# %% [markdown]
# ## Define models to test AND Calculate tokens for different tokenizers

# %%
from my_packages.utils.tokens import find_max_tokens
from transformers import AutoTokenizer
import tiktoken
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Log in to Hugging Face
load_dotenv("../.env")

access_token_read = os.environ.get('HF_API_KEY')

if access_token_read:
    login(token=access_token_read)
    print("Logged in to Hugging Face successfully!")
else:
    print("HF_API_KEY is not set in your environment variables.")

open_models = [
    {"name": "llama3.1", "tokenization": "meta-llama/Meta-Llama-3.1-8B"},
    # {"hg_name": "mistralai/Mistral-Small-Instruct-2409", "ollama_name": ""},
    # {"hg_name": "meta-llama/Llama-3.3-70B-Instruct", "ollama_name": ""},
    # {"hg_name": "meta-llama/CodeLlama-70b-Instruct-hf", "ollama_name": ""},
    # {"hg_name": "meta-llama/Llama-3.2-90B-Vision-Instruct", "ollama_name": ""}
]
gpt_models = [
    {"name": "o1-preview", "tokenization": "o200k_base"},
    {"name": "gpt-4o", "tokenization": "o200k_base"},
]
DATA_DIR = '../data/mbpp_transformed_code_examples/only_files'
OUTPUT_JSON = 'token_counts.json'

open_models_to_test = []
for model_info in open_models:
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenization"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_tokens = find_max_tokens(DATA_DIR, tokenizer)
    model_result = {
        "name": model_info["name"],
        "max_tokens": max_tokens,
    }
    open_models_to_test.append(model_result)

gpt_models_to_test = []
for model_info in gpt_models:
    encoding = tiktoken.encoding_for_model(model_info["name"])
    max_tokens = find_max_tokens(DATA_DIR, encoding)
    model_result = {
        "name": model_info["name"],
        "max_tokens": max_tokens
    }
    gpt_models_to_test.append(model_result)

print(open_models_to_test)
print(gpt_models_to_test)

# %% [markdown]
# ## Create Prompt

# %%
# Function to create few-shot prompt
def create_few_shot_prompt(train_prompts, train_responses, input_prompt):
    guiding = "You are going to solve some programming tasks for node-based programming language. Use minimal amount of library functions to solve the tasks.\n" 
    node_list = f"Only use the following library functions:\n {explained_used_libraries}\n\n"
    context = guiding + node_list
    few_shots = ""
    for i, (prompt, response) in enumerate(zip(train_prompts, train_responses)):
        few_shots += f"Example {i+1}:\nPrompt: {prompt}\nResponse: {response}\n\n"
    few_shots += f"Task:\n{input_prompt}\nResponse:"
    return (context, few_shots)

## Matbe use ModelFile in ollama https://github.com/ollama/ollama/blob/main/docs/modelfile.md 
# modelfile='''
# FROM llama3.2
# SYSTEM You are mario from super mario bros.
# '''

# ollama.create(model='example', modelfile=modelfile)

# %% [markdown]
# ## Test some Models, with different seeds, temperatures, top_ps

# %%
from my_packages.evaluation.compiler import is_code_compilable
from my_packages.utils.run_bash_script import run_bash_script_with_ssh
from my_packages.utils.ollama_utils import is_remote_server_reachable
import ollama # https://github.com/ollama/ollama-python
import time
from httpx import RemoteProtocolError

host = 'http://localhost:11434'

if is_remote_server_reachable(host + "/api/tags"):
    print("Server is reachable.")
else:
    print("Ollama server is not reachable. Batch job might have finished. Try running bash script again.")

client = ollama.Client(
  host=host,
)

# Function to generate and evaluate responses using Ollama
def evaluate(model, prompts, responses, seed, max_new_tokens=50, temperature=0.7, top_p=0.9):
    correct = 0
    total = len(prompts)
    for index, (prompt, true_response) in enumerate(zip(prompts, responses)):
        (context, few_shot) = create_few_shot_prompt(train_prompts, train_responses, prompt)

        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                print("generating response..")
                generated = client.generate(
                    model=model,
                    # system = context,
                    prompt=(context + few_shot),
                    options={       
                        "seed": seed,
                        "num_predict": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                ).response
                break
            except Exception as e:
                retries += 1
                print(f"Attempt {retries} failed with error: {e}")
                if is_remote_server_reachable(host + "/api/tags"):
                    print("Server is reachable.")
                else:
                    print(f"Trying to run ssh forwarding bash script " + "" if retries == 1 else "again" + " , to connect to fox.")
                    # run_bash_script_with_ssh("../SSH_FORWARDING.sh")
                    os.system("bash ../SSH_FORWARDING.sh")
                    # %bash ../SSH_FORWARDING.sh
                time.sleep(2)  # wait for 2 seconds before retrying
        else:
            print("Failed to get a response from the server after " + retries + " attempts.")
            generated = ""
        # print(f"\n\context: {context}")
        # print(f"\n\Few_shot: {few_shot}")
        # print(f"\n\nSample: {index}")
        # print(f"Prompt: {prompt}")
        # print(f"Generated response:\n {generated}")
        # print(f"True response:\n {true_response}")

        if is_code_compilable(generated):
            print("Correct response: Code compiled successfully.")
            correct += 1
        else:
            print("Invalid response: Midio code compilation failed.")

    return correct / total

results = {}
seeds = [3, 75, 346]

for model in open_models_to_test:
    print(f"Testing model: {model['name']}")
    # ollama.pull(model['name'])  # Pull the model from the server

    print("max_tokens in dataset with current pipeline:", model["max_tokens"])

    for seed in seeds:
        print(f"\nTesting with Seed: {seed}")
        temperatures = [0.5, 0.7, 0.9]
        top_ps = [0.2, 0.5, 1.0]
        best_accuracy = 0
        best_params = {"temperature": 0.7, "top_p": 0.9}

        for temp in temperatures:
            for top_p in top_ps:
                accuracy = evaluate (
                    model['name'],
                    val_prompts,
                    val_responses,
                    seed,
                    model["max_tokens"],
                    temp,
                    top_p
                )
                print(f"Tested with temp={temp} and top_p={top_p}. Gave accuracy={accuracy}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {"temperature": temp, "top_p": top_p}

        print(f"Best Hyperparameters for {model['name']}: {best_params}, Validation Accuracy: {best_accuracy:.2f}")

        test_accuracy = evaluate(
            model['name'],
            test_prompts,
            test_responses,
            seed,
            model["max_tokens"],
            temperature=best_params["temperature"],
            top_p=best_params["top_p"]
        )

        print(f"Test Accuracy for {model['name']}: {test_accuracy:.2f}")

        if model["name"] not in results:
            results[model["name"]] = []
        results[model["name"]].append({
            "seed": seed,
            "validation_accuracy": best_accuracy,
            "test_accuracy": test_accuracy
        })

print("\nFinal Results:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for run in metrics:print(f"  Seed {run['seed']}: Validation Accuracy: {run['validation_accuracy']:.2f}, Test Accuracy: {run['test_accuracy']:.2f}")


