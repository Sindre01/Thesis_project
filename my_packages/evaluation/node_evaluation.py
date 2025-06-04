import json
import numpy as np
import re
from my_packages.common.classes import RagData
from my_packages.evaluation.metrics import estimate_pass_at_k
from my_packages.prompting.prompt_building import create_few_shot_prompt, create_final_prompt
from my_packages.utils.server_utils import server_diagnostics
from my_packages.evaluation.models import invoke_anthropic_model, invoke_openai_model, invoke_o1_model, invoke_ollama_model
from colorama import Fore, Back, Style # type: ignore
from sklearn.metrics import f1_score
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langsmith.schemas import Example

##################### NOT used for the experiements, but was used for testing generation of nodes.##########
def extract_nodes(
        response_text: str
    ):
    """Extract nodes from the response using regex."""
    # Match content between ```language and ```
    match = re.search(fr"```midio(.*?)```", response_text, re.DOTALL)

    # Extract and clean up the nodes
    if match:
        return match.group(1).strip()  # Return only the midio block

    # If no match, assume the response might already be nodes without markdown formatting
    return response_text.strip()

def calculate_pass_at_k_scores(
        result_dict: dict[str, list[str]],
        ks: int
    ):
    """
    Pass@k evaluation for a given result dictionary.
    Pass@k asses the probability that out of k samples, at least one was correct.

    Parameters:
    - result_dict: A dictionary of true results and generated candidates.
    - ks: A list of k values to calculate pass@k for.

    Returns:
    - dict: A dictionary of pass@k scores for each k value.

    """
    # assert len(completion_id) == len(problems), "Some problems are not attempted."
    # test_results = run_tests(model_result, ks, debug) passed = [r[1]["passed"] for r in result] .#if all tests have passed, then passed = True
#     (0, {"passed": True, "result": "Some details about the execution"}),
#     (1, {"passed": False, "result": "Some details about the execution"}),
#     (2, {"passed": True, "result": "Some details about the execution"}),
# ]
    print("\n == Pass@k computation ==\n")

    # Calculate pass@k for is_subset for each problem and ks
    
    total, correct = [], []
    for true_result, k_results in result_dict.items():
        k_results.sort()
        true_result_set = set(true_result.split(","))
        passed = [
            True 
            if true_result_set.issubset(set(result.split(","))) 
            else 
            False 
            for result in k_results
        ]
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)
    print(f"Total: {total}, Correct: {correct}")
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() 
                 for k in ks if (total >= k).all()}
    print(f"Pass@K: {pass_at_k}")
    return pass_at_k

def calculate_f1_score(
        result_dict: dict[str, list[str]]
    ):
    """
    Calculate the average F1 score.

    Parameters:
    - result_dict: A dictionary of true results and generated candidates.
    Returns:
    - float: The average F1 score.
    """
    print("\n== F1 Score computation ==\n")
    f1_scores = []
    
    for true_response, k_results in result_dict.items():
        #chose only first generated and transform from string to set, by splitting on ','
        generated_nodes = set(k_results[0].replace(",", "").split())
        true_response = set(true_response.replace(",", "").split())

        # Combine all nodes to ensure alignment
        all_nodes = sorted(set(true_response).union(generated_nodes))
        y_true = [1 if node in true_response else 0 for node in all_nodes]
        y_pred = [1 if node in generated_nodes else 0 for node in all_nodes]

        # Calculate the F1 score
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
        if generated_nodes == true_response:
            print(f"{Fore.GREEN}{Style.BRIGHT}100% Correct response (exact match){Style.RESET_ALL}\n")
        else:
            print(f"{Fore.YELLOW}{Style.BRIGHT}Partial match: F1 Score={f1:.2f}{Style.RESET_ALL}\n")

    # Return the average F1 score
    print(f"F1 Scores: {f1_scores}, Mean: {np.array(f1_scores).mean()}")
    return np.array(f1_scores).mean()

def run_model(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model,
    dataset_nodes,
    all_nodes,
    data : list[Example],
    example_pool,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    ks=[1], 
    seed=None,                              
    debug=False
):
    """
    Returns: 
    {
        "node1, node2": ["node1, node2", "node1, node2", "node1, node2", "node1, node2"],  # 1 correct out of 3
        "node3, node4": ["node1, node2", "node1, node2", "node1, node2", "node1, node2"],    # 1 correct out of 3
        "node5, node6": ["node5, node6", "node1, node2", "node1, node2", "node1, node2"],   # 2 correct out of 3
    }
    
    """
    results = {}

    for index, sample in enumerate(data):
        generated_candidates: list[str] = []
        true_response = sample.outputs["response"]
        true_response_nodes = extract_nodes(true_response)
        task = sample.inputs["task"]

        # Do calulations before evaluation and just pass in an class that get the examples for the task.
        examples = example_pool.select_examples(sample.inputs)
        few_shot = create_few_shot_prompt(examples, 'NODES_TEMPLATE')
        final_prompt_template = create_final_prompt(few_shot, "NODE_GENERATOR_TEMPLATE", "NODES_TEMPLATE")
        prompt = final_prompt_template.format(task=task, external_functions=dataset_nodes)

        for attempt_i in range(max(ks)):
            max_retries = 3
            retries = 0
            new_seed = seed * attempt_i if seed else None # Use different seed for each attempt if not None
            generated = ""
            while retries < max_retries:
                try:
                    print(f"Generating response for sample {index}..", end="\r")
                    
                    if "claude" in model:
                        llm = client(
                            model=model,
                            temperature=temperature,
                            num_predict=max_new_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            stream=False,
                            num_ctx=10000,
                            stop=["```<|eot_id|>"],
                            seed=new_seed
                        )
                    else:
                        llm = client(
                            model=model,
                            temperature=temperature,
                            num_predict=max_new_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            stream=False,
                            num_ctx=10000,
                            stop=["```<|eot_id|>"],
                            seed=new_seed
                        )
                    
                    chain = (final_prompt_template | llm)

                    response = chain.invoke({
                        "task": task, 
                        "external_functions": dataset_nodes
                    },
                    {"run_name": "Node prediction"})

                    generated = response.content
                    break  # If generation succeeded, break out of retry loop
                except Exception as e:
                    retries += 1
                    print(f"Attempt {retries} failed with error: {e}")
                    server_diagnostics()

            else:
                print("Failed to get a response from the server after "
                      + str(retries) + " attempts.")
                generated = ""
                
            # Extract nodes from the generated response and transform to a set
            generated_nodes = extract_nodes(generated)
            generated_candidates.append(generated_nodes)
            
            generated_example: Example = sample.copy(update={"outputs": {"response": generated_nodes}})

        # --- Now we have up to k responses for this prompt. ---

        #Then add the generated candidates to the results dictionary
        results[true_response_nodes] = generated_candidates

        if debug:
            print(f"\n\n{Style.BRIGHT}=== Sample: {index} ===")
            print(f"{Fore.CYAN}{Style.BRIGHT} User prompt: {Style.RESET_ALL}\n{prompt}\n")
            for i, cand in enumerate(generated_candidates):
                print(f"{Fore.YELLOW}{Style.BRIGHT} Assistant response: #{i+1}:\n{cand}\n")
            print(f"{Fore.GREEN}{Style.BRIGHT} True response:{Style.RESET_ALL}\n {true_response_nodes}\n")

    return results

def evaluate_nodes(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model,
    dataset_nodes,
    all_nodes,
    data,
    example_pool,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    seed,
    ks,                               
    debug=False,
    rag_data: RagData = None
):
    model_result = run_model(
        client,
        model,
        dataset_nodes,
        all_nodes,
        data,
        example_pool,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        ks,
        seed,
        debug,
        rag_data=rag_data
    )

    #calculate pass@k and f1 score metrics
    pass_at_k_dict = calculate_pass_at_k_scores(model_result, ks)
    f1_score = calculate_f1_score(model_result)

    return f1_score, pass_at_k_dict


def run_validation(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model, 
    dataset_nodes,
    all_nodes, 
    val_data: list[Example],
    example_pool: list[Example],
    temperatures, 
    top_ps,
    top_ks, 
    ks=[1],
    seed=None, 
    debug=False
):
    # SEED = 42 # During Validation Phase for reproducibility
    validation_best_f1 = 0.0
    validation_pass_ks = {} #Not the best, but the ones that gave the best f1_score
    best_params = {"temperature": 0.5, "top_p": 0.5, "top_k": 50}

    print(f"{Fore.CYAN}{Style.BRIGHT}Validation Phase:{Style.RESET_ALL}")
    for temp in temperatures:
        for top_k in top_ks:
            for top_p in top_ps:
                # print(f"Validating with temperature: {temp}, top_k: {top_k} and top_p: {top_p}")
                val_f1, val_pass_k_dict = evaluate_nodes (
                    client,
                    model['name'],
                    dataset_nodes,
                    all_nodes,
                    val_data,
                    example_pool,
                    model["max_tokens"],
                    temp,
                    top_p,
                    top_k,
                    seed,
                    ks=ks,
                    debug=debug
                )
                print(f"Validation with temp={temp}, top_k={top_k} and top_p={top_p}. Gave f1={val_f1} and pass@ks={val_pass_k_dict}")

                #Choose best parameters based on f1_score
                if val_f1 > validation_best_f1:
                    validation_best_f1 = val_f1
                    validation_pass_ks = val_pass_k_dict
                    best_params = {"temperature": temp, "top_p": top_p, "top_k": top_k}

    result = {
        "seed": seed,
        "val_f1": validation_best_f1,
        "val_pass_ks": validation_pass_ks,
        "temperature": best_params["temperature"],
        "top_p": best_params["top_p"],
        "top_k": best_params["top_k"],
    }
    if debug:
        print_validation_result(result)

    return result
    
def run_testing(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model, 
    dataset_nodes,
    all_nodes, 
    test_data: list[Example],
    example_pool: list[Example],
    temperature, 
    top_p, 
    top_k,
    ks, 
    seeds, 
    debug=False,
):
    results = []
    
    #Test the model on the test set with different seeds and the best hyperparameters.
    print(f"{Fore.CYAN}{Style.BRIGHT}Testing Phase:{Style.RESET_ALL}")

    for seed in seeds:
        print(f"\nTesting with Seed: {seed} ..", end="\r")

        test_f1, test_pass_k_dict = evaluate_nodes(
            client,
            model['name'],
            dataset_nodes,
            all_nodes,
            test_data,
            example_pool,
            model["max_tokens"],
            temperature,
            top_p,
            top_k,
            seed,
            ks,
            debug
        )

        new_run = {
            "seed": seed,
            "test_f1": test_f1,
            "test_pass_ks": test_pass_k_dict,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        results.append(new_run)
        if debug:
            print_test_result(new_run)

    return results

def calculate_deviation(test_runs):
    """Calculate the standard deviation of the metrics"""
    f1_std= np.std([run["test_f1"] for run in test_runs])
    f1_std_mean= np.array([run["test_f1"] for run in test_runs]).mean()
    pass_ks_std = {}
    pass_at_ks_mean = {}
    pass_at_ks = list(test_runs[0]["test_pass_ks"].keys())
    for pass_at_k in pass_at_ks:
        pass_ks_std[pass_at_k] = np.std([run["test_pass_ks"][pass_at_k] for run in test_runs])
        pass_at_ks_mean[pass_at_k] = np.array([run["test_pass_ks"][pass_at_k] for run in test_runs]).mean()

    return f1_std_mean, f1_std, pass_at_ks_mean, pass_ks_std

def print_validation_result(run: dict):
    """Print the results of a run."""
    best_temperature = run["temperature"]
    best_top_p = run["top_p"]
    best_top_k = run["top_k"]
    val_f1 = run["val_f1"]
    val_pass_ks = run["val_pass_ks"]

    print(
        f"  = Validation =\n"
        f"  > Best temperature: {best_temperature:.2f}\n"
        f"  > Best top_k: {best_top_k:.2f}\n"
        f"  > Best top_p: {best_top_p:.2f}\n"
        f"  > F1: {val_f1:.2f}\n"
        f"  > Pass@ks {json.dumps(val_pass_ks, indent=4)}\n"
    )
        
def print_test_result(run: dict):
    """Print the result of a run."""
    seed = run["seed"]
    temperature = run["temperature"]
    top_p = run["top_p"]
    top_k = run["top_k"]
    test_f1 = run["test_f1"]
    test_pass_ks = run["test_pass_ks"]

    print(
        f"===  Seed {seed} ===\n"
        f"  > Temperature: {temperature:.2f}\n"
        f"  > Top_k: {top_k:.2f}\n"
        f"  > Top_p: {top_p:.2f}\n"
        f"  = Test =\n"
        f"{Fore.GREEN}{Style.BRIGHT}"
        f"  > F1: {test_f1:.2f}\n"
        f"  > Pass@ks: {json.dumps(test_pass_ks, indent=4)}"
        f"{Style.RESET_ALL}"
    )
        
