from collections import defaultdict
import itertools
import json

import numpy as np
from my_packages.utils.server_utils import server_diagnostics
from my_packages.evaluation.models import invoke_anthropic_model, invoke_openai_model, invoke_o1_model, invoke_ollama_model
import re
from colorama import Fore, Back, Style # type: ignore
from sklearn.metrics import f1_score

def extract_nodes(response_text):
    """Extract nodes from the response using regex."""
    # Match content between ```language and ```
    match = re.search(fr"```midio(.*?)```", response_text, re.DOTALL)

    # Extract and clean up the nodes
    if match:
        return match.group(1).strip()  # Return only the midio block

    # If no match, assume the response might already be nodes without markdown formatting
    return response_text.strip()

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    result = np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
    print(result)
    return result

def calculate_pass_at_k_scores(result_dict, ks):
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

    # Calculate pass@k.
    total, correct = [], []
    for true_result, k_results in result_dict.items():
        k_results.sort()
        passed = [True if result == true_result else False for result in k_results]
        total.append(len(passed))
        correct.append(sum(passed))

     
    total = np.array(total)
    correct = np.array(correct)
    print("== Pass@k computation ==\n")
    print(f"Total: {total}, Correct: {correct}")
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() 
                 for k in ks if (total >= k).all()}
    print(f"Pass@K: {pass_at_k}")
    return pass_at_k

def calculate_f1_score(result_dict):
    """
    Calculate the average F1 score.

    Parameters:
    - result_dict: A dictionary of true results and generated candidates.
    Returns:
    - float: The average F1 score.
    """
    print("== F1 Score computation ==\n")
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
    client,
    messages,
    model,
    available_nodes,
    prompts,
    responses,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    ks=[1], 
    seed=None,                              
    debug=False
):
    results = {}

    for index, (prompt, true_response) in enumerate(zip(prompts, responses)):
        generated_candidates = []

        for attempt_i in range(max(ks)):
            max_retries = 3
            retries = 0
            new_seed = seed * attempt_i if seed else None # Use different seed for each attempt if not None
            generated = ""
            while retries < max_retries:
                try:
                    full_prompt = messages + [{"role": "user", "content": prompt}]
                    
                    print(f"Generating response for sample {index}..", end="\r")

                    if "claude" in model:
                        generated = invoke_anthropic_model(
                            client, full_prompt, model, max_new_tokens, temperature, top_p
                        )
                    elif "o1" in model:
                        generated = invoke_o1_model(
                            client, full_prompt, model, max_new_tokens
                        )
                    elif "gpt" in model:
                        generated = invoke_openai_model(
                            client, full_prompt, model, max_new_tokens, temperature, top_p, seed=new_seed
                        )
                    else:
                        generated = invoke_ollama_model(
                            client, full_prompt, model, max_new_tokens, temperature, top_p, seed=new_seed
                        )
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

        # --- Now we have up to k responses for this prompt. ---

        # Extract the "true" nodes from the reference.
        true_response_nodes = extract_nodes(true_response)

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
    client,
    messages,
    model,
    available_nodes,
    prompts,
    responses,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    seed=None,
    ks=[1],                               
    debug=False
):
    model_result: defaultdict= run_model(
        client,
        messages,
        model,
        available_nodes,
        prompts,
        responses,
        max_new_tokens,
        temperature,
        top_p,
        ks,
        seed,
        debug
    )
    # model_result = {
    # "node1, node2": ["node1, node2", "node1, node2", "node1, node2", "node1, node2"],  # 1 correct out of 3
    # "node3, node4": ["node1, node2", "node1, node2", "node1, node2", "node1, node2"],    # 1 correct out of 3
    # "node5, node6": ["node5, node6", "node1, node2", "node1, node2", "node1, node2"],   # 2 correct out of 3
    # }
    # ks = [1, 3]

    print(f"model_result: {model_result}")

    #calculate pass@k and f1 score metrics
    pass_at_k_dict = calculate_pass_at_k_scores(model_result, ks)
    f1_score = calculate_f1_score(model_result)

    return f1_score, pass_at_k_dict

def run_validation(
    client,
    messages, 
    model, 
    available_nodes, 
    val_prompts, 
    val_responses, 
    temperatures=[0.5, 0.7, 0.9], 
    top_ps=[0.2, 0.5, 1.0], 
    ks=[1],
    seed=None, 
    debug=False
):
    # SEED = 42 # During Validation Phase for reproducibility
    validation_best_f1 = 0.0
    validation_pass_ks = {} #Not the best, but the ones that gave the best f1_score
    best_params = {"temperature": 0.5, "top_p": 0.5}

    print(f"{Fore.CYAN}{Style.BRIGHT}Validation Phase:{Style.RESET_ALL}")
    for temp in temperatures:
        for top_p in top_ps:
            print(f"Validating with temperature: {temp}, and top_p: {top_p}")
            val_f1, val_pass_k_dict = evaluate_nodes (
                client,
                messages,
                model['name'],
                available_nodes,
                val_prompts,
                val_responses,
                model["max_tokens"],
                temp,
                top_p,
                seed,
                ks=ks,
                debug=debug
            )
            print(f"Validated with temp={temp} and top_p={top_p}. Gave f1={val_f1} and pass@ks={val_pass_k_dict}")

            #Choose best parameters based on f1_score
            if  val_f1 > validation_best_f1:
                validation_best_f1 = val_f1
                validation_pass_ks = val_pass_k_dict
                best_params = {"temperature": temp, "top_p": top_p}

    result = {
        "seed": seed,
        "val_f1": validation_best_f1,
        "val_pass_ks": validation_pass_ks,
        "temperature": best_params["temperature"],
        "top_p": best_params["top_p"],
    }
    if debug:
        print_validation_result(result)

    return result
    
def run_testing(
    client,
    messages, 
    model, 
    available_nodes, 
    test_prompts, 
    test_responses, 
    temperature, 
    top_p, 
    ks=[1], 
    seeds=[3, 75, 346], 
    debug=False
):
    results = []
    
    #Test the model on the test set with different seeds and the best hyperparameters.
    print(f"{Fore.CYAN}{Style.BRIGHT}Testing Phase:{Style.RESET_ALL}")
    for seed in seeds:
        print(f"\nTesting with Seed: {seed} ..", end="\r")

        test_f1, test_pass_k_dict = evaluate_nodes(
            client,
            messages,
            model['name'],
            available_nodes,
            test_prompts,
            test_responses,
            model["max_tokens"],
            temperature,
            top_p,
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
        }
        results.append(new_run)
        if debug:
            print_test_result(new_run)

    return results

def print_validation_result(run: dict):
    """Print the results of a run."""
    best_temperature = run["temperature"]
    best_top_p = run["top_p"]
    val_f1 = run["val_f1"]
    val_pass_ks = run["val_pass_ks"]

    print(
        f"  = Validation =\n"
        f"  > Best temperature: {best_temperature:.2f}\n"
        f"  > Best top_p: {best_top_p:.2f}\n"
        f"  > F1 {val_f1:.2f}\n"
        f"  > Pass@ks {json.dumps(val_pass_ks, indent=4)}\n"
    )
        
def print_test_result(run: dict):
    """Print the result of a run."""
    seed = run["seed"]
    temperature = run["temperature"]
    top_p = run["top_p"]
    test_f1 = run["test_f1"]
    test_pass_ks = run["test_pass_ks"]

    print(
        f"===  Seed {seed} ===\n"
        # f"  > Temperature: {temperature:.2f}\n"
        # f"  > Top_p: {top_p:.2f}\n"
        f"  = Test =\n"
        f"{Fore.GREEN}{Style.BRIGHT}"
        f"  > F1: {test_f1:.2f}\n"
        f"  > Pass@ks: {json.dumps(test_pass_ks, indent=4)}"
        f"{Style.RESET_ALL}"
    )
        
