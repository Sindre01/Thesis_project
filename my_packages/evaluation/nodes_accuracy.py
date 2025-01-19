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


def calculate_f1_score(generated_nodes_list, true_response_list):
    """
    Calculate the overall F1 score-based accuracy for generated nodes compared to the true response.

    Parameters:
    - generated_nodes_list: List of sets of generated nodes.
    - true_response_list: List of sets of true response nodes.

    Returns:
    - float: The overall accuracy as a percentage (from 0 to 1).

    Examples:
        generated_nodes_list = [
            {"node1", "node2"},
            {"node1"},
            {"node1", "node2", "node3"}
        ]
        true_response_list = [
            {"node1", "node2"},
            {"node1", "node2"},
            {"node1", "node2"}
        ]
    """
    total_f1_score = 0
    total = len(true_response_list)

    for generated_nodes_set, true_response_set in zip(generated_nodes_list, true_response_list):
        # Combine all nodes to ensure alignment
        all_nodes = sorted(set(generated_nodes_set).union(true_response_set))
        y_true = [1 if node in true_response_set else 0 for node in all_nodes]
        y_pred = [1 if node in generated_nodes_set else 0 for node in all_nodes]

        # Calculate the F1 score
        f1 = f1_score(y_true, y_pred)

        if generated_nodes_set == true_response_set:
            print(f"{Fore.GREEN}{Style.BRIGHT}100% Correct response (exact match){Style.RESET_ALL}\n")
        else:
            print(f"{Fore.YELLOW}{Style.BRIGHT}Partial match: F1 Score={f1:.2f}{Style.RESET_ALL}\n")

        # Add F1 score to the total
        total_f1_score += f1

    # Return the average F1 score
    return total_f1_score / total


def evaluate_nodes(client, messages, model, available_nodes, prompts, responses, seed, max_new_tokens=50, temperature=0.7, top_p=0.9, debug=False):
    total = len(prompts)
    correct = 0.00
    for index, (prompt, true_response) in enumerate(zip(prompts, responses)):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                full_prompt = messages + [{"role": "user", "content":prompt}]
                print(f"Generating response for sample {index}..", end="\r")
                if "claude" in model:
                    generated = invoke_anthropic_model(client, full_prompt, model, max_new_tokens, temperature, top_p)
                elif "gpt" in model:
                    generated = invoke_openai_model(client, full_prompt, model, seed, max_new_tokens, temperature, top_p)
                elif "o1" in model:
                    generated = invoke_o1_model(client, full_prompt, model, max_new_tokens)
                else:
                    generated = invoke_ollama_model(client, full_prompt, model, seed, max_new_tokens, temperature, top_p)
                break
            except Exception as e:
                retries += 1
                print(f"Attempt {retries} failed with error: {e}")
                server_diagnostics()

        else:
            print("Failed to get a response from the server after " + str(retries) + " attempts.")
            generated = ""
            return -1
       
        # Extract nodes from the generated response and transform to a set
        generated_nodes = extract_nodes(generated)
        generated_nodes_set = set(generated_nodes.replace(",", "").split())

        # Extract nodes from the true response and transform to a set
        true_response_nodes = extract_nodes(true_response)
        true_response_set = set(true_response_nodes.replace(",", "").split())

        if debug:
            print(f"\n\n{Style.BRIGHT}Sample: {index}")
            print(f"{Fore.CYAN}{Style.BRIGHT} User prompt: {Style.RESET_ALL}\n{prompt}\n")
            print(f"{Fore.YELLOW}{Style.BRIGHT} Assistant response:{Style.RESET_ALL}\n{generated}\n")
            print(f" >{Fore.BLUE}{Style.BRIGHT} Extracted nodes:{Style.RESET_ALL}\n {generated_nodes}\n")
            print(f"{Fore.GREEN}{Style.BRIGHT} True response:{Style.RESET_ALL}\n {true_response}\n")
            print(f" >{Fore.BLUE}{Style.BRIGHT} Extracted nodes:{Style.RESET_ALL}\n {true_response_nodes}\n")

        #Remove invalid nodes from the generated nodes
        # valid_generated_nodes = generated_nodes_set.intersection(available_nodes)
        score = calculate_f1_score([generated_nodes_set], [true_response_set])
        correct += score
        # print(f"{Fore.GREEN}{Style.BRIGHT} Accuracy score: {score}{Style.RESET_ALL}\n")
        # if true_response_set == valid_generated_nodes:
        #     correct += 1
        # if true_response_set.issubset(valid_generated_nodes):
        #     print(f"{Fore.GREEN}{Style.BRIGHT} issubset Correct response{Style.RESET_ALL}\n")
        #     correct += 1
        # else:
        #     print(f"{Fore.RED}{Style.BRIGHT} Wrong response{Style.RESET_ALL}\n")
    return correct / total

def run_nodes_evaluation(client, messages, model, available_nodes, val_prompts, val_responses, test_prompts, test_responses, temperatures=[0.5, 0.7, 0.9], top_ps=[0.2, 0.5, 1.0], seeds=[3, 75, 346], debug=False):
    results = []

    SEED = 42 # During Validation Phase for reproducibility
    best_accuracy = 0.0
    best_params = {"temperature": 0.5, "top_p": 0.5}

    print(f"{Fore.CYAN}{Style.BRIGHT}Validation Phase:{Style.RESET_ALL}")
    for temp in temperatures:
        for top_p in top_ps:
            print(f"Validating with temperature: {temp}, and top_p: {top_p}")
            accuracy = evaluate_nodes (
                client,
                messages,
                model['name'],
                available_nodes,
                val_prompts,
                val_responses,
                SEED,
                model["max_tokens"],
                temp,
                top_p,
                debug
            )
            print(f"Tested with temp={temp} and top_p={top_p}. Gave accuracy={accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"temperature": temp, "top_p": top_p}
            
    print(f"Best Hyperparameters for {model['name']}: {best_params}, Validation Accuracy: {best_accuracy:.2f}")

    #Test the model on the test set with different seeds and the best hyperparameters.
    print(f"TESTING Phase")
    for seed in seeds:
        print(f"\nTesting with Seed: {seed}", end="\r")
        best_temperature = best_params["temperature"]
        best_top_p = best_params["top_p"]
        test_accuracy = evaluate_nodes(
            client,
            messages,
            model['name'],
            available_nodes,
            test_prompts,
            test_responses,
            seed,
            model["max_tokens"],
            best_temperature,
            best_top_p,
            debug
        )

        print(f"Test Accuracy for {model['name']}: {test_accuracy:.2f}")
        new_run = {
            "seed": seed,
            "validation_accuracy": best_accuracy,
            "test_accuracy": test_accuracy,
            "temperature": best_temperature,
            "top_p": best_top_p,
        }
        results.append(new_run)
        if debug:
            print_accuracy_result(new_run)
    return results

def print_accuracy_result(run: dict):
    """Print the success rate results of a run."""
    seed = run["seed"]
    val_ac = run["validation_accuracy"]
    test_ac = run["test_accuracy"]
    best_temperature = run["temperature"]
    best_top_p = run["top_p"]

    print(
        f"  Seed {seed}: "
        f"Validation accuracy {val_ac:.2f}, "
        f"{Fore.GREEN}{Style.BRIGHT}Test accuracy: {test_ac:.2f} {Style.RESET_ALL}, "
        f"Best temperature: {best_temperature:.2f},"
        f"Best top_p: {best_top_p:.2f},"
    )
        
