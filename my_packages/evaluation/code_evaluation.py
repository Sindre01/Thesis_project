from collections import defaultdict
import json
import numpy as np
from my_packages.evaluation.metrics import check_correctness, check_semantics, check_syntax, estimate_pass_at_k
from my_packages.evaluation.midio_compiler import compile_code, is_code_syntax_valid, is_code_semantically_valid, print_compiled_output
from my_packages.utils.server_utils import server_diagnostics
from my_packages.evaluation.models import invoke_anthropic_model, invoke_openai_model, invoke_o1_model, invoke_ollama_model
import re
from colorama import Fore, Back, Style

class Run:
    def __init__(
        self,
        phase: str,
        temperature: float,
        top_p: float,
        top_k: int,
        metric_results: dict,
        seed = None
    ):
        self.phase = phase
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.metric_results = metric_results
        self.seed = seed
    
    def print(self):

        if self.phase == "validation":
            print(
                f"\n===  VALIDATION ===\n"
                f"{Fore.GREEN}{Style.BRIGHT}"
                f"  > Temperature: {self.temperature:.2f}\n"
                f"  > Top_k: {self.top_k:.2f}\n"
                f"  > Top_p: {self.top_p:.2f}\n"
                f"{Style.RESET_ALL}"
                f"  > Optimized metric result:: {json.dumps(self.metric_results, indent=4)}"
            )

        elif self.phase == "testing":
            print(
                f"  = TESTING SEED {self.seed} =\n"
                f"  > Temperature: {self.temperature:.2f}\n"
                f"  > Top_k: {self.top_k:.2f}\n"
                f"  > Top_p: {self.top_p:.2f}\n"
                f"{Fore.GREEN}{Style.BRIGHT}"
                f"  > Metric results: {json.dumps(self.metric_results, indent=4)}"
                f"{Style.RESET_ALL}"
            )

        elif self.phase == "final":
            print(
                f"\n===  FINAL RESULTS ===\n"
                f"  > Temperature: {self.temperature:.2f}\n"
                f"  > Top_k: {self.top_k:.2f}\n"
                f"  > Top_p: {self.top_p:.2f}\n"
                f"{Fore.GREEN}{Style.BRIGHT}"
                f"  > Metric results: {json.dumps(self.metric_results, indent=4)}"
                f"{Style.RESET_ALL}"
            )

    

def extract_code(response_text):
    """Extract code snippet from the response using regex."""
    # Match content between ```language and ```
    match = re.search(fr"```midio(.*?)```", response_text, re.DOTALL)

    # Extract and clean up the code
    if match:
        return match.group(1).strip()  # Return only the code block

    # If no match, assume the response might already be code without markdown formatting
    return response_text.strip()

def calculate_pass_at_k_scores(result_dict, ks, metric):
    """
    Pass@k evaluation for a given result dictionary.
    Pass@k asses the probability that out of k samples, at least one was correct.

    Parameters:
    - result_dict: A dictionary of true results and generated candidates.
    - ks: A list of k values to calculate pass@k for.
    - run_tests: A boolean flag to indicate whether to run tests or not. If False, semantic correctness is used as metric.

    Returns:
    - dict: A dictionary of pass@k scores for each k value.
    """

    # test_results = run_tests(model_result, ks, debug) 
    #     (0, {"passed": True, "result": "Some details about the execution"}), #(id, test_result)
    #     (1, {"passed": False, "result": "Some details about the execution"}),
    #     (2, {"passed": True, "result": "Some details about the execution"}),
    # ]

    print("\n == Pass@k computation ==\n")

    if metric == "tests":
        #  for unit test runs
        print("Running tests..\n")
        #https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
        test_results = check_correctness(result_dict)

    elif metric == "syntax":
        print("Evaluating syntax..\n")
        # for semantic correctness
        test_results = check_syntax(result_dict)

    elif metric == "semantic":
        print("Evaluating semantic..\n")
        # for semantic correctness
        test_results = check_semantics(result_dict)
        print("test_results:\n")
        print(test_results)
    # elif metric == "EM": #Equal to the true response
    #     print("Evaluating EM..\n")
    #     test_results = check_EM(result_dict)

    total, correct = [], []
    #Endre true_result til en id som er unik for hver test
    for true_result, k_results in test_results.items():
        k_results.sort()
        passed = [result["passed"] for result in k_results] #if all tests have passed, then passed = True
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)
    print(f"Total: {total}, Correct: {correct}")
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() 
                 for k in ks if (total >= k).all()}
    print(f"Pass@K: {pass_at_k}")
    return pass_at_k

def run_model(
    client,
    messages,
    model,
    prompts,
    responses,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
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
                            client, full_prompt, model, max_new_tokens, temperature, top_p, top_k
                        )
                    elif "o1" in model:
                        generated = invoke_o1_model(
                            client, full_prompt, model, max_new_tokens
                        )
                    elif "gpt" in model:
                        generated = invoke_openai_model(
                            client, full_prompt, model, max_new_tokens, temperature, top_p, top_k, seed=new_seed
                        )
                    else:
                        generated = invoke_ollama_model(
                            client, full_prompt, model, max_new_tokens, temperature, top_p, top_k, seed=new_seed
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
                
            # Extract code from the generated response
            generated_code = extract_code(generated)
            generated_candidates.append(generated_code)

        # --- Now we have up to k responses for this prompt. ---

        # Extract the "true" code from the reference.
        true_response_code = extract_code(true_response)

        #Then add the generated candidates to the results dictionary
        results[true_response_code] = generated_candidates

        if debug:
            print(f"\n\n{Style.BRIGHT}=== Sample: {index} ===")
            print(f"{Fore.CYAN}{Style.BRIGHT} User prompt: {Style.RESET_ALL}\n{prompt}\n")
            for i, cand in enumerate(generated_candidates):
                print(f"{Fore.YELLOW}{Style.BRIGHT} Assistant response: #{i+1}:\n{cand}\n")
            print(f"{Fore.GREEN}{Style.BRIGHT} True response:{Style.RESET_ALL}\n {true_response_code}\n")

    return results

def evaluate_code(
    client,
    messages,
    model,
    prompts,
    responses,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    seed=None,
    ks=[1],                               
    debug=False,
    evaluation_metric = ["syntax", "semantic", "tests"]
):
    model_result = run_model(
        client,
        messages,
        model,
        prompts,
        responses,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        ks,
        seed,
        debug
    )

    metric_results = []
    for metric in evaluation_metric:
        if metric not in ["syntax", "semantic", "tests"]:
            raise ValueError("Invalid evaluation metric. Choose from 'syntax', 'semantic', 'tests'")
        else:
            pass_at_k_dict = calculate_pass_at_k_scores(model_result, ks, metric)
            metric_results.append(pass_at_k_dict)
    return metric_results

def run_validation(
        client, 
        messages, 
        model, 
        val_prompts, 
        val_responses, 
        temperatures, 
        top_ps, 
        top_ks,
        ks=[1], #optimizing for pass@k, for the first k in the list
        seed=None,
        debug=False,
        optimizer_metric="semantic" #tests, syntax, semantic
):
    val_best_pass_ks = {}
    val_best_metric = 0.0
    best_params = {"temperature": 0.5, "top_p": 0.5, "top_k": 50}

    print(f"{Fore.CYAN}{Style.BRIGHT}Validation Phase:{Style.RESET_ALL}")
    for temp in temperatures:
        for top_k in top_ks:
            for top_p in top_ps:
                # print(f"Validating with temperature: {temp}, top_k: {top_k} and top_p: {top_p}")
                metric_results_lists = evaluate_code (
                    client,
                    messages,
                    model['name'],
                    val_prompts,
                    val_responses,
                    model["max_tokens"],
                    temp,
                    top_p,
                    top_k,
                    seed,
                    ks=ks,
                    debug=debug,
                    evaluation_metric=[optimizer_metric]
                )
                pass_at_k_dict = metric_results_lists[0]
                val_metric = pass_at_k_dict[f"pass@{ks[0]}"]
                print(f"Validation with temp={temp}, top_k={top_k} and top_p={top_p}. Gave pass@{ks[0]}={val_metric} and pass@ks={pass_at_k_dict}")
                #Optimize for the best pass@ks[0] for the provided metric
                if  val_metric > val_best_metric:
                    val_best_metric = val_metric
                    val_best_pass_ks = pass_at_k_dict
                    best_params = {"temperature": temp, "top_p": top_p, "top_k": top_k}

    result = Run(
        phase="validation",
        temperature=best_params["temperature"],
        top_p=best_params["top_p"],
        top_k=best_params["top_k"],
        metric_results={f"pass@k_{optimizer_metric}": val_best_pass_ks},
        seed=seed
    )

    if debug:
        result.print()

    return result

def run_testing(
    client,
    messages, 
    model, 
    test_prompts, 
    test_responses, 
    temperature, 
    top_p, 
    top_k,
    ks, 
    seeds, 
    debug=False,
    metrics=["syntax", "semantic", "tests"]
):
    results = []
    
    #Test the model on the test set with different seeds and the best hyperparameters.
    print(f"{Fore.CYAN}{Style.BRIGHT}Testing Phase:{Style.RESET_ALL}")

    for seed in seeds:
        print(f"\nTesting with Seed: {seed} ..", end="\r")

        metric_results_lists = evaluate_code(
            client,
            messages,
            model['name'],
            test_prompts,
            test_responses,
            model["max_tokens"],
            temperature,
            top_p,
            top_k,
            seed,
            ks,
            debug,
            evaluation_metric=metrics
        )

        new_run = Run(
            phase="testing",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            metric_results=
            { # pass@k for each metric. E.g. pass@k syntax, pass@k semantic and pass@k tests
                # e.g. {"pass@k_syntax": {pass@1: 0.1}, "pass@k semantic": {pass@1: 0.1}}
                f"pass@k_{metrics[i]}": result # result is a dictionary of pass@k scores for each k value. 
                for result, i in metric_results_lists
            },
            seed=seed
        )

        results.append(new_run)
        if debug:
            new_run.print()

    return results

    
 
def calculate_final_result(testing_runs) -> Run:
    """Calculate the standard deviation mean of the metrics, across all testing runs.

    Parameters: 
        - testing_runs: A list of dictionaries containing the test results. 
            Example:
            [
                {
                    "seed": 0,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "top_k": 50,
                    "metric_results": {
                        "pass@k syntax": {"pass@1": 0.1, "pass@5": 0.3},
                        "pass@k semantic": {"pass@1": 0.2, "pass@5": 0.4}
                    }
                }
            ]
    Returns:
        - dict: A dictionary containing the mean and standard deviation of the metrics.
            Example:
            {
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 50,
                "metric_results": {
                    "pass@k_syntax": {"pass@1": 0.1, "pass@5": 0.3},
                    "pass@k_semantic": {"pass@1": 0.2, "pass@5": 0.4}
            }
    """

    pass

# def print_validation_result(run: dict):
#     """Print the results of a run.
#     run: A dictionary containing the results of a run.
#     e.g. {
#         "seed": seed,
#         "temperature": best_params["temperature"],
#         "top_p": best_params["top_p"],
#         "top_k": best_params["top_k"],
#         "metric_results": {
#             f"pass@k_{optimizer_metric}": pass_at_k_dict
#         }
#     """
    
#     best_temperature = run["temperature"]
#     best_top_p = run["top_p"]
#     best_top_k = run["top_k"]
#     pass_at_k = run["metric_results"]

#     print(
#         f"  = Validation =\n"
#         f"  > Best temperature: {best_temperature:.2f}\n"
#         f"  > Best top_k: {best_top_k:.2f}\n"
#         f"  > Best top_p: {best_top_p:.2f}\n"
#         f"  > Optimized metric result: {json.dumps(pass_at_k, indent=4)}\n"
#     )

# def print_test_result(run: dict):
#     """Print the result of a run."""
#     seed = run["seed"]
#     temperature = run["temperature"]
#     top_p = run["top_p"]
#     top_k = run["top_k"]
#     pass_at_ks = run["metric_results"]

#     print(
#         f"===  Seed {seed} ===\n"
#         f"  > Temperature: {temperature:.2f}\n"
#         f"  > Top_k: {top_k:.2f}\n"
#         f"  > Top_p: {top_p:.2f}\n"
#         f"  = Test =\n"
#         f"{Fore.GREEN}{Style.BRIGHT}"
#         f"  > Metric results: {json.dumps(pass_at_ks, indent=4)}"
#         f"{Style.RESET_ALL}"
#     )

# def print_final_result(run: dict):
#     """Print the result of a run."""
#     temperature = run["temperature"]
#     top_p = run["top_p"]
#     top_k = run["top_k"]
#     pass_at_ks = run["metric_results"]

#     print(
#         f"\n===  FINAL RESULTS ===\n"
#         f"  > Temperature: {temperature:.2f}\n"
#         f"  > Top_k: {top_k:.2f}\n"
#         f"  > Top_p: {top_p:.2f}\n"
#         f"{Fore.GREEN}{Style.BRIGHT}"
#         f"  > Metric results: {json.dumps(pass_at_ks, indent=4)}"
#         f"{Style.RESET_ALL}"
#     )
