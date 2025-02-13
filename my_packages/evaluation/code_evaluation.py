from collections import defaultdict
import json
from typing import Tuple
import numpy as np
from my_packages.evaluation.metrics import check_correctness, check_semantics, check_syntax, estimate_pass_at_k
from my_packages.prompting.few_shot import create_few_shot_prompt, create_final_prompt, get_prompt_template_variables
from my_packages.utils.server_utils import server_diagnostics
import re
from colorama import Fore, Back, Style
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langsmith.schemas import Example

class Run:
    def __init__(
        self,
        phase: str,
        temperature: float,
        top_p: float,
        top_k: int,
        metric_results: dict,
        seed = None,
        metadata: dict | None = None
    ):
        self.phase = phase
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.metric_results = metric_results
        self.seed = seed
        self.metadata = metadata
    
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
                f" > Metadata: {json.dumps(self.metadata, indent=4)}"
            )

        elif self.phase == "testing":
            print(
                f"  = TESTING SEED {self.seed} =\n"
                f"  > Temperature: {self.temperature:.2f}\n"
                f"  > Top_k: {self.top_k:.2f}\n"
                f"  > Top_p: {self.top_p:.2f}\n"
                f"{Fore.GREEN}{Style.BRIGHT}"
                f"  > Metric results: {json.dumps(self.metric_results, indent=4)}"
                f"  > Metadata: {json.dumps(self.metadata, indent=4)}"
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
                f"  > Metadata: {json.dumps(self.metadata, indent=4)}"
                f"{Style.RESET_ALL}"
            )

    


def extract_code(response_text: str) -> str:
    """
    Extracts a code snippet from the response using a regex for ```midio code blocks.
    Also removes any line that starts with // or #.
    """
    # Match content between ```midio and ```
    match = re.search(r"```midio(.*?)```", response_text, re.DOTALL)
    
    # Extract code or assume entire response might be the code
    if match:
        code_block = match.group(1)
    else:
        code_block = response_text
    
    # Remove lines that start with // or #
    # Explanation:
    #   - ^\s*  : Matches the start of a line plus any leading spaces
    #   - (?:\/\/|#) : Matches // or #
    #   - .*?$ : Matches the rest of the line (non-greedy)
    # The 'flags=re.MULTILINE' ensures '^' and '$' match start/end of each line
    code_without_comments = re.sub(r'^\s*(?:\/\/|#).*?$', '', code_block, flags=re.MULTILINE)

    # Strip leading/trailing whitespace and return
    return code_without_comments.strip()

def calculate_pass_at_k_scores(
        result_dict: dict[int, list[str]],
        ks: list[int], 
        metric: str
    ):
    """
    Pass@k evaluation for a given result dictionary.
    Pass@k asses the probability that out of k samples, at least one was correct.

    Parameters:
    - result_dict: A dictionary of taks_id and generated candidates. dict[int, list[str]]
        E.g:
        (<task_id>, {"passed": True, "result": "Some details about the execution"}),
        (<task_id>, {"passed": False, "result": "Some details about the execution"}),
        (<task_id>, {"passed": True, "result": "Some details about the execution"}),
    ]
    - ks: A list of k values to calculate pass@k for.
    - metric: which metric to calulcate pass_at_k for. E.g. syntax, semantic, tests

    Returns:
    - dict: A dictionary of pass@k scores for each k value.
    """
    

    if metric == "tests":
        print("\n == Pass@k computation for tests ==\n")
        #https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
        test_results = check_correctness(result_dict)

    elif metric == "syntax":
        print("\n == Pass@k computation for syntax ==\n")
        test_results = check_syntax(result_dict)

    elif metric == "semantic":
        print("\n == Pass@k computation for semantics ==\n")
        test_results = check_semantics(result_dict)

    # elif metric == "EM": #Equal to the true response
    #     print("Evaluating EM..\n")
    #     test_results = check_EM(result_dict)

    total, correct = [], []
    for task_id, k_results in test_results.items():
        # k_results.sort(key=lambda d: d["passed"]) #sort the results by the passed key
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
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model,
    available_nodes,
    data : list[Example],
    example_pool,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    ks=[1], 
    seed=None,                              
    debug=False,
    metrics=None
)-> tuple[dict[int, list[str]], int]:
    
    results: dict[int, list[str]] = {}
    largest_prompt_ctx_size = 0
    for index, sample in enumerate(data):
        generated_candidates: list[str] = []
        true_response = sample.outputs["response"]
        true_response_code = extract_code(true_response)
        task = sample.inputs["task"]
        task_id = int(sample.metadata["task_id"])
        
        few_shot_examples = example_pool.select_examples(sample.inputs)
        if "tests" in metrics: # Uses signature prompt
            few_shot = create_few_shot_prompt(few_shot_examples, 'CODE_SIGNATURE_TEMPLATE')
            final_prompt_template = create_final_prompt(few_shot, "CODE_GENERATOR_TEMPLATE", "CODE_SIGNATURE_TEMPLATE")

            prompt_variables_dict ={
                "task": task, 
                "function_signature": sample.inputs["function_signature"],
                "external_functions": available_nodes
            }
        else: # Uses regular prompt
            few_shot = create_few_shot_prompt(few_shot_examples, 'CODE_TEMPLATE')
            final_prompt_template = create_final_prompt(few_shot, "CODE_GENERATOR_TEMPLATE", "CODE_TEMPLATE")
            prompt_variables_dict ={
                "task": task, 
                "external_functions": available_nodes
            }

        prompt = final_prompt_template.format(**prompt_variables_dict)
        prompt_size = client(model=model).get_num_tokens(prompt) # Will print warning if prompt is too big for model
        print(f"Tokens in the final prompt: {prompt_size}")
        
        if prompt_size > largest_prompt_ctx_size:
            largest_prompt_ctx_size = prompt_size
        
        for attempt_i in range(max(ks)):
            max_retries = 3
            retries = 0
            new_seed = seed * attempt_i if seed else None # different seed for each attempt if not None
            generated = ""
            while retries < max_retries:
                try:
                    print(f"Generating response for sample {index + 1}..", end="\r")
                    
                    if "claude" in model:
                        llm = client(
                            model=model,
                            temperature=temperature,
                            num_predict=max_new_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            stream=False,
                            num_ctx=largest_prompt_ctx_size,
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
                            num_ctx=largest_prompt_ctx_size,
                            stop=["```<|eot_id|>"],
                            seed=new_seed
                        )
                    
                    chain = (final_prompt_template | llm)

                    response = chain.invoke(
                        prompt_variables_dict,
                        {"run_name": f"Few-shot code prediction"}
                    )

                    generated = response.content
                    break  # If generation succeeded, break out of retry loop
                except Exception as e:
                    retries += 1
                    print(f"Attempt {retries} failed with error: {e}")
                    server_diagnostics()

            if retries == max_retries:
                print("Failed to get a response from the server after "
                      + str(retries) + " attempts.")
                generated = ""
                
            # Extract code from the generated response
            generated_code = extract_code(generated)
            generated_candidates.append(generated_code)

            generated_example: Example = sample.copy(update={"outputs": {"response": generated_code}})
        # --- Now we have up to k responses for this prompt. ---

        results[task_id] = generated_candidates

        if debug:
            print(f"\n\n{Style.BRIGHT}=== Sample: {index} ===")
            print(f"{Fore.CYAN}{Style.BRIGHT} User prompt: {Style.RESET_ALL}\n{prompt}\n")
            for i, cand in enumerate(generated_candidates):
                print(f"{Fore.YELLOW}{Style.BRIGHT} Assistant response: #{i+1}:\n{cand}\n")
            print(f"{Fore.GREEN}{Style.BRIGHT} True response:{Style.RESET_ALL}\n {true_response_code}\n")

    return results, largest_prompt_ctx_size

def evaluate_code(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model,
    available_nodes,
    data,
    example_pool,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    seed,
    ks,                               
    debug=False,
    evaluation_metric = ["syntax", "semantic", "tests"]
)-> tuple[list[dict[str, dict[int, float]]], int]:
    
    model_result, largest_context = run_model(
        client,
        model,
        available_nodes,
        data,
        example_pool,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        ks,
        seed,
        debug, 
        evaluation_metric
    )

    metric_results = []
    for metric in evaluation_metric:
        if metric not in ["syntax", "semantic", "tests"]:
            raise ValueError("Invalid evaluation metric. Choose from 'syntax', 'semantic', 'tests'")
        else:
            pass_at_k_dict = calculate_pass_at_k_scores(model_result, ks, metric)
            metric_results.append(pass_at_k_dict)
    return metric_results, largest_context

def run_validation(
        client: ChatOllama | ChatOpenAI | ChatAnthropic,
        model, 
        available_nodes, 
        val_data: list[Example],
        example_pool: list[Example],
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
                metric_results_lists, largest_context = evaluate_code (
                    client,
                    model['name'],
                    available_nodes,
                    val_data,
                    example_pool,
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
                if val_metric > val_best_metric:
                    val_best_metric = val_metric
                    val_best_pass_ks = pass_at_k_dict
                    best_params = {"temperature": temp, "top_p": top_p, "top_k": top_k}

    result = Run(
        phase="validation",
        temperature=best_params["temperature"],
        top_p=best_params["top_p"],
        top_k=best_params["top_k"],
        metric_results={f"pass@k_{optimizer_metric}": val_best_pass_ks},
        seed=seed,
        metadata={"largest_prompt_size": largest_context}
    )

    if debug:
        result.print()

    return result

def run_testing(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model, 
    available_nodes, 
    test_data: list[Example],
    example_pool: list[Example],
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

        metric_results_lists, largest_context = evaluate_code(
            client,
            model['name'],
            available_nodes,
            test_data,
            example_pool,
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
                f"pass@k_{metrics[i]}": metric_results # result is a dictionary of pass@k scores for each k value. 
                for i, metric_results in enumerate(metric_results_lists)
            },
            seed=seed,
            metadata={"largest_prompt_size": largest_context}
        )

        results.append(new_run)
        if debug:
            new_run.print()

    return results

from collections import defaultdict
import numpy as np

def calculate_final_result(testing_runs: list[Run]) -> Run:
    """
    Calculate the mean and standard deviation of the metrics, across all testing runs.

    Parameters: 
        testing_runs: A list of Run objects (or dictionaries) containing the test results.
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
                    "metadata": {"largest_prompt_size": 1000}
                },
                ...
            ]
    Returns:
        A final Run object that contains the mean and standard deviation for each metric.
        Example:
            {
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 50,
                "metric_results": {
                    "pass@k_syntax": {
                        "pass@1": {"mean": 0.1, "std": 0.0},
                        "pass@5": {"mean": 0.3, "std": 0.0}
                    },
                    "pass@k_semantic": {
                        "pass@1": {"mean": 0.2, "std": 0.0},
                        "pass@5": {"mean": 0.4, "std": 0.0}
                    }
                }
            }
    """

    # Nested dictionary to collect all values per metric and per pass@k:
    # Structure: { metric_name: { pass@k: [list of values across runs] } }
    aggregated_metrics = defaultdict(lambda: defaultdict(list))
    
    for run in testing_runs:
        for metric_name, metric_vals in run.metric_results.items():

            for pass_k, value in metric_vals.items():
                aggregated_metrics[metric_name][pass_k].append(value)
    
    # Build the final metric_results with computed mean and std.
    final_metric_results = {}
    for metric_name, pass_data in aggregated_metrics.items():
        metric_summary = {}
        for pass_k, values in pass_data.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            metric_summary[pass_k] = {"mean": mean_val, "std": std_val}
        # Normalize metric name if desired (e.g., replace spaces with underscores)
        final_metric_results[metric_name.replace(" ", "_")] = metric_summary

    final_metric_results = round_results(final_metric_results)
    # Return a new Run object for the final result
    return Run(
        phase="final",
        temperature=testing_runs[0].temperature,
        top_p=testing_runs[0].top_p,
        top_k=testing_runs[0].top_k,
        metric_results=final_metric_results,
        metadata=testing_runs[0].metadata
    )
def round_results(metric_results, ndigits=2):
    rounded = {}
    for metric, passes in metric_results.items():
        rounded[metric] = {}
        for pass_k, stats in passes.items():
            rounded[metric][pass_k] = {
                "mean": round(stats["mean"], ndigits),
                "std": round(stats["std"], ndigits)
            }
    return rounded


