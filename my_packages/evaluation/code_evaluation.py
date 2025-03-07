from collections import defaultdict
import numpy as np
from my_packages.common import CodeEvaluationResult, PromptType, Run
from my_packages.db_service.best_params_service import save_best_params_to_db
from my_packages.db_service.error_service import save_errors_to_db
from my_packages.db_service.results_service import save_results_to_db
from my_packages.evaluation.metrics import check_correctness, check_semantics, check_syntax, estimate_pass_at_k
from my_packages.prompting.few_shot import create_few_shot_prompt, create_final_prompt

from my_packages.utils.file_utils import save_results_as_string, save_results_to_file
from my_packages.utils.server_utils import server_diagnostics
import re
from colorama import Fore, Back, Style
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.example_selectors.base import BaseExampleSelector

def extract_code(response_text: str) -> str:
    """
    Extracts a code snippet from the response using a regex for ```midio code blocks.
    Also removes any comments—whether they are on lines by themselves or inline.
    """
    # Split the response to get content after the last </think>
    parts = response_text.rsplit('</think>', 1)
    code_section = parts[-1]  # Content after the last </think>

    # Find all `midio` code blocks
    matches = re.findall(r'```midio(.*?)(```|$)', code_section, re.DOTALL)

    # If multiple code blocks exist, take the last one
    if matches:
        code_block = matches[-1][0]  # Get only the code part
    else:
        code_block = "No Midio code found in response!\n"
        code_block += code_section

    # This regex finds any occurrence of '//' or '#' and removes everything until the end of the line.
    code_without_comments = re.sub(r'(?://|#).*$', '', code_block, flags=re.MULTILINE)

    return code_without_comments.strip()

def evaluate_code_metric(
      result_dict: dict[int, list[str]],
      metric: str
    ) -> dict[int, list[CodeEvaluationResult]]:
    """
    Parameters:
        - candidates_dict: A dictionary of task_id and generated candidates. dict[int, list[str]]
              E.g:
                    {
                        {<task_id>, [candidate1, candidate2, candidate3]},
                        {<task_id>, [candidate1, candidate2, candidate3]},
                        {<task_id>, [candidate1, candidate2, candidate3]},
                    }
        - metric: which metric to calulcate pass_at_k for. E.g. syntax, semantic, tests
    Returns:
        - test_results: A dictionary of task_id and CodeEvaluationResult objects.
    
    """
    if metric == "syntax":
        test_results = check_syntax(result_dict)
    elif metric == "semantic":
        test_results = check_semantics(result_dict)
    elif metric == "tests":
        test_results = check_correctness(result_dict)
    
    return test_results

def calculate_pass_at_k_scores(
        test_results: dict[int, CodeEvaluationResult],
        ks: list[int], 
    ):
    """
    Pass@k evaluation for a given result dictionary.
    Pass@k asses the probability that out of k samples, at least one was correct.

    Parameters:
        - result_dict: A dictionary of taks_id and generated candidates. dict[int, list[str]]
            E.g:
            [
                (<task_id>, {"passed": True, "result": "Some details about the execution"}),
                (<task_id>, {"passed": False, "result": "Some details about the execution"}),
                (<task_id>, {"passed": True, "result": "Some details about the execution"}),
            ]
        - ks: A list of k values to calculate pass@k for.
        - metric: which metric to calulcate pass_at_k for. E.g. syntax, semantic, tests

    Returns:
        - dict: A dictionary of pass@k scores for each k value.
    """

    total, correct = [], []
    for task_id, eval_results in test_results.items():
        # eval_results.sort(key=lambda d: d["passed"]) #sort the results by the passed key
        passed = [result.passed for result in eval_results] #if all tests have passed, then passed = True
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)
    # print(f"Total: {total}, Correct: {correct}")

    pass_at_k = {}

    for k in ks:
        # Check if the total is greater than or equal to k for all elements
        # only calculate pass@k for this k if all elements have at least k samples
        if (total >= k).all():
            # Estimate the pass rate at k and take the mean
            pass_at_k[f"pass@{k}"] = estimate_pass_at_k(total, correct, k).mean()

    # Now pass_at_k contains the pass rates for each valid k
    # print(f"Pass@K: {pass_at_k}")

    return pass_at_k


def run_model(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model,
    available_nodes,
    data: list[dict[str, str]],
    example_pool: BaseExampleSelector,
    max_new_tokens: int,
    temperature,
    top_p,
    top_k,
    n=1, # Number of generations per task
    seed=None,                              
    debug=False,
    prompt_type=PromptType.REGULAR,
    ollama_port="11434"
)-> tuple[dict[int, list[str]], int]:
    
    results: dict[int, list[str]] = {}
    largest_prompt_ctx_size = 0
    for index, sample in enumerate(data):
        generated_candidates: list[str] = []
        true_response = sample["response"]
        true_response_code = extract_code(true_response)
        task = sample["task"]
        task_id = int(sample["task_id"])
        
        few_shot_examples = example_pool.select_examples(sample)
        
        if prompt_type == PromptType.SIGNATURE: # Uses signature prompt
            few_shot = create_few_shot_prompt(few_shot_examples, 'CODE_SIGNATURE_TEMPLATE')
            final_prompt_template = create_final_prompt(few_shot, "CODE_GENERATOR_TEMPLATE", "CODE_SIGNATURE_TEMPLATE")

            prompt_variables_dict ={
                "external_functions": available_nodes,
                "task": task, 
                "function_signature": sample["function_signature"],
            }
        elif prompt_type == PromptType.REGULAR: # Uses regular prompt
            few_shot = create_few_shot_prompt(few_shot_examples, 'CODE_TEMPLATE')
            final_prompt_template = create_final_prompt(few_shot, "CODE_GENERATOR_TEMPLATE", "CODE_TEMPLATE")
            prompt_variables_dict ={
                "external_functions": available_nodes,
                "task": task, 
            }
        elif prompt_type == PromptType.COT: # Uses COT prompt
            few_shot = create_few_shot_prompt(few_shot_examples, 'COT_TEMPLATE')
            final_prompt_template = create_final_prompt(few_shot, "CODE_GENERATOR_TEMPLATE", "COT_TEMPLATE")
            prompt_variables_dict = {
                "external_functions": available_nodes,
                "task": task, 
                "function_signature": sample["function_signature"],
            }
        else:
            raise ValueError("Invalid prompt type. Choose from 'signature', 'regular' or 'cot'")

        prompt = final_prompt_template.format(**prompt_variables_dict)
        prompt_size = client(model=model).get_num_tokens(prompt) # Will print warning if prompt is too big for model

        
        if prompt_size > largest_prompt_ctx_size:
            largest_prompt_ctx_size = prompt_size

        print(f"Generating response..  ({index + 1}/{len(data)})", end="\r")

        current_n = 0
        for attempt_i in range(n):
            max_retries = 3
            retries = 0
            new_seed = seed * attempt_i if seed else None # different seed for each attempt if not None
            generated = ""
            while retries < max_retries:
                try:

                    print(f"    > Generating k response..  ({current_n + 1}/{n})", end="\r")
                    if "gpt" in model:
                        llm = client(
                            model=model,
                            temperature=temperature,
                            seed=new_seed,
                            max_tokens=max_new_tokens,
                            stop=["```<|eot_id|>"],
                            top_p=top_p,
                            # top_k=top_k, #NOT AVAILABLE IN GPT
                            streaming=False,
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
                            seed=new_seed,
                            base_url=f"http://localhost:{ollama_port}"
                        )
                    
                    chain = (final_prompt_template | llm)

                    response = chain.invoke(
                        prompt_variables_dict,
                        {"run_name": f"Few-shot code prediction"}
                    )
                    print(generated)
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
            current_n += 1
            # Extract code from the generated response
            generated_code = extract_code(generated)
            generated_candidates.append(generated_code)
        # --- Now we have up to k responses for this prompt. ---

        results[task_id] = generated_candidates

        if debug:
            print(f"\n\n{Style.BRIGHT}=== Sample: {index+1} ===")
            print(f"{Fore.CYAN}{Style.BRIGHT} User prompt: {Style.RESET_ALL}\n{prompt}\n")
            print(f"{Fore.YELLOW}{Style.BRIGHT} Full Assistant response: #{1}:\n{generated}\n")
            for i, cand in enumerate(generated_candidates):
                print(f"{Fore.YELLOW}{Style.BRIGHT} Assistant response: #{i+1}:\n{cand}\n")
            print(f"{Fore.GREEN}{Style.BRIGHT} True response:{Style.RESET_ALL}\n {true_response_code}\n")

    return results, largest_prompt_ctx_size

def evaluate_code(
    candidate_dict: dict[int, list[str]],
    ks: list[int],                               
    evaluation_metric:list[str],
    experiment_name: str,
    model_name: str,
    env: str,
    hyperparams: dict,
    phase: str,
    db_connection=None
)-> list[dict[str, dict[int, float]]]:
    """
    Evaluate the code quality of the generated candidates.

    Returns: a list of dictionaries containing the pass@k scores for each metric and ks.
    - E.g. [

        {
            "pass@k_syntax": {1: 0.1, 5: 0.3},
    """

    metric_results = []
    for metric in evaluation_metric:
        if metric not in ["syntax", "semantic", "tests"]:
            raise ValueError("Invalid evaluation metric. Choose from 'syntax', 'semantic', 'tests'")
        else:
            test_results = evaluate_code_metric(candidate_dict, metric)
            pass_at_k_dict = calculate_pass_at_k_scores(test_results, ks)
            print(f"Pass@k for {metric}: {pass_at_k_dict}")

            # Save errors
            if env == "dev":
                save_results_as_string(test_results, f"{metric}_{experiment_name}.txt")
                save_results_to_file(test_results, f"{metric}_{experiment_name}.json")
            elif env == "prod":
                save_errors_to_db(
                    experiment_name,
                    model_name,
                    test_results,
                    hyperparams,
                    phase,
                    db_connection
                )
                # print(f"✅ Errors saved to database for {metric} in {experiment_name}")

            metric_results.append(pass_at_k_dict)

    return metric_results

def run_validation(
        client: ChatOllama | ChatOpenAI | ChatAnthropic,
        model, 
        available_nodes, 
        val_data: list[dict],
        example_pool: BaseExampleSelector,
        temperatures: list[float], 
        top_ps: list[float],
        top_ks: list[int], 
        ks: list[int], #optimizing for pass@k, for the first k in the list
        seed: int, 
        debug: bool,
        optimizer_metric: str, #tests, syntax, semantic
        experiment_name: str,
        env: str,
        prompt_type: PromptType
    ) -> Run:

    """
    Run the model on the validation set with different hyperparameters and seeds.
    Calculate the pass@k for the optimizer_metric and ks provided. We use always k=1.

    Returns: 
        A Run object containing the best hyperparameters and pass@k for the optimizer
    """
    print(f"Starting validation phase..")
    val_best_metric = 0.0
    best_run = Run("validation", 0.2, 0.6, 10, {f"pass@k_{optimizer_metric}": {"pass@1": 0.0}}, 9, {})
    n = max(ks) # Number of generations per task
    print(f"{Fore.CYAN}{Style.BRIGHT}Validation Phase:{Style.RESET_ALL}")
    print(f"Optimizing for metric: {optimizer_metric}")

    if model["name"] in "gpt-4o":
        print("RESETTING TOP_K TO NONE FOR OPENAI MODELS")
        top_ks = []
        
    for temp in temperatures:
        for top_k in top_ks or [-1]: #Ensures loop runs once, when top_ks is empty
            for top_p in top_ps:
                # print(f"Validating with temperature: {temp}, top_k: {top_k} and top_p: {top_p}")
                
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

                metric_results_lists = evaluate_code (
                    model_result,
                    ks=ks,
                    evaluation_metric=[optimizer_metric],
                    experiment_name=experiment_name,
                    model_name=model["name"],
                    env=env,
                    hyperparams={"seed": seed, "temperature": temp, "top_p": top_p, "top_k": top_k},
                    phase="validation"
                )
                ## Optimizing for the first k in the ks list
                pass_at_k_dict = metric_results_lists[0]
                val_metric = pass_at_k_dict[f"pass@{ks[0]}"]
                print(f"Validation with temp={temp}, top_k={top_k} and top_p={top_p}. Gave pass@{ks[0]}={val_metric} and pass@ks={pass_at_k_dict}")
                
                #Optimize for the best pass@ks[0] for the provided metric
                if val_metric > val_best_metric:
                    val_best_metric = val_metric
                    best_run = Run(
                        phase="validation",
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                        metric_results={f"pass@k_{optimizer_metric}": pass_at_k_dict},
                        seed=seed,
                        metadata={"largest_prompt_size": largest_context}
                    )

    # Save to MongoDB
    if env == "prod":
        save_best_params_to_db(
            experiment_name, 
            model["name"], 
            optimizer_metric, 
            best_run
        )

    if debug:
        best_run.print()

    return best_run

def run_testing(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model: dict[str,str], 
    available_nodes, 
    test_data: list[dict],
    example_pool: BaseExampleSelector,
    temperature:float, 
    top_p:float, 
    top_k:int,
    ks:list[int], 
    seeds:list[int], 
    debug: bool,
    metrics:list[str],
    experiment_name: str,
    env: str,
    prompt_type: PromptType
) -> tuple[list[Run], Run]:
    """
    Run the model on the test set with different seeds and the best hyperparameters.
    Calculate the pass@k for each metric and ks provided.

    Returns: a list of Run objects containing the test results for each seed.
    """
    print(f"Starting testing phase..")
    results = []
    n = max(ks) # Number of generations per task
    #Test the model on the test set with different seeds and the best hyperparameters.
    print(f"{Fore.CYAN}{Style.BRIGHT}Testing Phase:{Style.RESET_ALL}")

    for seed in seeds:
        print(f"\nTesting with Seed: {seed} ..", end="\r")

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
            debug, 
            prompt_type
        )

        metric_results_lists = evaluate_code (
            model_result,
            ks=ks,
            evaluation_metric=metrics,
            experiment_name=experiment_name,
            model_name=model["name"],
            env=env,
            hyperparams={"seed": seed, "temperature": temperature, "top_p": top_p, "top_k": top_k},
            phase="testing"
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


    final_result = calculate_final_result(results)

    save_results_to_db(
        experiment_name,
        model["name"],
        seeds,
        ks,
        metrics,
        final_result
    )
    print(f"✅ Final results saved to database for '{model['name']}' in {experiment_name}")
    return results, final_result

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


