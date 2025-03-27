from collections import defaultdict
import os
import numpy as np
import torch
from my_packages.common.classes import CodeEvaluationResult, PromptType, Run
from my_packages.common.rag import RagData
from my_packages.db_service.best_params_service import save_best_params_to_db
from my_packages.db_service.error_service import save_errors_to_db
from my_packages.db_service.results_service import save_results_to_db
from my_packages.evaluation.metrics import check_correctness, check_semantics, check_syntax, estimate_pass_at_k
from my_packages.evaluation.models import generate_n_responses
from my_packages.prompting.prompt_building import add_RAG_to_prompt, build_prompt, code_data_to_node_data
from my_packages.utils.file_utils import save_results_as_string, save_results_to_file
import re
from syncode import Syncode
from colorama import Fore, Style
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.example_selectors.base import BaseExampleSelector
from my_packages.utils.tokens_utils import measure_prompt_tokens
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

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


def two_step_run(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model: str,
    dataset_nodes: list[dict],
    all_nodes: list[dict],
    data: list[dict],
    example_pool: BaseExampleSelector,
    node_max_new_tokens: int,
    code_max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    n=1,
    seed=None,
    debug=False,
    prompt_type: PromptType = PromptType.REGULAR,
    ollama_port="11434",
    rag_data: RagData = None,
    max_ctx=16000,
) -> tuple[dict[int, list[str]], int]:
    """
    Run a model on a list of tasks in two stages:
      1) Node step
      2) Code step
    Return a dict mapping task_id -> list of generated code snippets,
    plus the largest context usage observed.
    """

    # Basic input checks
    if "phi" in model.lower() and max_ctx > 16000:
        raise ValueError("Max context for Phi model is 16000 tokens.")
    elif max_ctx > 130000:
        raise ValueError("Max context in our setup is 130000 tokens.")


    results: dict[int, list[str]] = {}
    largest_ctx_size = 0

    for idx, sample in enumerate(data):
        node_candidates = []
        # (A) Step 1: Node step (optionally use rag_data to retrieve nodes if you want).
        generation_kwargs = { # Get the best from validation
            "client": client,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "n": 1,
            "debug": debug,
        }
        node_candidates, node_prompt_size = run_prompt_step(
            response_type="NODE",
            sample=sample,
            example_pool=example_pool,
            prompt_type=prompt_type,
            dataset_nodes=dataset_nodes,
            all_nodes=all_nodes,
            client=client,
            model=model,
            max_ctx=max_ctx,
            max_new_tokens=node_max_new_tokens,
            generation_kwargs=generation_kwargs,
            rag_data=None, # Not using RAG
            debug=debug,
            ollama_port=ollama_port
        )
        node_candidates = node_candidates[0].split(",")
        print("Extracted nodes into list: ", node_candidates)

        # (B) Step 2: Code step
        generation_kwargs = { #
            "client": client,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n": n,
            "seed": seed,
            "debug": debug,
        }
        code_candidates, code_prompt_size = run_prompt_step(
            response_type="CODE",
            sample=sample,
            example_pool=example_pool,
            prompt_type=prompt_type,
            dataset_nodes=dataset_nodes,
            all_nodes=all_nodes,
            client=client,
            model=model,
            max_ctx=max_ctx,
            max_new_tokens=code_max_new_tokens,
            generation_kwargs=generation_kwargs,
            rag_data=rag_data,
            debug=debug,
            candidate_nodes=node_candidates,
            ollama_port=ollama_port
        )

        # Track largest context usage
        used_ctx_node = node_prompt_size + node_max_new_tokens
        used_ctx_code = code_prompt_size + code_max_new_tokens
        largest_ctx_size = max(largest_ctx_size, used_ctx_node, used_ctx_code)

        # Store final code from step 2 in results
        task_id = int(sample["task_id"])
        true_response_code = sample["response"]
        results[task_id] = code_candidates

        print(f"\nDone with: === Task {idx+1}/{len(data)} (TASK ID={task_id}) ===")
        
    return results, largest_ctx_size

def run_prompt_step(
    response_type: str, # CODE or NODE
    sample: dict,
    example_pool: BaseExampleSelector,
    prompt_type: PromptType,
    dataset_nodes: list[dict],
    all_nodes: list[dict],
    client,
    model: str,
    max_ctx: int,
    max_new_tokens: int,
    generation_kwargs: dict,
    rag_data: RagData = None,
    debug: bool = False,
    candidate_nodes: list = [],
    ollama_port: str = "11434"
) -> tuple[list[str], int]:
    """
    1. Build prompt
    2. Measure tokens.
    3. Possibly add RAG context. Between System message and few-shots.
    4. Generate responses.
    5. Return generated candidates & final prompt size.
    """
    # Build the base prompt
    true_response = sample["response"]
    few_shot_examples = example_pool.select_examples(sample)
    available_nodes = dataset_nodes
    
    if response_type == "NODE":
        few_shot_examples= code_data_to_node_data(few_shot_examples)
        true_response = sample["external_functions"]
        available_nodes = all_nodes
        
    prompt, final_prompt_template, prompt_variables_dict = build_prompt(
        response_type=response_type,
        prompt_type=prompt_type,
        few_shot_examples=few_shot_examples,
        sample=sample,
        available_nodes=available_nodes,
    )
    # Measure initial prompt size
    prompt_size = measure_prompt_tokens(client, model, prompt, max_ctx)

    # Reserve output tokens and get the leftover context for RAG
    print(f"Using {max_ctx} for context window")    
    available_ctx = max_ctx - prompt_size - max_new_tokens
    if debug:
        print(f"\nPrompt size = {prompt_size}, leaving {available_ctx} tokens for RAG + buffer.")

    # Possibly inject RAG data
    if rag_data:
        # print(f"available context after subtracting prompt_size of {prompt_size} tokens and output tokens of {max_new_tokens} tokens: {available_ctx}")
        prompt, final_prompt_template, prompt_variables_dict, used_tokens = add_RAG_to_prompt(
            client = client,
            model = model,
            task = sample["task"],
            available_ctx = available_ctx,
            rag_data = rag_data,
            final_prompt_template = final_prompt_template,
            prompt_variables_dict = prompt_variables_dict,
            candidate_nodes = candidate_nodes,
            all_nodes = all_nodes,
        )
        prompt_size = prompt_size + used_tokens

    if debug:
        print(f"Final prompt size: {prompt_size} (plus {max_new_tokens} for output).")

    # Generate responses
    generated_candidates = generate_n_responses(
        **generation_kwargs,
        max_new_tokens=max_new_tokens,
        final_prompt_template=final_prompt_template,
        prompt_variables_dict=prompt_variables_dict,
        context=prompt_size + max_new_tokens,
        ollama_port=ollama_port
    )
    if debug:
        # print(f"\n\n{Style.BRIGHT}=== Sample: {index+1} ===")
        print(f"{Fore.CYAN}{Style.BRIGHT} User prompt: {Style.RESET_ALL}\n{prompt}\n")
        for i, cand in enumerate(generated_candidates):
            print(f"{Fore.YELLOW}{Style.BRIGHT} Assistant response: #{i+1}:\n{cand}\n")
        print(f"{Fore.GREEN}{Style.BRIGHT} True response:{Style.RESET_ALL}\n {true_response}\n")

    return generated_candidates, prompt_size
   
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
    ollama_port="11434",
    rag_data: RagData = None,
    max_ctx=16000,
    constrained_output=False
)-> tuple[dict[int, list[str]], int]:
    """Run a model on a list of tasks and return the generated code snippets."""

    if "phi" in model and max_ctx > 16000:
        raise ValueError("Max context for Phi model is 16000 tokens")
    elif max_ctx > 130000:
        raise ValueError("Max context in our setup is 130000 tokens")
    
    constrained_llm = None
    if constrained_output:
        print("Constrained output is set to True.")
        model_kwargs = {
            "seed": seed,
            "max_length": max_ctx,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "quantize": True, # Sets to torch.float16
            # "pad_token_id": 0,
            # "eos_token_id": 1,
        }
        # Load the Syncode augmented model with huggingface model
        if "phi4" in model:
            # hf_model = "microsoft/phi-4"
            # hf_model = "meta-llama/Llama-3.2-3B"
            hf_model = "microsoft/phi-2"
        elif "llama" in model:
            hf_model = "meta-llama/Llama-3.3-70B-Instruct"
        else:
            raise ValueError("Constrained output is only available for Phi4 model.")
        constrained_llm = Syncode(
            model=hf_model, 
            grammar=f"{project_root}/data/midio_grammar.lark", 
            parse_output_only=True, 
            **model_kwargs
        )

    results: dict[int, list[str]] = {}
    largest_ctx_size = 0
    for index, sample in enumerate(data):

        generated_candidates, prompt_size = run_prompt_step(
            response_type="CODE",
            sample=sample,
            example_pool=example_pool,
            prompt_type=prompt_type,
            available_nodes=available_nodes,
            client=client,
            model=model,
            max_ctx=max_ctx,
            max_new_tokens=max_new_tokens,
            generation_kwargs={
                "client": client,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "n": n,
                "seed": seed,
                "debug": debug,
                "constrained_llm": constrained_llm
            },
            rag_data=rag_data,
            debug=debug,
            ollama_port=ollama_port,
        )
        if constrained_llm:
            torch.cuda.empty_cache()
        if prompt_size > largest_ctx_size:
            largest_ctx_size = prompt_size + max_new_tokens

        task_id = int(sample["task_id"])
        results[task_id] = generated_candidates

    return results, largest_ctx_size

def evaluate_code(
    candidate_dict: dict[int, list[str]],
    ks: list[int],                               
    evaluation_metrics:list[str],
    experiment_name: str,
    model_name: str,
    env: str,
    hyperparams: dict,
    phase: str,
    eval_method: str = "hold_out",
    fold: int = None,
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
    for metric in evaluation_metrics:
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
                    eval_method=eval_method,
                    fold=fold,
                    db_connection=db_connection
                )
                print(f"✅ Errors saved to database for {metric} in {experiment_name}")

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
        prompt_type: PromptType,
        rag_data: RagData = None,
        max_ctx: int = 60000
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
                    prompt_type,
                    rag_data=rag_data,
                    max_ctx=max_ctx
                )

                metric_results_lists = evaluate_code (
                    model_result,
                    ks=ks,
                    evaluation_metrics=[optimizer_metric],
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
    prompt_type: PromptType,
    two_step: bool = False,
    rag_data: RagData = None,
    max_ctx: int = 60000,

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
        if two_step:
            model_result, largest_context = two_step_run(
                client,
                model=model["name"],
                available_nodes=available_nodes,
                data=test_data,
                example_pool=example_pool,
                node_max_new_tokens=model["node_max_tokens"],
                code_max_new_tokens=model["max_tokens"],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                n=n,
                seed=seed,
                debug=debug,
                prompt_type=prompt_type,
                ollama_port="11434",
                rag_data=rag_data,
                max_ctx=max_ctx
            )
        
        else:
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
                prompt_type,
                rag_data=rag_data,
                max_ctx=max_ctx
            )

        metric_results_lists = evaluate_code (
            model_result,
            ks=ks,
            evaluation_metrics=metrics,
            experiment_name=experiment_name,
            model_name=model["name"],
            env=env,
            hyperparams={"seed": seed, "temperature": temperature, "top_p": top_p, "top_k": top_k},
            phase="testing"
        )
        print(f"Testing with seed={seed}, Gave pass@ks={metric_results_lists}")

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

def calculate_final_result(
        testing_runs: list[Run], 
        only_mean: bool = False
    ) -> Run:
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

    if only_mean:
        # Only return mean values as integers, instead of dict of mean and std
        for metric, passes in final_metric_results.items():
            for pass_k, stats in passes.items():
                final_metric_results[metric][pass_k] = stats["mean"]

    # To get the unique seeds:
    try:
        seeds = {seed for run in testing_runs for seed in run.seed}
    except TypeError:
        print("Seeds is not 2d list")
        seeds = {run.seed for run in testing_runs}

    # Return a new Run object for the final result
    return Run(
        phase="final",
        temperature=testing_runs[0].temperature,
        top_p=testing_runs[0].top_p,
        top_k=testing_runs[0].top_k,
        seed = list(seeds),
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

def calculate_best_params(
        validation_runs: list[Run],
        metrics: list[str],
        k: int
) -> Run:
    """Calculate the best run from the validation runs"""
    pass_k_metric = f"pass@k_{metrics[0]}"
    best_metric_score = 0.0
    best_run = Run("validation", 0.2, 0.6, 10, {pass_k_metric: {f"pass@{k}": 0.0}}, 9, {})
    for run in validation_runs:
        
        if len(run.metric_results) != 1:
            raise ValueError(f"Expected one metric result, got {len(run.metric_results)}")
        
        pass_at_k_dict = run.metric_results[pass_k_metric]
        metric_score = pass_at_k_dict[f"pass@{k}"] # Can only optimize for one metric

        if metric_score > best_metric_score:
            # print(f"New best pass@{ks[0]} found: {val_metric}")
            best_metric_score = metric_score
            best_run = Run(
                phase="validation",
                temperature=run.temperature,
                top_p=run.top_p,
                top_k=run.top_k,
                metric_results={pass_k_metric: pass_at_k_dict}, # Can only optimize for one metric
                seed=run.seed,
                metadata=run.metadata,
            )
    return best_run