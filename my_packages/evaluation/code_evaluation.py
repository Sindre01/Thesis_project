from my_packages.evaluation.midio_compiler import compile_code, is_code_syntax_valid, is_code_semantically_valid, print_compiled_output
from my_packages.utils.server_utils import server_diagnostics
from my_packages.evaluation.models import invoke_anthropic_model, invoke_openai_model, invoke_o1_model, invoke_ollama_model
import re
from colorama import Fore, Back, Style


def extract_code(response_text):
    """Extract code snippet from the response using regex."""
    # Match content between ```language and ```
    match = re.search(fr"```midio(.*?)```", response_text, re.DOTALL)

    # Extract and clean up the code
    if match:
        return match.group(1).strip()  # Return only the code block

    # If no match, assume the response might already be code without markdown formatting
    return response_text.strip()

def evaluate_code_success_rate(client, messages, model, prompts, responses, seed, max_new_tokens=50, temperature=0.7, top_p=0.9, debug=False):
    correct_syntax = 0 # Number of samples that is syntactically correct
    correct_semantic = 0 # Number of samples that is both syntactically and semantically correct
    # all_tests_passed = 0 # Number of samples that passed all tests
    total = len(prompts)

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
        
        generated_code = extract_code(generated)
        compiled = compile_code(generated_code)

        if debug:
            print(f"\n\n{Style.BRIGHT}Sample: {index}")
            print(f"{Fore.CYAN}{Style.BRIGHT} User prompt: {Style.RESET_ALL}\n{prompt}\n")
            print(f"{Fore.YELLOW}{Style.BRIGHT} Assistant response:{Style.RESET_ALL}\n{generated}\n")
            print(f" >{Fore.BLUE}{Style.BRIGHT} Extracted code:{Style.RESET_ALL}\n {generated_code}\n")
            print(f"{Fore.GREEN}{Style.BRIGHT} True response:{Style.RESET_ALL}\n {true_response}\n")
            print(f" >{Fore.BLUE}{Style.BRIGHT} Extracted code:{Style.RESET_ALL}\n {extract_code(true_response)}\n")
            print_compiled_output(compiled)

        # Check if the generated code is syntactically correct, aka it compiles
        if is_code_syntax_valid(compiled):
            correct_syntax += 1
            # Then, check if the generated code is semantically correct
            if is_code_semantically_valid(compiled):
                correct_semantic += 1

    return (correct_syntax / total, correct_semantic / total)

def run_code_evaluation(client, messages, model, val_prompts, val_responses, test_prompts, test_responses, temperatures=[0.5, 0.7, 0.9], top_ps=[0.2, 0.5, 1.0], seeds=[3, 75, 346], debug=False):
    results = []

    SEED = 42 # During Validation Phase for reproducibility

    best_syntax_success_rate = 0.0

    best_semantic_success_rate = 0.0
    best_params = {"temperature": 0.5, "top_p": 0.5}

    print(f"{Fore.CYAN}{Style.BRIGHT}Validation Phase:{Style.RESET_ALL}")
    for temp in temperatures:
        for top_p in top_ps:
            print(f"Validating with temperature: {temp}, and top_p: {top_p}")
            syntax_success_rate, semantic_success_rate = evaluate_code_success_rate (
                client,
                messages,
                model['name'],
                val_prompts,
                val_responses,
                SEED,
                model["max_tokens"],
                temp,
                top_p,
                debug
            )
            print(f"Tested with temp={temp} and top_p={top_p}. Gave semantic_success_rate={semantic_success_rate}")
            #Prioritize semantic success_rate over syntax success_rate
            if semantic_success_rate > best_semantic_success_rate:
                best_semantic_success_rate = semantic_success_rate
                best_syntax_success_rate = syntax_success_rate
                best_params = {"temperature": temp, "top_p": top_p}
            #If semantic success_rate is the same, prioritize syntax success_rate
            elif semantic_success_rate == best_semantic_success_rate and syntax_success_rate > best_syntax_success_rate:
                best_semantic_success_rate = semantic_success_rate
                best_syntax_success_rate = syntax_success_rate
                best_params = {"temperature": temp, "top_p": top_p}
            
    print(f"Best Hyperparameters for {model['name']}: {best_params}, Validation Syntax success rate: {best_syntax_success_rate:.2f}, Validation Semantic success rate: {best_semantic_success_rate:.2f}")

    #Test the model on the test setm with different seeds and the best hyperparameters.
    print(f"TESTING Phase")
    for seed in seeds:
        print(f"\nTesting with Seed: {seed}", end="\r")
        best_temperature = best_params["temperature"]
        best_top_p = best_params["top_p"]
        test_syntax_success_rate, test_semantic_success_rate = evaluate_code_success_rate(
            client,
            messages,
            model['name'],
            test_prompts,
            test_responses,
            seed,
            model["max_tokens"],
            best_temperature,
            best_top_p,
            debug
        )

        print(f"Test Syntax success rate for {model['name']}: {test_syntax_success_rate:.2f}")
        print(f"Test Semantic success rate for {model['name']}: {test_semantic_success_rate:.2f}")

        new_run = {
            "seed": seed,
            "validation_syntax_success_rate": best_syntax_success_rate,
            "validation_semantic_success_rate": best_semantic_success_rate,
            "test_syntax_success_rate": test_syntax_success_rate,
            "test_semantic_success_rate": test_semantic_success_rate ,
            "temperature": best_temperature,
            "top_p": best_top_p,
        }
        results.append(new_run)

        if debug:
            print_success_rate_result(new_run)

    return results
    
def print_success_rate_result(run: dict):
    """Print the success rate results of a run."""
    seed = run["seed"]
    val_syn = run["validation_syntax_success_rate"]
    val_sem = run["validation_semantic_success_rate"]
    test_syn = run["test_syntax_success_rate"]
    test_sem = run["test_semantic_success_rate"]
    best_temperature = run["temperature"]
    best_top_p = run["top_p"]
    print(
        f"  Seed {seed}: "
        f"Validation Syntax success rate: {val_syn:.2f}, "
        f"Validation Semantic success rate: {val_sem:.2f}, "
        f"Test Syntax success rate: {test_syn:.2f}, "
        f"{Fore.GREEN}{Style.BRIGHT}Test Semantic success rate: {test_sem:.2f} {Style.RESET_ALL}, "
        f"Best temperature: {best_temperature:.2f},"
        f"Best top_p: {best_top_p:.2f},"
    )
        
