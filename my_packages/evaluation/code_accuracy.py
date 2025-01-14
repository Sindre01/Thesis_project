from my_packages.evaluation.midio_compiler import compile_code, is_code_syntax_valid, is_code_semantically_valid, print_compiled_output
from my_packages.utils.server_utils import server_diagnostics
import re

def extract_code(response_text):
    """Extract code snippet from the response using regex."""
    # Match content between ```language and ```
    match = re.search(fr"```midio(.*?)```", response_text, re.DOTALL)

    # Extract and clean up the code
    if match:
        return match.group(1).strip()  # Return only the code block

    # If no match, assume the response might already be code without markdown formatting
    return response_text.strip()

def evaluate_code_accuracy(client, messages, model, prompts, responses, seed, max_new_tokens=50, temperature=0.7, top_p=0.9, debug=False):
    correct_syntax = 0 # Number of samples that is syntactically correct
    correct_semantic = 0 # Number of samples that is both syntactically and semantically correct
    # all_tests_passed = 0 # Number of samples that passed all tests
    total = len(prompts)
    # if model.startswith("o1"): #o1 models are not working with developer or system role
    #     print("Testing GPT model")
    #     few_shot_messages = create_few_shot_prompt(train_prompts, train_responses, context_role="user")
    # else:
    #     few_shot_messages = create_few_shot_prompt(train_prompts, train_responses, context_role="developer")

    for index, (prompt, true_response) in enumerate(zip(prompts, responses)):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                print(f"Generating response for sample {index}..", end="\r")
                generated = client.chat.completions.create(
                    model=model,
                    messages=messages + [{"role": "user", "content":prompt}],
                    max_tokens=max_new_tokens,
                    seed=seed,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                    stop=["```<|eot_id|>"]  # Ensure the response stops after the code block
                )
                filtered_generated = generated.choices[0].message.content.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
                generated_code = extract_code(filtered_generated)
                break
            except Exception as e:
                retries += 1
                print(f"Attempt {retries} failed with error: {e}")
                server_diagnostics()

        else:
            print("Failed to get a response from the server after " + str(retries) + " attempts.")
            generated_code = ""
            return -1
        compiled = compile_code(generated_code)

        if debug:
            print(f"\n\nSample: {index}")
            print(f"Prompt: {prompt}")
            print(f"Generated response:\n {filtered_generated}")
            print(f"True response:\n {true_response}")
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

    best_syntax_accuracy = 0.0

    best_semantic_accuracy = 0.0
    best_params = {"temperature": 0.5, "top_p": 0.5}

    print(f"VALIDATION Phase")
    for temp in temperatures:
        for top_p in top_ps:
            print(f"Validating with temperature: {temp}, and top_p: {top_p}")
            syntax_accuracy, semantic_accuracy = evaluate_code_accuracy (
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
            print(f"Tested with temp={temp} and top_p={top_p}. Gave semantic_accuracy={semantic_accuracy}")
            #Prioritize semantic accuracy over syntax accuracy
            if semantic_accuracy > best_semantic_accuracy:
                best_semantic_accuracy = semantic_accuracy
                best_syntax_accuracy = syntax_accuracy
                best_params = {"temperature": temp, "top_p": top_p}
            #If semantic accuracy is the same, prioritize syntax accuracy
            elif semantic_accuracy == best_semantic_accuracy and syntax_accuracy > best_syntax_accuracy:
                best_semantic_accuracy = semantic_accuracy
                best_syntax_accuracy = syntax_accuracy
                best_params = {"temperature": temp, "top_p": top_p}
            
    print(f"Best Hyperparameters for {model['name']}: {best_params}, Validation Syntax Accuracy: {best_syntax_accuracy:.2f}, Validation Semantic Accuracy: {best_semantic_accuracy:.2f}")

    #Test the model on the test setm with different seeds and the best hyperparameters.
    print(f"TESTING Phase")
    for seed in seeds:
        print(f"\nTesting with Seed: {seed}", end="\r")
        best_temperature = best_params["temperature"]
        best_top_p = best_params["top_p"]
        test_syntax_accuracy, test_semantic_accuracy = evaluate_code_accuracy(
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

        print(f"Test Syntax Accuracy for {model['name']}: {test_syntax_accuracy:.2f}")
        print(f"Test Semantic Accuracy for {model['name']}: {test_semantic_accuracy:.2f}")

        
        results.append({
            "seed": seed,
            "validation_syntax_accuracy": best_syntax_accuracy,
            "validation_semantic_accuracy": best_semantic_accuracy,
            "test_syntax_accuracy": test_syntax_accuracy,
            "test_semantic_accuracy": test_semantic_accuracy,
            "temperature": best_temperature,
            "top_p": best_top_p,
        })

        if debug:
            print(
                f"  Seed {seed}: "
                f"Val Syntax Acc: {best_syntax_accuracy}, "
                f"Val Semantic Acc: {best_semantic_accuracy}, "
                f"Test Syntax Acc: {test_syntax_accuracy}, "
                f"Test Semantic Acc: {test_semantic_accuracy},"
                f"Best temperature: {best_temperature},"
                f"Best top_p: {best_top_p},"
            )
    return results
    
