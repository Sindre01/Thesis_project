from my_packages.utils.server_utils import server_diagnostics
import re

def extract_nodes(response_text):
    """Extract nodes from the response using regex."""
    # Match content between ```language and ```
    match = re.search(fr"```midio(.*?)```", response_text, re.DOTALL)

    # Extract and clean up the nodes
    if match:
        return match.group(1).strip()  # Return only the midio block

    # If no match, assume the response might already be nodes without markdown formatting
    return response_text.strip()
def evaluate_nodes_accuracy(client, messages, model, available_nodes, prompts, responses, seed, max_new_tokens=50, temperature=0.7, top_p=0.9, debug=False):
    total = len(prompts)
    correct = 0
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
        generated_nodes = extract_nodes(generated.choices[0].message.content)
        generated_nodes_set = set(generated_nodes.replace(",", "").split())

        # Extract nodes from the true response and transform to a set
        true_response_nodes = extract_nodes(true_response)
        true_response_set = set(true_response_nodes.replace(",", "").split())

        if debug:
            print(f"\n\nSample: {index}")
            print(f"Prompt: {prompt}")
            print(f"Generated response:\n {generated_nodes}")
            print(f"True response:\n {true_response}")

        #Remove invalid nodes from the generated nodes
        valid_generated_nodes = generated_nodes_set.intersection(available_nodes)
        
        if true_response_set.issubset(valid_generated_nodes):
            correct += 1
    
    return correct / total

def run_nodes_evaluation(client, messages, model, available_nodes, val_prompts, val_responses, test_prompts, test_responses, temperatures=[0.5, 0.7, 0.9], top_ps=[0.2, 0.5, 1.0], seeds=[3, 75, 346], debug=False):
    results = []

    SEED = 42 # During Validation Phase for reproducibility
    best_accuracy = 0.0
    best_params = {"temperature": 0.5, "top_p": 0.5}
    print(debug)
    print(f"VALIDATION Phase")
    for temp in temperatures:
        for top_p in top_ps:
            print(f"Validating with temperature: {temp}, and top_p: {top_p}")
            accuracy = evaluate_nodes_accuracy (
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
        test_accuracy = evaluate_nodes_accuracy(
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
        
        results.append({
            "seed": seed,
            "validation_accuracy": best_accuracy,
            "test_accuracy": test_accuracy,
            "temperature": best_temperature,
            "top_p": best_top_p,
        })
        if debug:
            print(
                f"  Seed {seed}: "
                f"validation_accuracy: {best_accuracy}, "
                f"test_accuracy: {test_accuracy}, "
                f"Best temperature: {best_temperature},"
                f"Best top_p: {best_top_p},"
            )
    return results