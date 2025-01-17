import json
import os
import tiktoken # type: ignore

def find_max_tokens_code_tokenizer(input_folder, tokenizer):
    """Uses the provided tokenizer to find the maximum number of tokens in a folder of .midio files"""
    max_tokens = 0
    files = 0

    # Loop through all .midio files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".midio"):
            file_path = os.path.join(input_folder, filename)

            # Read the .midio file content
            with open(file_path, "r") as f:
                content = f.read()
            # Tokenize the content and count tokens
            tokens = tokenizer.encode(content)
            token_count = len(tokens)

            print(f"File: {filename} | Tokens: {token_count}")

            # Check if this file has the most tokens so far
            if token_count > max_tokens:
                max_tokens = token_count
            files+=1
    print(f"Processed {files} files in total.")
    return max_tokens

def find_max_tokens_code_client(input_folder, model, embedder):
    """Uses the OpenAI client to find the maximum number of tokens in a folder of .midio files"""
    max_tokens = 0
    files = 0

    # Loop through all .midio files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".midio"):
            file_path = os.path.join(input_folder, filename)

            # Read the .midio file content
            with open(file_path, "r") as f:
                content = f.read()
            # Tokenize the content and count tokens
            tokens = embedder.create(model= model, input= content).usage.total_tokens

            print(f"File: {filename} | Tokens: {tokens}")

            # Check if this file has the most tokens so far
            if tokens > max_tokens:
                max_tokens = tokens
            files+=1
    print(f"Processed {files} files in total.")
    return max_tokens

def find_max_tokens_nodes_client(list, model, embedder):
    """Uses the OpenAI client to find the maximum number of tokens in a list of strings."""
    max_tokens = 0
   
    for content in list:
        # Tokenize the content and count tokens
        tokens = embedder.create(model= model, input= content).usage.total_tokens

        # Check if this file has the most tokens so far
        if tokens > max_tokens:
            max_tokens = tokens
    return max_tokens

def find_max_tokens_nodes_tokenizer(list, tokenizer):
    """Uses the provided tokenizer to find the maximum number of tokens in a list of strings."""
    max_tokens = 0
   
    for content in list:
        # Tokenize the content and count tokens
        tokens = tokenizer.encode(content)
        token_count = len(tokens)
        
        # Check if this file has the most tokens so far
        if token_count > max_tokens:
            max_tokens = token_count
    return max_tokens
    

def write_models_code_tokens_to_file(client, models, DATA_DIR, write_to_file = "code_max_tokens.json"):
    """
    Calculates tokens for each model based on DATADIR and writes to a file.
    - If model name contains 'gpt' or 'o1', it uses the tiktoken tokenizer to calculate the tokens.
    - Else, it uses the client sdk to get embeddings for model and calculate the tokens.
    """
    # Ensure the file exists; if not, create an empty file
    if not os.path.exists(write_to_file):
        with open(write_to_file, "w") as f:
            json.dump({}, f)  # Create an empty JSON list

    # Read the existing models
    try:
        with open(write_to_file, "r") as f:
            existing_models = json.load(f)
    except json.JSONDecodeError:
        # Handle the case where the file exists but is not valid JSON
        existing_models = []

    updated_models = existing_models

    for model_name in models:
        print(f"Model: {model_name}")

        if "gpt" in model_name or "o1" in model_name: # Use the tokenizer to calculate tokens for openai models
            encoding = tiktoken.encoding_for_model(model_name)
            max_tokens = find_max_tokens_code_tokenizer(DATA_DIR, encoding)
        else: # Use the client sdk for ollama model
            max_tokens = find_max_tokens_code_client(DATA_DIR, model_name, client.embeddings)
        updated_models[model_name] = {'name': model_name, 'max_tokens': max_tokens}
    
    with open(write_to_file, "w") as f:
        json.dump(updated_models, f)

def write_models_nodes_tokens_to_file(client, models, strings, write_to_file = "nodes_max_tokens.json"):
    """Calculates tokens for each model based on a list of strings and writes to a file."""
    """- If model name contains 'gpt' or 'o1', it uses the tiktoken tokenizer to calculate the tokens."""
    """- Else, it uses the client sdk to get embeddings for model and calculate the tokens."""
    # Ensure the file exists; if not, create an empty file
    if not os.path.exists(write_to_file):
        with open(write_to_file, "w") as f:
            json.dump({}, f)  # Create an empty JSON list

    # Read the existing models
    try:
        with open(write_to_file, "r") as f:
            existing_models = json.load(f)
    except json.JSONDecodeError:
        # Handle the case where the file exists but is not valid JSON
        existing_models = []

    updated_models = existing_models

    for model_name in models:
        print(f"Model: {model_name}")

        if "gpt" in model_name or "o1" in model_name: # Use the tokenizer to calculate tokens for openai models
            encoding = tiktoken.encoding_for_model(model_name)
            max_tokens = find_max_tokens_nodes_tokenizer(strings, encoding)

        else: # Use the client sdk for ollama model
            max_tokens = find_max_tokens_nodes_client(strings, model_name, client.embeddings)
        updated_models[model_name] = {'name': model_name, 'max_tokens': max_tokens}
    
    with open(write_to_file, "w") as f:
        json.dump(updated_models, f)
        

def get_model_code_tokens_from_file(model, file_path) -> int:
    """Reads the file and returns the number of tokens for the given model."""
    with open(file_path, "r") as f:
        existing_models = json.load(f)
    if model not in existing_models:
        raise KeyError(f"Model {model} not found in the file.")
    model_info = existing_models[model]
    return model_info

def get_model_node_tokens_from_file(model, file_path) -> int:
    """Reads the file and returns the number of tokens for the given model."""
    with open(file_path, "r") as f:
        existing_models = json.load(f)
    if model not in existing_models:
        raise KeyError(f"Model {model} not found in the file.")
    model_info = existing_models[model]
    return model_info

def models_not_in_file(models, file_path):
    """Returns a list of models that are not in the file."""
    with open(file_path, "r") as f:
        existing_models = json.load(f)
    return [model for model in models if model not in existing_models]