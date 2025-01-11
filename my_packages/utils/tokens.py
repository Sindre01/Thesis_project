import os

def find_max_tokens_code_tokenizer(input_folder, tokenizer):

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

def find_max_tokens_code_api(input_folder, model, embedder):
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

def find_max_tokens_nodes_api(list, model, embedder):
    max_tokens = 0
   
    for content in list:
        # Tokenize the content and count tokens
        tokens = embedder.create(model= model, input= content).usage.total_tokens

        # print(f"Tokens: {tokens}")

        # Check if this file has the most tokens so far
        if tokens > max_tokens:
            max_tokens = tokens
    return max_tokens

def find_max_tokens_nodes_tokenizer(list, tokenizer):
    max_tokens = 0
   
    for content in list:
        # Tokenize the content and count tokens
        tokens = tokenizer.encode(content)
        token_count = len(tokens)
        # print(f"Tokens: {tokens}")

        # Check if this file has the most tokens so far
        if token_count > max_tokens:
            max_tokens = token_count
    return max_tokens
    
