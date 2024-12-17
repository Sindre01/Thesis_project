import os

def find_max_tokens(input_folder, tokenizer):

    max_tokens = 0

    # Loop through all .midio files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".midio"):
            file_path = os.path.join(input_folder, filename)

            # Read the .midio file content
            with open(file_path, "r") as f:
                content = f.read()

            # Tokenize the content and count tokens
            tokens = tokenizer(content)["input_ids"]
            token_count = len(tokens)

            print(f"File: {filename} | Tokens: {token_count}")

            # Check if this file has the most tokens so far
            if token_count > max_tokens:
                max_tokens = token_count

    return max_tokens
