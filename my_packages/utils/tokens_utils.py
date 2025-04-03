import json
import os
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import tiktoken # type: ignore

def find_max_tokens_client(responses: list[str], model, client):
    """Uses the langchain model client to find the maximum number of tokens in a folder of .midio files"""
    max_tokens = 0

    all_tokens = [
        client(model=model, num_ctx=10000).get_num_tokens(response)
        for response in responses
    ]
    # Tokenize the content and count tokens
    max_tokens = max(all_tokens)

    return max_tokens

def find_max_tokens_tokenizer(responses: list[str], tokenizer):
    """Uses the provided tokenizer to find the maximum number of tokens in a list of strings."""
    max_tokens = 0
   
    for content in responses:
        # Tokenize the content and count tokens
        tokens = tokenizer.encode(content)
        token_count = len(tokens)
        
        # Check if this file has the most tokens so far
        if token_count > max_tokens:
            max_tokens = token_count
    return max_tokens
    

def write_models_tokens_to_file(client, models, responses, write_to_file):
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
            max_tokens = find_max_tokens_tokenizer(responses, encoding)
        else: # Use the client sdk for ollama model
            max_tokens = find_max_tokens_client(responses, model_name, client)
        print(f"Max tokens for {model_name}: {max_tokens}")
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

def measure_prompt_tokens(
    client, model: str, prompt: str, max_ctx: int
) -> int:
    """
    Return the number of tokens used by `prompt`.
    For GPT models, uses tiktoken.
    Otherwise, uses the client's built-in .get_num_tokens().
    """
    if "gpt" in model.lower():
        encoding = tiktoken.encoding_for_model(model)
        # A helper function you wrote that sums tokens in a list of strings
        return find_max_tokens_tokenizer([prompt], encoding)
    else:
        # Ollama or Anthropic style
        return client(model=model, num_ctx=max_ctx).get_num_tokens(prompt)
    

def fit_docs_by_tokens(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    docs: list[Document], 
    available_ctx: int,
    model: str,
) -> tuple[str, int]:
    """
    Select a subset of documents that fit within the available context tokens.
    Supports both tiktoken and ollama models own tokenizer.
    """
    kwargs = {"client": client, "model": model, "max_ctx": available_ctx}


    selected_docs = docs[:]
    total_tokens = sum(measure_prompt_tokens(prompt=doc.page_content, **kwargs) for doc in selected_docs)
    print(f"Initial docs contain {total_tokens} tokens of availbale {available_ctx} tokens")

    # Prune if over budget
    while total_tokens > available_ctx and selected_docs:
        removed_doc = selected_docs.pop()
        total_tokens -= measure_prompt_tokens(prompt=removed_doc.page_content, **kwargs)
        # print(f"Removed document, new token count: {total_tokens}")

    # Try adding more docs while staying under the token limit
    for doc in docs[len(selected_docs):]:
        doc_tokens = measure_prompt_tokens(prompt=doc.page_content, **kwargs)
        # print(f"Document token count: {doc_tokens}")
        if total_tokens + doc_tokens <= available_ctx:
            selected_docs.append(doc)
            total_tokens += doc_tokens
            # print(f"Added document, new token count: {total_tokens}")
        else:
            # print("Not room for more documents")
            break  # No more room
    print(f"Removed {len(docs) - len(selected_docs)} documents to fit within {available_ctx} tokens. From {len(docs)} docs to {len(selected_docs)} docs")
    return "\n\n".join(doc.page_content for doc in selected_docs), total_tokens
