
import os
import sys
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from my_packages.utils.server_utils import is_remote_server_reachable, server_diagnostics
from my_packages.utils.tokens_utils import models_not_in_file, write_models_tokens_to_file

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../..")
env_path = os.path.abspath(f"{project_dir}/.env")

sys.path.append(project_dir)
print("Script is located in:", script_dir)
print("Project is located in:", project_dir)
print("Env is located in:", env_path)

def model_configs(
        all_responses: list[str], 
        model_provider: str, 
        models: list[str] = None, 
        ollama_port = "11434"
    )->tuple[ChatOllama|ChatAnthropic|ChatAnthropic, list[str]]:  
    """
    Initializes the model provider and models to be used for the job.
    Also checks if the server is reachable.
    Also writes the models and their max output tokens to a file, accordign to out MBPP-Midio-50 Dataset.
    """
    # print(f"Loading .env file from {env_path}")
    # load_dotenv(env_path)

    match model_provider:
        case 'ollama':
            host = f'http://localhost:{ollama_port}'
            if is_remote_server_reachable(url = host + "/api/tags"):
                print("Server is reachable.")
            else:
                server_diagnostics(host=host)
                if not is_remote_server_reachable(url = host + "/api/tags"):
                    print("Ollama server is not reachable. Batch job might have finished. Try running bash script again.")

            client = ChatOllama
            if not models:
                models = [
                    #14b models:
                    "phi4:14b-fp16", #16k context length
                    # "qwen2.5:14b-instruct-fp16", #128 k

                    # #32b models:
                    # "qwq:32b-preview-fp16", #ctx: 32,768 tokens
                    # "qwen2.5-coder:32b-instruct-fp16", #32,768 tokens
        
                    # #70b models:
                    # "llama3.3:70b-instruct-fp16", #ctx: 130k
                    # "qwen2.5:72b-instruct-fp16", #ctx: 139k
                ]
            models_not_tokenized = models_not_in_file(models, f'{project_dir}/data/max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{project_dir}/data/max_tokens.json')

        case 'openai':
            openai_token = os.getenv('OPENAI_API_KEY')
            if not openai_token:
                raise Exception("OpenAI API key not found in .env file")
            client = ChatOpenAI
            models = [
                "gpt-4o",
                # "o1-preview", 
            ]
            # few_shot_messages = create_few_shot_messages(explained_used_libraries, train_prompts, train_responses, "NODE_GENERATOR_TEMPLATE", "developer")
            models_not_tokenized = models_not_in_file(models, f'{project_dir}/data/max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{project_dir}/data/max_tokens.json')
            
        case 'anthropic':
            anthropic_token = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_token:
                raise Exception("Anthropic API key not found in .env file")
            client = ChatAnthropic
            # embed_client = AnthropicEmbeddings
            models = [
                "claude-3-5-sonnet-latest"
            ]
            models_not_tokenized = models_not_in_file(models, f'{project_dir}/data/max_tokens.json')
            write_models_tokens_to_file(client, models_not_tokenized, all_responses, f'{project_dir}/data/max_tokens.json')
        case _:
            raise Exception("Model provider not supported")
    return client, models