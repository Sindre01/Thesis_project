
import re
from syncode import Syncode
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from my_packages.utils.server_utils import server_diagnostics
from transformers import set_seed
import torch

def extract_response(response_text: str, last_snippet = False) -> str:
    """
    Extracts a code snippet from the response using a regex for ```midio code blocks.
    If multiple code blocks are found, returns:
      - The last one if <think> is present and </think> is not
      - The first one otherwise
    Also removes any comments—whether they are on lines by themselves or inline.
    """
    # Split the response to get content after the last </think>
    parts = response_text.rsplit('</think>', 1)
    code_section = parts[-1]  # Content after the last </think>

    # Find all `midio` code blocks
    matches = re.findall(r'```mid\w*(.*?)(```|$)', code_section, re.DOTALL)

    if matches:
    
        if last_snippet or ("<think>" in response_text and "</think>" not in response_text):
            code_block = matches[-1][0] # Get last code block from thinking phase.
        else:
            code_block = matches[0][0]  # Get first code block from non-thinking phase.
    else:
        code_block = code_section

    # Remove inline and full-line comments
    code_without_comments = re.sub(r'(?://|#).*$', '', code_block, flags=re.MULTILINE)

    return code_without_comments.strip()

def invoke_anthropic_model(client, full_prompt, model, max_new_tokens, temperature, top_p, top_k,):
    """Invoke the anthropic model using the anthropic client sdk."""
    kwargs = {
        'model': model,
        'messages': full_prompt,
        'max_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'stream': False,
        #'stop': ["```<|eot_id|>"],  # Ensure the response stops after the code block
        #'seed': seed
    }
    generated = client.create(**kwargs)
    
    filtered_generated = generated.content[0].text.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated

def invoke_ollama_model(client, full_prompt, model, max_new_tokens, temperature, top_p, top_k, seed= None):
    """Invoke an opensource model using the ollama client sdk."""
    kwargs = {
        'model': model,
        'messages': full_prompt,
        'stream': False,
        'options': {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'num_predict': max_new_tokens,
            'num_ctx': 10000,
            'stop': ["```<|eot_id|>"]  # Ensure the response stops after the code block
        }
    }
    if seed:
        kwargs['options']['seed'] = seed

    generated = client.chat(**kwargs)

    filtered_generated = generated['message']['content'].replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated

def invoke_openai_model(client, full_prompt, model, max_new_tokens, temperature, top_p, top_k, seed=None):
    """Invoke the openai model using the openai client sdk."""
    kwargs = {
        'model': model,
        'messages': full_prompt,
        'max_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k,': top_k,
        'stream': False,
        'stop': ["```<|eot_id|>"],  # Ensure the response stops after the code block
    }
    if seed:
        kwargs['seed'] = seed
    generated = client.chat.completions.create(**kwargs)
    
    filtered_generated = generated.choices[0].message.content.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated

def invoke_o1_model(client, full_prompt, model, max_new_tokens): 
    """Invoke the o1 model using the openai client sdk. **This model does not support seed, temperature, or top_p."""
    kwargs = {
        'model': model,
        'messages': full_prompt,
        'max_tokens': max_new_tokens,
        # 'temperature': temperature,
        # 'top_p': top_p,
        # 'top_k,': top_k,
        'stream': False,
        'stop': ["```<|eot_id|>"],  # Ensure the response stops after the code block
    }
    generated = client.chat.completions.create(**kwargs)
    
    filtered_generated = generated.choices[0].message.content.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated


def generate_n_responses(
    n:int, # Number of generations per task
    client: ChatOllama | ChatOpenAI,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed:int,   
    final_prompt_template: ChatPromptTemplate,       
    prompt_variables_dict: dict,      
    context: int,
    debug: bool=False,              
    ollama_port:str="11434",
    constrained_llm: Syncode = None,
    extract_last_snippet: bool = False,
    syncode_debug: bool = False,
)-> list[str]:
    """Generate n responses for a given prompt."""
    
    generated_candidates = []

    for current_n in range(n):
        
        new_seed = seed * (current_n+1) if seed else None # different seed for each attempt if not None
        print(f"generation seed: {new_seed}")
        generated = ""

        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                print(f"    > Generating n response..  ({current_n + 1}/{n})", end="\r")

                if constrained_llm:
                    print("Using Syncode")
                    generated = generate_syncode_reponse(
                        client=constrained_llm,
                        final_prompt_template=final_prompt_template,
                        prompt_variables_dict=prompt_variables_dict,
                        seed=new_seed,
                        debug=syncode_debug
                    )
                else:
                    # print("Using Langchain")
                    generated = generate_langchain_response(
                        client=client,
                        model=model,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=new_seed,
                        final_prompt_template=final_prompt_template,
                        prompt_variables_dict=prompt_variables_dict,
                        context=context,
                        ollama_port=ollama_port,
                    )
                    if debug:
                        print("Generated langchain unfiltered response:", generated)
                break  # If generation succeeded, break out of retry loop

            except torch.OutOfMemoryError as e:
                print("⚠️ Caught OOM during generation")
                torch.cuda.empty_cache()
                continue

            except Exception as e:
                retries += 1
         
                # print("Running command: 'gc.collect()' and 'torch.cuda.empty_cache()' ")
                # import gc
                # gc.collect()
                # torch.cuda.empty_cache()
                if constrained_llm:
                    print(e)
                    print("Syncode generation failed. Setting generated to errror.")
                    generated = e
                    break
                    
                print(f"Attempt {retries} failed with error: {e}")
                server_diagnostics(host=f"http://localhost:{ollama_port}")
                if retries < max_retries:
                    print(f"Retrying for {retries}. time...")

        if retries == max_retries:
            print("Failed to get a response from the server after " + str(retries) + " attempts.")
            generated = "Failed to get a response from the server after " + str(retries) + " attempts."
            raise Exception("Failed to generate a response.")
        
        if not generated:
            raise Exception("Failed to generate a response.")
        #Extract code from the generated response
        generated_code = extract_response(generated, last_snippet=extract_last_snippet)
        generated_candidates.append(generated_code)

    if not generated_candidates:
        generated_candidates = [""] * n

    return generated_candidates

def generate_single_response(
    n:int, # cueent n of generations per task
    client: ChatOllama | ChatOpenAI,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed:int,   
    final_prompt_template: ChatPromptTemplate,       
    prompt_variables_dict: dict,      
    context: int,
    debug: bool=False,              
    ollama_port:str="11434",
    constrained_llm: Syncode = None,
    extract_last_snippet: bool = False,
    syncode_debug: bool = False,
)-> list[str]:
    """Generate n responses for a given prompt."""
    
    generated_candidates = []

    new_seed = seed * (n+1) if seed else None # different seed for each attempt if not None
    print(f"generation seed: {new_seed}")
    generated = ""

    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            # print(f"    > Generating n response..  ({(n+1)}/{n})", end="\r")

            if constrained_llm:
                print("Using Syncode")
                generated = generate_syncode_reponse(
                    client=constrained_llm,
                    final_prompt_template=final_prompt_template,
                    prompt_variables_dict=prompt_variables_dict,
                    seed=new_seed,
                    debug=syncode_debug
                )
            else:
                # print("Using Langchain")
                generated = generate_langchain_response(
                    client=client,
                    model=model,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=new_seed,
                    final_prompt_template=final_prompt_template,
                    prompt_variables_dict=prompt_variables_dict,
                    context=context,
                    ollama_port=ollama_port,
                )
                if debug:
                    print("Generated langchain unfiltered response:", generated)
            break  # If generation succeeded, break out of retry loop

        except torch.OutOfMemoryError as e:
            print("⚠️ Caught OOM during generation")
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            retries += 1
        
            # print("Running command: 'gc.collect()' and 'torch.cuda.empty_cache()' ")
            # import gc
            # gc.collect()
            # torch.cuda.empty_cache()
            if constrained_llm:
                print(e)
                print("Syncode generation failed. Setting generated to errror.")
                generated = e
                break
                
            print(f"Attempt {retries} failed with error: {e}")
            server_diagnostics(host=f"http://localhost:{ollama_port}")
            if retries < max_retries:
                print(f"Retrying for {retries}. time...")

    if retries == max_retries:
        print("Failed to get a response from the server after " + str(retries) + " attempts.")
        generated = "Failed to get a response from the server after " + str(retries) + " attempts."
        raise Exception("Failed to generate a response.")
    
    if not generated:
        raise Exception("Failed to generate a response.")
    #Extract code from the generated response
    generated_code = extract_response(generated, last_snippet=extract_last_snippet)
    generated_candidates.append(generated_code)

    if not generated_candidates:
        generated_candidates = [""] * n

    return generated_candidates

def generate_syncode_reponse(
    client: Syncode,
    final_prompt_template: ChatPromptTemplate,
    prompt_variables_dict: dict,
    seed: int,
    debug: bool = False,
)-> str:
    """Generate a response for a given prompt using the Syncode model."""
    hf_role_mappings = {
        "system": "system",
        "human": "user",
        "ai": "assistant"
    }

    set_seed(seed)
    langchain_messages = final_prompt_template.format_messages(**prompt_variables_dict)

    # Convert to Hugging Face-style chat format
    hf_messages = [{"role": hf_role_mappings[msg.type], "content": msg.content} for msg in langchain_messages]
    # print("SynCode input with hf_messages:", hf_messages)
    output = client.infer(hf_messages, stop_words=["}\n\n```\n"], debug=debug)
    print("SynCode output:", output[0])
    # print("* MARKERER SLUTT PÅ OUTPUT. SynCode output length:", len(output[0]))
    response = output[0]
    return response
    
def generate_langchain_response(
    client: ChatOllama | ChatOpenAI,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    final_prompt_template: ChatPromptTemplate,
    prompt_variables_dict: dict,
    context: int,
    ollama_port: str,
 )-> str:
    """Generate a response for a given prompt using langchain."""
    if "gpt" in model:
        print("GPT MODEL")
        llm = client(
            model=model,
            temperature=temperature,
            seed=seed,
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
            top_k=top_k if top_k != -1 else None,
            stream=False,
            num_ctx=context,
            stop=["```<|eot_id|>"],
            seed=seed,
            base_url=f"http://localhost:{ollama_port}"
        )
    

    chain = (final_prompt_template | llm)

    response = chain.invoke(
        prompt_variables_dict,
        {"run_name": f"Few-shot code prediction"}
    )
    generated = response.content
    return generated