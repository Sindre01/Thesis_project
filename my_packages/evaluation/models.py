
def invoke_anthropic_model(client, full_prompt, model, max_new_tokens=50, temperature=0.7, top_p=0.9):
    """Invoke the anthropic model using the anthropic client sdk."""
    kwargs = {
        'model': model,
        'messages': full_prompt,
        'max_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'stream': False,
        #'stop': ["```<|eot_id|>"],  # Ensure the response stops after the code block
        #'seed': seed
    }
    generated = client.create(**kwargs)
    
    filtered_generated = generated.content[0].text.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated

def invoke_ollama_model(client, full_prompt, model, max_new_tokens, temperature, top_p, seed= None):
    """Invoke an opensource model using the ollama client sdk."""
    kwargs = {
        'model': model,
        'messages': full_prompt,
        'stream': False,
        'options': {
            'temperature': temperature,
            'top_p': top_p,
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

def invoke_openai_model(client, full_prompt, model, max_new_tokens, temperature, top_p, seed=None):
    """Invoke the openai model using the openai client sdk."""
    kwargs = {
        'model': model,
        'messages': full_prompt,
        'max_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
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
        'stream': False,
        'stop': ["```<|eot_id|>"],  # Ensure the response stops after the code block
    }
    generated = client.chat.completions.create(**kwargs)
    
    filtered_generated = generated.choices[0].message.content.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated