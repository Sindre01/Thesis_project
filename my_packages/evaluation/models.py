
def invoke_anthropic_model(client, full_prompt, model, max_new_tokens=50, temperature=0.7, top_p=0.9):
    """Invoke the anthropic model using the anthropic client sdk."""
    generated = client.create(
        model=model,
        messages=full_prompt,
        max_tokens=max_new_tokens,
        # seed=seed,
        temperature=temperature,
        top_p=top_p,
        stream=False,
        # stop=["```<|eot_id|>"]  # Ensure the response stops after the code block
    )
    filtered_generated = generated.content[0].text.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated

def invoke_ollama_model(client, full_prompt, model, seed, max_new_tokens, temperature, top_p):
    """Invoke an opensource model using the ollama client sdk."""
    generated = client.chat(
        model=model,
        messages=full_prompt,
        options= {
            'temperature': temperature, #temperature,
            'top_p': top_p,
            'num_predict': max_new_tokens,
            'seed': seed,
            'num_ctx': 10000,
            'stop': ["```<|eot_id|>"] # Ensure the response stops after the code block
        },
        stream=False,
        #format = "json"

    )
    filtered_generated = generated['message']['content'].replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated

def invoke_openai_model(client, full_prompt, model, seed, max_new_tokens, temperature, top_p):
    """Invoke the openai model using the openai client sdk."""
    generated = client.chat.completions.create(
        model=model,
        messages=full_prompt,
        max_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        stream=False,
        stop=["```<|eot_id|>"]  # Ensure the response stops after the code block
    )
    filtered_generated = generated.choices[0].message.content.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated

def invoke_o1_model(client, full_prompt, model, max_new_tokens): 
    """Invoke the o1 model using the openai client sdk. **This model does not support seed, temperature, or top_p."""
    generated = client.chat.completions.create(
        model=model,
        messages=full_prompt,
        max_tokens=max_new_tokens,
        # seed=seed,
        # temperature=temperature,
        # top_p=top_p,
        stream=False,
        stop=["```<|eot_id|>"]  # Ensure the response stops after the code block
    )
    filtered_generated = generated.choices[0].message.content.replace("//", "").strip() # '//' outside main module can lead to compiler not ending properly
    return filtered_generated