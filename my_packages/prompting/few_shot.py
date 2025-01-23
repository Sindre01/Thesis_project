import json
import os

def read_file(_file):
    with open(_file) as reader:
        return reader.read()
def replace_third_word(text, new_word):
    words = text.split()
    if len(words) < 3:
        return text  # or raise an error if you want strict behavior
    words[2] = new_word  # Index 2 is the third word (0-based indexing)
    return " ".join(words)

def remove_first_three_words(text: str) -> str:
    words = text.split()
    if len(words) <= 4:
        # If the string has 3 or fewer words, return an empty string
        return ""
    # Slice from the 4th word onward (index 3, zero-based)
    return " ".join(words[4:])

def split_and_format(data: json, template_name: str, code_folder = None):
    script_path = os.path.dirname(os.getcwd())
    prompt_template_path = os.path.join(script_path, f'../templates/prompts/{template_name}.file')
    response_template_path = os.path.join(script_path, f'../templates/responses/{template_name}.file')

    if not (os.path.exists(prompt_template_path)):
        print(f"{prompt_template_path}.file not found!!")
        return
    if not (os.path.exists(response_template_path)):
        print(f"{response_template_path}.file not found!!")
        return
    
    prompt_template = read_file(prompt_template_path)
    response_template = read_file(response_template_path)

    prompts = [prompt_template.format(task_description=task['prompts'][0]) for task in data]

    if code_folder: #Code generation
        responses = []
        for task in data:
            file_path = f"{code_folder}/task_id_{task['task_id']}.midio"
            try:
                with open(file_path, 'r') as file:
                    solution_code = response_template.format(code=file.read().strip())
                    responses.append(solution_code)
            except FileNotFoundError:
                responses.append("File not found")
            except Exception as e:
                responses.append(f"Error: {e}")
    else: #Node generation
        responses = [response_template.format(nodes=(", ".join(task['library_functions']))) for task in data]
    return (prompts, responses)

def create_context(explained_used_libraries, context_template_name):
    script_path = os.path.dirname(os.getcwd())
    context_template_path = os.path.join(script_path, f'../templates/contexts/{context_template_name}.file')
    
    if not (os.path.exists(context_template_path)):
        print(f"{context_template_name}.file not found!!")
        return
    context_template = read_file(context_template_path)
    context = context_template.format(library_functions=explained_used_libraries)
    return context

def create_few_shot_messages(explained_used_libraries, train_prompts, train_responses, context_template_name, context_role= "developer"):
    context = create_context(explained_used_libraries, context_template_name)

    few_shots_messages = [{"role": context_role, "content": context}] 
    for i, (prompt, response) in enumerate(zip(train_prompts, train_responses)):
        few_shots_messages.append({"role": "user", "content": prompt})
        few_shots_messages.append({"role": "assistant", "content": response})
    return few_shots_messages

def create_few_shot_string(explained_used_libraries, train_prompts, train_responses, context_template_name, context_role= "developer"):
    context = create_context(explained_used_libraries, context_template_name)

    few_shots_messages = f"{context_role}: {context}\n"
    for i, (prompt, response) in enumerate(zip(train_prompts, train_responses)):
        few_shots_messages+= f"\n\n\nTask {i}\n"
        few_shots_messages+= f"User prompt: {prompt}\n"
        few_shots_messages+= f"Assistant response: {response}\n"

    return few_shots_messages