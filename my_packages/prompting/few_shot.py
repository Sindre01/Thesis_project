import json
import os
import re
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from my_packages.prompting.example_selectors import CoverageExampleSelector

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

# def split_and_format(data: json, template_name: str, code_folder = None):
#     script_path = os.path.dirname(os.getcwd())
#     prompt_template_path = os.path.join(script_path, f'../templates/prompts/{template_name}.file')
#     response_template_path = os.path.join(script_path, f'../templates/responses/{template_name}.file')

#     if not (os.path.exists(prompt_template_path)):
#         print(f"{prompt_template_path}.file not found!!")
#         return
#     if not (os.path.exists(response_template_path)):
#         print(f"{response_template_path}.file not found!!")
#         return
    
#     prompt_template = read_file(prompt_template_path)
#     response_template = read_file(response_template_path)

#     prompts = [prompt_template.format(task_description=task['prompts'][0]) for task in data]

#     if code_folder: #Code generation
#         responses = []
#         for task in data:
#             file_path = f"{code_folder}/task_id_{task['task_id']}.midio"
#             try:
#                 with open(file_path, 'r') as file:
#                     solution_code = response_template.format(code=file.read().strip())
#                     responses.append(solution_code)
#             except FileNotFoundError:
#                 responses.append("File not found")
#             except Exception as e:
#                 responses.append(f"Error: {e}")
#     else: #Node generation
#         responses = [response_template.format(nodes=(", ".join(task['external_functions']))) for task in data]
#     return (prompts, responses)

def get_system_template(context_template_name):
    script_path = os.path.dirname(os.getcwd())
    context_template_path = os.path.join(script_path, f'../templates/contexts/{context_template_name}.file')
    
    if not (os.path.exists(context_template_path)):
        print(f"{context_template_name}.file not found!!")
        return
    return read_file(context_template_path)


def get_response_template(template_name):
    script_path = os.path.dirname(os.getcwd())
    template_path = os.path.join(script_path, f'../templates/responses/{template_name}.file')
    
    if not (os.path.exists(template_path)):
        print(f"{template_name}.file not found!!")
        return
    return read_file(template_path)

# def create_few_shot_messages(explained_used_libraries, train_prompts, train_responses, context_template_name, context_role= "developer"):
#     context = create_context(explained_used_libraries, context_template_name)

#     few_shots_messages = [{"role": context_role, "content": context}] 
#     for i, (prompt, response) in enumerate(zip(train_prompts, train_responses)):
#         few_shots_messages.append({"role": "user", "content": prompt})
#         few_shots_messages.append({"role": "assistant", "content": response})
#     return few_shots_messages

# def create_few_shot_string(explained_used_libraries, train_prompts, train_responses, context_template_name, context_role= "developer"):
#     context = create_context(explained_used_libraries, context_template_name)

#     few_shots_messages = f"{context_role}: {context}\n"
#     for i, (prompt, response) in enumerate(zip(train_prompts, train_responses)):
#         few_shots_messages+= f"\n\n\nTask {i}\n"
#         few_shots_messages+= f"User prompt: {prompt}\n"
#         few_shots_messages+= f"Assistant response: {response}\n"

#     return few_shots_messages



##############LANGCHAIN################
def get_prompt_template(template_name):
    script_path = os.path.dirname(os.getcwd())
    template_path = os.path.join(script_path, f'../templates/prompts/{template_name}.file')
    
    if not (os.path.exists(template_path)):
        print(f"{template_name}.file not found!!")
        return
    return read_file(template_path)

def get_semantic_similarity_example_selector(example_pool, embedding, shots, input_keys):
    return SemanticSimilarityExampleSelector.from_examples(
        # The list of examples available to select from.
        example_pool,
        # The embedding class used to produce embeddings which are used to measure semantic similarity.
        embedding,
        # The VectorStore class that is used to store the embeddings and do a similarity search over.
        FAISS,
        # The number of examples to produce.
        k=shots,
        input_keys=input_keys
    )
def get_coverage_example_selector(example_pool: list[dict], embedding, shots):
    return CoverageExampleSelector.from_examples(
        # The list of examples available to select from.
        example_pool,
        # The embedding class used to produce embeddings which are used to measure semantic similarity.
        embedding,
        # The VectorStore class that is used to store the embeddings and do a similarity search over.
        FAISS,
        # The number of examples to produce.
        k=shots,
    )

def create_few_shot_prompt(
        examples: list[dict], 
        template_name: str
    ) -> FewShotChatMessagePromptTemplate:
    
    prompt_template = get_prompt_template(template_name)
    response_template = get_response_template(template_name)

    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", prompt_template),
            ("ai", response_template)
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt_template, #Formats each individual example
    )
    return few_shot_prompt

def create_final_node_prompt(
        few_shot_prompt: FewShotChatMessagePromptTemplate, 
        system_template_name: str, 
        prompt_template_name: str
    ) -> ChatPromptTemplate:

    system_msg = get_system_template(system_template_name)
    prompt_template = get_prompt_template(prompt_template_name)
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            few_shot_prompt,
            ("human", prompt_template)
        ]
    )
    return final_prompt

def get_prompt_template_variables(template_name):
    # Extract variables inside { } or {{ }}
    template = get_prompt_template(template_name)
    variables = re.findall(r'\{\{(.*?)\}\}|\{(.*?)\}', template)

    # Flatten and remove None values
    variable_names = [var for group in variables for var in group if var]
    return variable_names

def transform_node_data(data):
    """[{'task_id': str, 'task': str, 'response': list, 'MBPP_task_id': str}]"""
    new_data_format = []
    for sample in data:
        new_obj = {}
        new_obj['MBPP_task_id'] = str(sample['MBPP_task_id'])
        new_obj['task_id'] = str(sample['task_id'])
        new_obj['task'] = sample['prompts'][0]
        external_functions = [func.replace("root.std.", "") for func in sample['external_functions']]
        new_obj['response'] = ', '.join(external_functions)
        new_data_format.append(new_obj)
    return new_data_format

def transform_code_data(data):
    new_data_format = []
    for task in data:
        new_obj = {}
        new_obj['task_id'] = task['task_id']
        new_obj['task'] = task['prompts'][0]
        new_obj['response'] = task['code']
        # new_obj['tests'] = get_tests(new_obj['task_id'])
        new_obj['python_tests'] = task['testing']['tests_list']
        new_data_format.append(new_obj)
    return 