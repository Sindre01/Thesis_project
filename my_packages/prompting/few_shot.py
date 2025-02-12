import json
import os
import random
import re

import numpy as np
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)

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

##############LANGCHAIN################
def get_prompt_template(template_name):
    script_path = os.path.dirname(os.getcwd())
    template_path = os.path.join(script_path, f'../templates/prompts/{template_name}.file')
    
    if not (os.path.exists(template_path)):
        print(f"{template_name}.file not found!!")
        return
    return read_file(template_path)

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
    print("Example Prompt ", examples[0])
    print("Example Prompt Template: ", example_prompt_template)
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt_template, #Formats each individual example
    )
    return few_shot_prompt

def create_final_prompt(
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
    """[{
        'task_id': str, 
        'task': str, 
        'response': list, 
        'MBPP_task_id': str,
        'external_functions': list
        }]"""
    new_data_format = []
    for sample in data:
        new_obj = {}
        new_obj['MBPP_task_id'] = str(sample['MBPP_task_id'])
        new_obj['task_id'] = str(sample['task_id'])
        new_obj['task'] = sample['prompts'][0]
        external_functions = [func.replace("root.std.", "") for func in sample['external_functions']]
        new_obj['response'] = ', '.join(external_functions)
        new_obj['external_functions'] = ', '.join(external_functions)
        new_data_format.append(new_obj)
    return new_data_format

def transform_code_data(data):
    """[{
        'task_id': str, 
        'task': str, 
        'response': str, 
        'MBPP_task_id': str,
        'external_functions': list,
        'tests': list,
        'signature': str
        }]"""
    new_data_format = []
    for sample in data:
        new_obj = {}
        new_obj['MBPP_task_id'] = str(sample['MBPP_task_id'])
        new_obj['task_id'] = str(sample['task_id'])
        new_obj['task'] = sample['prompts'][0]
        new_obj['response'] = read_code_file(sample['task_id'])
        new_obj['external_functions'] = ', '.join([func.replace("root.std.", "") for func in sample['external_functions']])
        new_obj['tests'] = sample['testing']['tests']
        new_obj['function_signature'] = sample['specification']['function_signature']
        new_data_format.append(new_obj)
    return new_data_format

def read_code_file(task_id):
    script_path = os.path.dirname(os.getcwd())
    file_path = os.path.join(script_path, f'../data/MBPP_Midio_50/only_files/task_id_{task_id}.midio')
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error: {e}"
    
