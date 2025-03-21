import os
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from my_packages.common.classes import PromptType
from my_packages.common.rag import RagData
from my_packages.utils.file_utils import read_code_file, read_file
from my_packages.utils.tokens_utils import find_max_tokens_tokenizer, fit_docs_by_tokens, measure_prompt_tokens
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

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

    context_template_path = os.path.join(project_root, f'templates/contexts/{context_template_name}.file')
    
    if not (os.path.exists(context_template_path)):
        print(f"{context_template_name}.file not found!!")
        return
    return read_file(context_template_path)


def get_response_template(template_name):

    template_path = os.path.join(project_root, f'templates/responses/{template_name}.file')
    
    if not (os.path.exists(template_path)):
        print(f"{template_name}.file not found!!")
        return
    return read_file(template_path)

def get_prompt_template(template_name):
    template_path = os.path.join(project_root, f'templates/prompts/{template_name}.file')
    
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
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt_template, #Formats each individual example
    )
    return few_shot_prompt
def create_final_prompt(
        few_shot_prompt: FewShotChatMessagePromptTemplate, 
        system_template_name: str, 
        prompt_template_name: str,
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



def transform_code_data(data: list[dict])-> list[dict]:
    """[{
        'task_id': str, 
        'task': str, 
        'response': str, 
        'MBPP_task_id': str,
        'external_functions': list,
        'tests': list,
        'signature': str,
        'preconditions': str,
        'postconditions': str,
        'flow_description': str
        }]"""
    new_data_format = []
    for sample in data:
        new_obj = {}
        new_obj['MBPP_task_id'] = str(sample['MBPP_task_id'])
        new_obj['task_id'] = str(sample['task_id'])
        new_obj['task'] = sample['prompts'][0]
        new_obj['response'] = read_code_file(sample['task_id'])
        new_obj['external_functions'] = ', '.join([func.replace("root.std.", "") for func in sample['external_functions']])
        new_obj['python_tests'] = sample['testing']['python_tests']
        new_obj['tests'] = sample['testing']['tests']
        new_obj['function_signature'] = sample['specification']['function_signature']
        new_obj['preconditions'] = sample['specification']['preconditions']
        new_obj['postconditions'] = sample['specification']['postconditions']
        new_obj['flow_description'] = sample['flow_description']
        new_data_format.append(new_obj)
    return new_data_format


def code_data_to_node_data(data: list[dict]) -> list[dict]:
    """
    Returns a new list of dicts identical to `data` except that
    each dict's "response" field is replaced with the value of "external_functions".
    """
    new_data_format = []
    for sample in data:
        # Make a shallow copy so we don't overwrite the original
        new_sample = sample.copy()
        
        # Replace 'response' with whatever is in 'external_functions'
        new_sample["response"] = sample["external_functions"]
        
        new_data_format.append(new_sample)

    return new_data_format
def add_RAG_to_prompt(
    client: ChatOllama | ChatOpenAI | ChatAnthropic,
    model: str,
    task: str,
    available_ctx: int, 
    rag_data: RagData, 
    final_prompt_template: ChatPromptTemplate,
    prompt_variables_dict: dict,
    candidate_nodes: list
)-> tuple[str, ChatPromptTemplate, dict, int]:
    
    """Add RAG context to the prompt."""

    TOTAL_DOCS_TOKENS = 40000
    TOTAL_LANG_DOCS_TOKENS = 2650
    rag_template = HumanMessagePromptTemplate.from_template(get_prompt_template("RAG"))
    
    if available_ctx > TOTAL_DOCS_TOKENS:
        # Use all data
        formatted_language_context = rag_data.formatted_language_context
        formatted_node_context = rag_data.formatted_node_context
    else:    
        # Use RAG
        print("Total available tokens after prompt: ", available_ctx)
        # Split token budget
        # max_lang_tokens = int(available_ctx * 0.5)
        # print("allocating ", max_lang_tokens, " tokens to language context")

        # formatted_language_context, used_tokens = fit_docs_by_tokens(
        #     client,
        #     rag_data.language_retriever.similarity_search(task, k=10),
        #     available_ctx=max_lang_tokens,
        #     model=model
        # )
        formatted_language_context = rag_data.formatted_language_context
        used_lang_tokens = TOTAL_LANG_DOCS_TOKENS
        print(f"Used {used_lang_tokens} tokens for language context\n")

        max_node_tokens = available_ctx - (used_lang_tokens)
        print("allocating remaining", max_node_tokens, " tokens to node context")
        avg_doc_tokens = 150

        if candidate_nodes:
            MAX_DOCS_PER_NODE = 2
            # Use predicted/fetched nodes for extracted relevant docuemtnation
            print("Using predictd nodes to extract relevant docuemntation.")
            num_nodes = len(candidate_nodes)
            estimated_k = int(available_ctx / avg_doc_tokens)
            print(f"Have availbale context to extract k = {estimated_k} documents.")
            possible_docs_per_node = int(estimated_k / num_nodes)
            if possible_docs_per_node > MAX_DOCS_PER_NODE:
                docs_per_node = MAX_DOCS_PER_NODE
            else:
                docs_per_node = possible_docs_per_node
            print(f"Extracting {docs_per_node} documents per node.")
            docs = []
            for node in candidate_nodes:
                node_docs = rag_data.node_retriever.similarity_search(node, k=docs_per_node)
                docs.extend(node_docs)
                print(f"Node {node} extracted these docs")
                for doc in node_docs:
                    print(doc.page_title)

            formatted_node_context, used_node_tokens = fit_docs_by_tokens(
                client,
                docs,
                available_ctx=max_node_tokens,
                model=model
            )

        else:
            # Estimate number of NODE documents to extract
            estimated_k = int(available_ctx/ avg_doc_tokens)
            print(f"Estimated to extract k = {estimated_k} documents.")
            docs = rag_data.node_retriever.similarity_search(task, k=estimated_k) # init too many docs, to later reduce to fit context
            formatted_node_context, used_node_tokens = fit_docs_by_tokens(
                client,
                docs,
                available_ctx=max_node_tokens,
                model=model
            )

    prompt_variables_dict.update({
        "language_context": formatted_language_context,
        "node_context": formatted_node_context,
    })

    final_prompt_template.messages.insert(1, rag_template) # Betweem system message and few-shots
    used_tokens = used_lang_tokens + used_node_tokens

    final_rag_prompt = final_prompt_template.format(**prompt_variables_dict)
    return final_rag_prompt, final_prompt_template, prompt_variables_dict, used_tokens


def build_prompt(
    response_type: str,  # CODE or NODE
    prompt_type: PromptType,
    few_shot_examples: list[dict],
    sample: dict[str, str],
    available_nodes: list[str],

)-> tuple[str, ChatPromptTemplate, dict[str, str]]:
    """Create a prompt for the model based on the prompt type."""  
    print(f"Building {response_type} prompt..")

    task = sample["task"]

    if prompt_type.value is PromptType.SIGNATURE.value and prompt_type == "CODE": # Uses signature prompt
        few_shot = create_few_shot_prompt(few_shot_examples, f'{response_type}_SIGNATURE_TEMPLATE')
        final_prompt_template = create_final_prompt(few_shot, f"{response_type}_GENERATOR_TEMPLATE", f"{response_type}_SIGNATURE_TEMPLATE")

        prompt_variables_dict = {
            "external_functions": available_nodes,
            "task": task, 
            "function_signature": sample["function_signature"],
        }
    elif prompt_type.value is PromptType.REGULAR.value or (prompt_type == "NODE"): # Uses regular prompt
        few_shot = create_few_shot_prompt(few_shot_examples, f'{response_type}_TEMPLATE')
        final_prompt_template = create_final_prompt(few_shot, f"{response_type}_GENERATOR_TEMPLATE", f"{response_type}_TEMPLATE")
        prompt_variables_dict ={
            "external_functions": available_nodes,
            "task": task, 
        }
    # elif prompt_type.value is PromptType.COT.value: # Uses COT prompt
    #     few_shot = create_few_shot_prompt(few_shot_examples, 'COT_TEMPLATE')
    #     final_prompt_template = create_final_prompt(few_shot, f"{response_type}_GENERATOR_TEMPLATE", "COT_TEMPLATE")
    #     prompt_variables_dict = {
    #         "external_functions": available_nodes,
    #         "task": task, 
    #         "function_signature": sample["function_signature"],
    #     }
    else:
        raise ValueError("Invalid prompt type. Choose from 'regular', 'signature', 'signature' or 'cot'")
        
    prompt = final_prompt_template.format(**prompt_variables_dict)
    return prompt, final_prompt_template, prompt_variables_dict


