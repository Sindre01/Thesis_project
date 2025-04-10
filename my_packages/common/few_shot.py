from langchain_ollama import OllamaEmbeddings
from my_packages.prompting.example_selectors import get_coverage_example_selector, get_semantic_similarity_example_selector
from langchain_core.example_selectors.base import BaseExampleSelector


def init_example_selector(
        num_shots: int, 
        example_pool: dict, 
        semantic_selector: bool,
        similarity_keys: list[str] = ["task"], # External_functions
    )-> BaseExampleSelector:

    example_pool.sort(key=lambda x: int(x['task_id']))
    print(f"Number of examples in the pool: {len(example_pool)}")

    if semantic_selector:
        selector = get_semantic_similarity_example_selector(
            example_pool, 
            OllamaEmbeddings(model="nomic-embed-text"),
            shots=num_shots,
            input_keys=similarity_keys,
        )
    else:
        selector = get_coverage_example_selector(
            example_pool, 
            label = "external_functions",
            shots=num_shots,
            seed=9
        )
    return selector
