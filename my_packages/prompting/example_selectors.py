import numpy as np
from langchain_core.example_selectors.base import BaseExampleSelector
import random
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

def get_semantic_similarity_example_selector(example_pool, embedding, shots, input_keys):
    return SemanticSimilarityExampleSelector.from_examples(
        examples=example_pool,  # List[dict] containing your examples
        embeddings=embedding,   # An initialized Embeddings instance (e.g., OpenAIEmbeddings())
        vectorstore_cls=FAISS,  # FAISS as your vector store backend
        k=shots,                # Number of examples to select (the "k" shots)
        input_keys=input_keys,  # Keys to select for conversion to text
    )

def get_coverage_example_selector(example_pool: list[dict], label: str, shots: int, seed: int):
    return CoverageExampleSelector(
        # The list of examples available to select from.
        example_pool,
        label,
        shots,
        # The number of examples to produce.
        seed=seed
    )

class CoverageExampleSelector(BaseExampleSelector):
    """Selects examples based on coverage of the input space, caching the selection for reuse."""
    
    def __init__(self, examples, label, k, seed=None):
        """
        Args:
            examples (list): A list of examples.
            label (str): Key to extract outputs from an example.
            k (int): The desired number of examples to select.
            seed (int, optional): A random seed for tie-breaking. Defaults to 42 if not provided.
        """
        self.examples = examples
        self.label = label
        self.k = k
        self.seed = seed if seed is not None else 42
        self._selected_cache = None  # Cached selection result.
        self._dirty = True           # Indicates if the cache needs updating.

    def add_example(self, example):
        """Adds an example to the selector and marks the cache as dirty."""
        self.examples.append(example)
        self._dirty = True

    def _get_outputs(self, example):
        """
        Extracts the outputs from an example.
        Returns a set of unique outputs.
        """
        result = example[self.label]
        return set(result)

    def _compute_selection(self, k):
        """
        Computes k examples that together maximize the number of unique outputs.
        This implements the greedy selection algorithm.
        """
        selected = []
        covered = set()  # Set of outputs that have already been covered.
        candidates = self.examples.copy()  # Work on a copy of the examples list.
        
        # Use a random number generator with the provided seed for reproducible tie-breaking.
        rng = random.Random(self.seed)
        rng.shuffle(candidates)
        
        # Greedy selection: add the candidate that brings in the largest number of new outputs.
  
        while len(selected) < k and len(candidates) > 0:
            best_new_coverage = -1
            best_candidates = []
            for ex in candidates:
                outputs = self._get_outputs(ex)
                new_coverage = len(outputs - covered)
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_candidates = [ex]
                elif new_coverage == best_new_coverage:
                    best_candidates.append(ex)
                    
            # If no candidate adds any new outputs, break out of the loop.
            if best_new_coverage <= 0:
                break
            
            # Randomly choose one among the best candidates.
            chosen = rng.choice(best_candidates)
            selected.append(chosen)
            covered.update(self._get_outputs(chosen))
            candidates.remove(chosen)
        
        # If we haven't reached k examples, fill in with arbitrary candidates.
        if len(selected) < k:
            remaining = k - len(selected)
            selected.extend(candidates[:remaining])
        
        return selected

    def select_examples(self, fake_input=dict|None, k=None):
        """
        Returns k selected examples.
        If the underlying examples haven't changed, returns the cached selection.
        
        Args:
            fake_input (dict, optional): A fake input to match the SemanticSimilaritySelector args.
            k (int, optional): Number of examples to select. If not provided, self.k is used.
        
        Returns:
            list: A list of selected examples.
        """
        if k is None:
            k = self.k
        
        # Check if the selection has already been computed and is still valid.
        if not self._dirty and self._selected_cache is not None and k == self.k:
            # print("Returnes cached selection")
            return self._selected_cache
        
        # Compute and cache the selection.
        # print("Computes selection")
        self._selected_cache = self._compute_selection(k)
        self._dirty = False  # Reset the dirty flag after recomputation.
        return self._selected_cache
