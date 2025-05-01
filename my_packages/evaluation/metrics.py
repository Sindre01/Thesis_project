import itertools
import numpy as np
from my_packages.common.classes import CodeEvaluationResult
from my_packages.evaluation.midio_compiler import compile_code, is_code_semantically_valid, is_code_syntax_valid
from my_packages.evaluation.visual_metric import evaluate_visual_flow
from my_packages.utils.file_utils import get_test_module_from_file

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""
    """pass@k [6] estimating the probability that, if asked k times,the LLM will at least once give a correct answer"""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    result = np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
    # print(result)
    return result

 
def check_correctness(
        candidates_dict: dict[int, list[str]],

    ) -> dict[int, list[CodeEvaluationResult]]:
    """
    Parameter:
        - candidates_dict: A dictionary of taks_id and generated candidates. dict[int, list[str]]
            E.g:
                {
                    {<task_id>, [candidate1, candidate2, candidate3]},
                    {<task_id>, [candidate1, candidate2, candidate3]},
                    {<task_id>, [candidate1, candidate2, candidate3]},
                }
                
    Returns a dictionary of the results of the correctness (all unit tests passed) for each candidate code.
    """
    results: dict[int, list[CodeEvaluationResult]] = {}

    for task_id, candidates in candidates_dict.items():
        test_code = get_test_module_from_file(task_id)
        checked_canidates = []

        for i, candidate in enumerate(candidates):
            # print(f"    Compiling code with tests for candidate {i+1}...")
            # Add the testing code to the candidate code
            test_candidate = candidate + "\n" + test_code
            compiled = compile_code(test_candidate)
            
            evaluation_result = CodeEvaluationResult("tests", test_candidate, task_id, (i+1), compiled)

            #check if the code syntax and semantics are valid for new test_candidate
            if not is_code_syntax_valid(compiled):
                evaluation_result.add_syntax_error(compiled)
            
            elif not is_code_semantically_valid(compiled):
                evaluation_result.add_semantic_error(compiled)
            
            else: # If the code is semantically valid, check tests and extract the test results
                # print(f"    Running tests...", flush=True)
                compiled_tests = compile_code(test_candidate, "test", "--json")
                evaluation_result.add_tests_result(compiled_tests)

            checked_canidates.append(evaluation_result)

        results[task_id] = checked_canidates

    return results

def check_syntax(
        candidates_dict: dict[int, list[str]],
    ) -> dict[int, list[CodeEvaluationResult]]:
    """

    Parameter:
        - candidates_dict: A dictionary of taks_id and generated candidates. dict[int, list[str]]
            E.g:
                {
                    {<task_id>, [candidate1, candidate2, candidate3]},
                    {<task_id>, [candidate1, candidate2, candidate3]},
                    {<task_id>, [candidate1, candidate2, candidate3]},
                }
    Returns a dictionary of the results of the syntax check for each candidate code.

    """
    results: dict = {}
    for task_id, candidates in candidates_dict.items():
        # print(f"\n> Checking syntax for task {task_id}...")
        checked_canidates = []
        for i, candidate in enumerate(candidates):
            # print(f"    Compiling code for candidate {i+1}...")

            compiled = compile_code(candidate)
            evaluation_result = CodeEvaluationResult("syntax", candidate, task_id, (i+1), compiled)

            if not is_code_syntax_valid(compiled):
                evaluation_result.add_syntax_error(compiled)

            checked_canidates.append(evaluation_result)
        results[task_id] = checked_canidates
    
    return results

def check_semantics(
        candidates_dict: dict[int, list[str]],
    ) -> dict[int, list[CodeEvaluationResult]]:
    """
    Parameter:
        - candidates_dict: A dictionary of taks_id and generated candidates. dict[int, list[str]]
            E.g:
                {
                    {<task_id>, [candidate1, candidate2, candidate3]},
                    {<task_id>, [candidate1, candidate2, candidate3]},
                    {<task_id>, [candidate1, candidate2, candidate3]},
                }

    Returns a dictionary of the results of the semantic check for each candidate code.
    """
    results = {}
    for task_id, candidates in candidates_dict.items():
        # print(f"\n> Checking semantics for task {task_id}...")
        checked_canidates = []
        for i, candidate in enumerate(candidates):
            # print(f"    Compiling code for candidate {i+1}...")
            compiled = compile_code(candidate)
            evaluation_result = CodeEvaluationResult("semantic", candidate, task_id, (i+1), compiled)

            if not is_code_syntax_valid(compiled):
                evaluation_result.add_syntax_error(compiled)

            elif not is_code_semantically_valid(compiled):
                evaluation_result.add_semantic_error(compiled)

            checked_canidates.append(evaluation_result)
            
        results[task_id] = checked_canidates
    return results

# def check_visualization(
#         candidates_dict: dict[int, list[str]],
#     ) -> dict[int, list[float]]:
#     """
#     Parameter:
#         - candidates_dict: A dictionary of taks_id and generated candidates. dict[int, list[str]]
#             E.g:
#                 {
#                     {<task_id>, [candidate1, candidate2, candidate3]},
#                     {<task_id>, [candidate1, candidate2, candidate3]},
#                     {<task_id>, [candidate1, candidate2, candidate3]},
#                 }

#     Returns a dictionary of the results for each candidate code.
#     """
#     results = {}
#     for task_id, candidates in candidates_dict.items():
#         # print(f"\n> Checking visual for task {task_id}...")
#         checked_canidates = []
#         for i, candidate in enumerate(candidates):

#             evaluation_result = evaluate_visual_flow(candidate)
#             checked_canidates.append(evaluation_result)
            
#         results[task_id] = checked_canidates
#     return results

def check_visualization(
        candidates_dict: dict[int, list[str]],
    ) -> dict[int, list[float | None]]:
    """
    Compute the visual-flow score **only** for candidates that are
    syntactically and semantically valid.

    Args:
        candidates_dict: {task_id: [candidate_code, ...], ...}

    Returns:
        {task_id: [score_or_None, ...], ...}
        – The list order matches the input candidates list.
          Invalid candidates receive None.
    """
    results: dict[int, list[float | None]] = {}

    for task_id, candidates in candidates_dict.items():
        task_scores: list[float | None] = []

        for candidate in candidates:
            compiled = compile_code(candidate)

            # Skip if syntax OR semantics fail
            if (not is_code_syntax_valid(compiled)
                    or not is_code_semantically_valid(compiled)):
                # task_scores.append(None)          # or 0.0 if you prefer
                continue

            # Only evaluate visual flow for “clean” programs
            score = evaluate_visual_flow(candidate)
            task_scores.append(score)

        if task_scores:
            results[task_id] = task_scores

    return results
