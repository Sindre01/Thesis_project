import itertools
import os
import numpy as np
from my_packages.data_processing.code_files import extract_tests_module
from my_packages.evaluation.midio_compiler import clean_output, compile_code, extract_errors, get_errors, get_json_test_result, get_output, get_test_result, is_all_tests_passed, is_code_semantically_valid, is_code_syntax_valid
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
    print(result)
    return result


 
def check_correctness(
        result_dict: dict[int, list[str]],

    ) -> dict[int, list[dict[str, bool | str]]]:
    """
    Returns a dictionary of the results of the correctness (all unit tests passed) for each candidate code.

    E.g:
    [
        {<task_id>, {"passed": True, "result": "Some details about the execution"}},
        {<task_id>, {"passed": False, "result": "Some details about the execution"}},
        {<task_id>, {"passed": True, "result": "Some details about the execution"}},
    ]
    """
    results: dict[int, list[dict[str, str]]] = {}
    for task_id, candidates in result_dict.items():
        print(f"Checking correctness for task {task_id}...")
        test_code = get_test_module_from_file(task_id)
        checked_canidates = []
        for i, candidate in enumerate(candidates):
            print(f"> Compiling code with tests for candidate {i+1}...")
            # Add the testing code to the candidate code
            test_candidate = candidate + "\n" + test_code
            compiled = compile_code(test_candidate)
            #check if the code syntax and semantics are valid for new test_candidate
            if not is_code_syntax_valid(compiled):
                print("     syntax error found")
                checked_canidates.append({"passed": False, "info": clean_output(get_errors(compiled))})
            elif not is_code_semantically_valid(compiled):
                print("     semantics error found")
                checked_canidates.append({"passed": False, "info": extract_errors(get_output(compiled))})
            else: # If the code is semantically valid, run the tests
                print("     Running tests...")
                test_result = compile_code(test_candidate, "test", "--json")
                json_test_result = get_json_test_result(test_result)
                print("     json_test_result: ")
                print(json_test_result)

                if is_all_tests_passed(json_test_result):
                    print("     All tests passed for this code:\n")
                    print(test_candidate)
                    print("\n\n")
                    checked_canidates.append({"passed": True, "info": "All tests passed"})
                else:
                    print("     Some tests failed")
                    checked_canidates.append({"passed": False, "info": get_test_result(json_test_result)})
            
        results[task_id] = checked_canidates

    return results

def check_syntax(
        result_dict: dict[int, list[str]],
    ):
    """
    Returns a dictionary of the results of the syntax check for each candidate code.

    E.g:
    [
        (<task_id>, {"passed": True, "result": "Some details about the execution"}),
        (<task_id>, {"passed": False, "result": "Some details about the execution"}),
        (<task_id>, {"passed": True, "result": "Some details about the execution"}),
    ]
    """
    results: dict = {}
    for task_id, candidates in result_dict.items():
        print(f"Checking correctness for task {task_id}...")
        checked_canidates = []
        for i, candidate in enumerate(candidates):
            print(f"> Compiling code for candidate {i+1}...")
            compiled = compile_code(candidate)
            if is_code_syntax_valid(compiled):
                checked_canidates.append({"passed": True, "info": "Syntax is correct"})
            else:
                checked_canidates.append({"passed": False, "info": get_errors(compiled)})
        results[task_id] = checked_canidates
    return results

def check_semantics(result_dict):
    """
    Returns a dictionary of the results of the semantic check for each candidate code.

    E.g:
    [
        (<task_id>, {"passed": True, "result": "Some details about the execution"}),
        (<task_id>, {"passed": False, "result": "Some details about the execution"}),
        (<task_id>, {"passed": True, "result": "Some details about the execution"}),
    ]
    """
    results = {}
    for task_id, candidates in result_dict.items():
        print(f"> Checking correctness for task {task_id}...")
        checked_canidates = []
        for i, candidate in enumerate(candidates):
            print(f"    Compiling code for candidate {i+1}...")
            compiled = compile_code(candidate)
            if is_code_semantically_valid(compiled):
                checked_canidates.append({"passed": True, "info": "semantics is correct"})
            else:
                checked_canidates.append({"passed": False, "info": get_output(compiled)})
        results[task_id] = checked_canidates
    return results
