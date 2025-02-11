import itertools
import os
import numpy as np
from my_packages.data_processing.code_files import extract_tests_module
from my_packages.evaluation.midio_compiler import clean_output, compile_code, extract_errors, get_errors, get_json_test_result, get_output, get_test_result, is_all_tests_passed, is_code_semantically_valid, is_code_syntax_valid

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

def read_test_code(task_id: int) -> str:
    """Reads the test code from the file."""
    current_path = os.path.dirname(os.getcwd())
    test_file = os.path.join(current_path, f'../data/mbpp_transformed_code_examples/includes_tests/task_id_{task_id}_tests.midio')
    test_file = os.path.abspath(test_file)     # Normalize the path to resolve '..' segments.

    with open(test_file, "r") as file:
        content = file.read()
    module_tests = extract_tests_module(content)
    if module_tests:
        print(f"Found tests module block for task {task_id}")
    else:
        print("No module tests block found")
    return module_tests
 
def check_correctness(
        result_dict: dict[int, list[str]],

    ) -> dict[int, list[dict[str, bool | str]]]:
    """
    Returns a dictionary of the results of the correctness (all unit tests passed) for each candidate code.

    e.g. {key: [result1, result2, result3, ...], key2: [result1, result2, result3, ...]}
    where results are dictionaries with the keys "passed": true/false and "info".
    """
    results: dict[int, list[dict[str, str]]] = {}
    for task_id, candidates in result_dict.items():
        test_code = read_test_code(task_id)
        checked_canidates = []
        for candidate in candidates:
            compiled = compile_code(candidate)
            #check if the code syntax is valid and semantically valid
            if not is_code_syntax_valid(compiled):
                print("syntax error")
                checked_canidates.append({"passed": False, "info": clean_output(get_errors(compiled))})
            elif not is_code_semantically_valid(compiled):
                print("semantics error")
                checked_canidates.append({"passed": False, "info": extract_errors(get_output(compiled))})
            else: # If the code is semantically valid, run the tests

                # Add the testing code to the candidate code
                test_candidate = candidate + "\n" + test_code
                print("candidate with test code: ", test_candidate)

                test_result = compile_code(test_candidate, "test", "--json")
                json_test_result = get_json_test_result(test_result)
                print(json_test_result)

                if is_all_tests_passed(json_test_result):
                    checked_canidates.append({"passed": True, "info": "All tests passed"})
                else:
                    checked_canidates.append({"passed": False, "info": get_test_result(json_test_result)})
            
        results[task_id] = checked_canidates

    return results

def check_syntax(
        result_dict: dict[int, list[str]],
    ):
    """
    Returns a dictionary of the results of the syntax check for each candidate code.

    e.g. {key: [result1, result2, result3, ...], key2: [result1, result2, result3, ...]}
    where results are dictionaries with the keys "passed": true/false and "info".
    """
    results: dict = {}
    for key, candidates in result_dict.items():
        checked_canidates = []
        for candidate in candidates:
            compiled = compile_code(candidate)
            if is_code_syntax_valid(compiled):
                checked_canidates.append({"passed": True, "info": "Syntax is correct"})
            else:
                checked_canidates.append({"passed": False, "info": get_errors(compiled)})
        results[key] = checked_canidates
    return results

def check_semantics(result_dict):
    """
    Returns a dictionary of the results of the semantic check for each candidate code.

    e.g. {key: [result1, result2, result3, ...], key2: [result1, result2, result3, ...]}
    where results are dictionaries with the keys "passed": true/false and "info".
    """
    results = {}
    for key, candidates in result_dict.items():
        checked_canidates = []
        for candidate in candidates:
            compiled = compile_code(candidate)
            if is_code_semantically_valid(compiled):
                checked_canidates.append({"passed": True, "info": "semantics is correct"})
            else:
                checked_canidates.append({"passed": False, "info": get_output(compiled)})
        results[key] = checked_canidates
    return results
