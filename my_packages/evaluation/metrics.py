import itertools
import numpy as np
from my_packages.evaluation.midio_compiler import compile_code, get_errors, get_output, get_test_result, is_all_tests_passed, is_code_semantically_valid, is_code_syntax_valid

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


def check_correctness(result_dict):
    """
    Returns a dictionary of the results of the correctness (all unit tests passed) for each candidate code.

    e.g. {key: [result1, result2, result3, ...], key2: [result1, result2, result3, ...]}
    where results are dictionaries with the keys "passed": true/false and "info".
    """
    results = {}
    for key, candidates in result_dict.items():
        checked_canidates = []
        for candidate in candidates:
            compiled = compile_code(candidate)
            #check if the code syntax is valid and semantically valid
            if not is_code_syntax_valid(compiled):
                checked_canidates.append({"passed": False, "info": get_errors(compiled)})
            elif not is_code_semantically_valid(compiled):
                checked_canidates.append({"passed": False, "info": get_output(compiled)})
            else:
                # If the code is semantically valid, run the tests
                test_result_json = compile_code(candidate, "test --json")

                if is_all_tests_passed(test_result_json):
                    checked_canidates.append({"passed": True, "info": "All tests passed"})
                else:
                    checked_canidates.append({"passed": False, "info": get_test_result(test_result_json)})
            
        results[key] = checked_canidates

    return results

def check_syntax(result_dict):
    """
    Returns a dictionary of the results of the syntax check for each candidate code.

    e.g. {key: [result1, result2, result3, ...], key2: [result1, result2, result3, ...]}
    where results are dictionaries with the keys "passed": true/false and "info".
    """
    results = {}
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
