from my_packages.evaluation.code_evaluation import calculate_pass_at_k_scores
from my_packages.evaluation.metrics import check_correctness, check_semantics, check_syntax
from my_packages.utils.file_utils import read_test_code_file, read_code_file


def test_check_correctness():
    """Tes the check_correctness function, using one test file."""
    task_id = 1
    true_file = read_test_code_file(task_id)
    test_result = check_correctness({task_id: [true_file]})
    print(test_result)
    assert test_result.get(task_id)[0]['passed'] == True

def test_check_syntax():
    """Test the check_syntax function, using one test file."""
    task_id = 1
    true_file = read_test_code_file(task_id)
    test_result = check_syntax({task_id: [true_file]})
    print(test_result)
    assert test_result.get(task_id)[0]['passed'] == True

def test_check_semantics():
    """Test the check_semantics function, using one test file."""
    task_id = 1
    true_file = read_test_code_file(task_id)
    test_result = check_semantics({task_id: [true_file]})
    print(test_result)
    assert test_result.get(task_id)[0]['passed'] == True


def test_pass_at_k_functional_correctness():
    """Test the pass_at_k function for functional correctness with unit tests, on all code files."""
    results = {}
    for i in range(50):
        task_id = i+1
        results[task_id] = [read_code_file(task_id)]

    pass_at_ks = calculate_pass_at_k_scores(results, ks=[1], metric="tests")
    assert pass_at_ks['pass@1'] == 1.0

def test_pass_at_k_syntax():
    """Test the pass_at_k function for syntax correctness, on all code files."""
    results = {}
    for i in range(50):
        task_id = i+1
        results[task_id] = [read_code_file(task_id)]

    pass_at_ks = calculate_pass_at_k_scores(results, ks=[1], metric="syntax")
    assert pass_at_ks['pass@1'] == 1.0

def test_pass_at_k_semantic():
    """Test the pass_at_k function for semantic correctness, on all code files."""
    results = {}
    for i in range(50):
        task_id = i+1
        results[task_id] = [read_code_file(task_id)]

    pass_at_ks = calculate_pass_at_k_scores(results, ks=[1], metric="semantic")
    assert pass_at_ks['pass@1'] == 1.0
