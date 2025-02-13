from my_packages.evaluation.metrics import check_correctness, check_semantics, check_syntax, estimate_pass_at_k
from my_packages.evaluation.midio_compiler import compile_code, get_json_test_result, is_all_tests_passed, is_code_semantically_valid, is_code_syntax_valid
from my_packages.utils.file_utils import read_test_code_file, read_code_file, write_code_file

def test_check_correctness():
    """Test the correctness of the check_correctness function, using one test file."""
    task_id = 1
    true_file = read_test_code_file(task_id)
    test_result = check_correctness({task_id: [true_file]})
    print(test_result)
    assert test_result.get(task_id)[0]['passed'] == True

def test_dataset_files():
    """Test that all dataset files are valid"""
    all_passed = True
    for i in range(50):
        code = read_test_code_file(i+1)
        test_result = compile_code(code, "test", "--json")
        if not is_code_syntax_valid(test_result):
            print("SYNTAX ERROR")
            all_passed = False
            break
        elif not is_code_semantically_valid(test_result):
            print("SEMANTIC ERROR")
            all_passed = False
            break
        else: 
            test_result = get_json_test_result(test_result) 
        
            if not is_all_tests_passed(test_result):
                all_passed = False
                break
    assert all_passed