import os
import pytest
from my_packages.evaluation.metrics import check_correctness, check_semantics, check_syntax, estimate_pass_at_k

DATA_DIR = "../data/MBPP_Midio_50/includes_tests"
def read_test_file(file: str) -> str:
    test_path = os.getcwd()
    test_file = test_path + f'{DATA_DIR}/{file}'
    with open(test_file, "r") as file:
        return file.read()
    
def test_check_correctness():
    true_file = read_test_file("task_id_1_tests.midio")
    test_result = check_correctness({1: [true_file]})

    assert test_result[1]['passed'] == True
