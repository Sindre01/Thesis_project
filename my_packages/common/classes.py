from datetime import datetime
import json
from subprocess import CompletedProcess
from colorama import Fore, Style
from my_packages.evaluation.midio_compiler import clean_output, extract_errors, get_json_test_result, is_all_tests_passed

class CodeEvaluationResult:
    """
    A class to represent the result of evaluating a code snippet.
    """
    def __init__(
            self, 
            metric: str, 
            code: str, 
            task_id: int, 
            candidate_id: int,
            compiler_msg: CompletedProcess[str]
        ):
        self.metric = metric
        self.code = code
        self.task_id = task_id
        self.candidate_id = candidate_id
        self.compiler_msg = compiler_msg
        self.passed = True
        self.error_type = None
        self.error_msg = None
        self.test_result = None

    def add_syntax_error(self, compiled: CompletedProcess[str]):
        # print("     Syntax error found")
        self.passed = False
        self.error_type = "syntax"
        self.error_msg = extract_errors(compiled.stdout) 
        self.compiler_msg = compiled

    def add_semantic_error(self, compiled: CompletedProcess[str]):
        # print("     Semantics error found")
        self.passed = False
        self.error_type = "semantic"
        self.error_msg = extract_errors(compiled.stdout)
        self.compiler_msg = compiled

    def add_tests_result(self, compiled_tests: CompletedProcess[str]):
        json_result = get_json_test_result(compiled_tests)
    
        if is_all_tests_passed(json_result):
            self.passed = True
        else:
            # print("     Tests error found")
            self.passed = False
            self.error_type = "tests"
            self.error_msg = extract_errors(compiled_tests.stdout)
        
        self.compiler_msg = compiled_tests
        self.test_result = get_json_test_result(compiled_tests)

    def __str__(self):
        """
        Returns a detailed string representation of the evaluation result, including:
        - Task ID and Candidate ID
        - Metric used for evaluation
        - Whether the candidate passed or failed
        - The specific type of error encountered (if any)
        - The error message (if applicable)
        - The test results (if tests were run)
        - The actual evaluated code snippet
        """

        # Determine pass/fail icon
        status_icon = "✅" if self.passed else "❌"

        result_str = f"""
===================================================================================
        {status_icon} Code Evaluation Result
===================================================================================
        📌 Task ID       : {self.task_id}
        🔢 Candidate ID  : {self.candidate_id}
        📊 Metric        : {self.metric}
        {"✔️ Passed        : Yes" if self.passed else "❌ Passed        : No"}
        """

        if self.error_type:
            error_icon = "⚠️" if self.error_type != "tests" else "🛠️"
            result_str += f"""
        {error_icon} Error Type    : {self.error_type}
        🔍 Error Message : {self.error_msg}
        ------------------------------

        
🖥️ Compiler stderr: {clean_output(self.compiler_msg.stderr) if clean_output(self.compiler_msg.stderr) else "N/A"}
📤 Compiler output: {clean_output(self.compiler_msg.stdout) if clean_output(self.compiler_msg.stdout) else "N/A"}
        """

        if self.test_result:
            result_str += f"""
------------------------------
🧪 Test Result   : {self.test_result}
        """

        result_str += f"""
------------------------------
📜 Evaluated Code:
------------------------------
{self.code}
==============================
        """
        return result_str
class Run:
    def __init__(
        self,
        phase: str,
        temperature: float,
        top_p: float,
        top_k: int,
        metric_results: dict,
        seed = None,
        metadata: dict | None = None,
        created_at: datetime = None
    ):
        self.phase = phase
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.metric_results = metric_results
        self.seed = seed
        self.metadata = metadata
        self.created_at = created_at
    
    def print(self):

        if self.phase == "validation":
            print(
                f"\n===  VALIDATION ===\n"
                f"{Fore.GREEN}{Style.BRIGHT}"
                f"  > Temperature: {self.temperature:.2f}\n"
                f"  > Top_k: {self.top_k:.2f}\n"
                f"  > Top_p: {self.top_p:.2f}\n"
                f"{Style.RESET_ALL}"
                f"  > Optimized metric result:: {json.dumps(self.metric_results, indent=4)}"
                f" > Metadata: {json.dumps(self.metadata, indent=4)}"
            )

        elif self.phase == "testing":
            print(
                f"  = TESTING SEED {self.seed} =\n"
                f"  > Temperature: {self.temperature:.2f}\n"
                f"  > Top_k: {self.top_k:.2f}\n"
                f"  > Top_p: {self.top_p:.2f}\n"
                f"{Fore.GREEN}{Style.BRIGHT}"
                f"  > Metric results: {json.dumps(self.metric_results, indent=4)}"
                f"  > Metadata: {json.dumps(self.metadata, indent=4)}"
                f"{Style.RESET_ALL}"
            )

        elif self.phase == "final":
            print(
                f"\n===  FINAL RESULTS ===\n"
                f"  > Temperature: {self.temperature:.2f}\n"
                f"  > Top_k: {self.top_k:.2f}\n"
                f"  > Top_p: {self.top_p:.2f}\n"
                f"{Fore.GREEN}{Style.BRIGHT}"
                f"  > Metric results: {json.dumps(self.metric_results, indent=4)}"
                f"  > Metadata: {json.dumps(self.metadata, indent=4)}"
                f"{Style.RESET_ALL}"
            )


# class CodeDataSchema(BaseModel):
#     task_id: str
#     task: str
#     response: str
#     MBPP_task_id: str
#     external_functions: str
#     tests: List[str] = Field(default_factory=str)
#     signature: str

from enum import Enum

class ModelProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class PromptType(Enum):
    REGULAR = "regular"
    SIGNATURE = "signature"
    COT = "cot"

# class Evaluation(Enum):
#     BEST_PARAMS = "best_params"
#     FINAL_RESULTS = "final_results"
class Phase(Enum):
    VALIDATION = "validation"
    TESTING = "testing"


def get_prompt_type(file_name: str) -> PromptType | None:
    for prompt in PromptType:
        if prompt.value in file_name:
            return prompt
    return None 