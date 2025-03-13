import json
import logging
import subprocess
import os
import tempfile
import re
import psutil

# Function to load code from a file
def load_code_from_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()
def run_compiler_with_timeout_quiet(command, timeout=10, max_output_chars=20000):
    """
    Run a subprocess quietly with timeout and return a CompletedProcess object.
    stdout/stderr are captured internally.
    """

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        # Kill child processes
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True

    # Truncate output to avoid log spam
    if max_output_chars is not None:
        stdout = stdout[:max_output_chars]
        stderr = stderr[:max_output_chars]
        if len(stdout) == max_output_chars:
            print(f"Truncated output to {max_output_chars} chars.")

            with open("truncated.log", "w") as f:
                f.write(stdout)
            
    if timed_out:
        stderr += "\n[ERROR][TIMEOUT] Process timed out after {} seconds.".format(timeout)
        stdout += "\n[ERROR][TIMEOUT] Process timed out after {} seconds.".format(timeout)
    stderr = stderr.strip()
    stdout = stdout.strip()
    return subprocess.CompletedProcess(
        args=command,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr
    )

def is_compile_ready(code: str) -> bool:
    """
    Check if the code is testable by checking if the first word contains Midio specifics.
    To avoid compiling dangerous code and save execution time.
    """
    node_modules = [ #Not critical if not updated. Code wil fail either way if it starts with thes. Just for error messages to be applied.
        "Url", 
        "Std", 
        "Http", 
        "Strings", 
        "Time", 
        "Testing", 
        "Data", 
        "Json", 
        "CSV", 
        "List", 
        "Map",
        "Iteration", 
        "Math", 
        "LinearAlgebra", 
        "Logic",
        "Scheduling",
        "Net",
        "Image",
        "File",
        "Env",
        "Buffer",
        "Sets",
        "Process",
        "Base64",
        "Hashing"
    ]
    correct_starts = ["import", "func", "module"]
    other_keywords = ["instance", "data_instance", "getter", "setter", "in", "out"]

    first_word = code.split()[0] if code.strip() else ""

    if any(kw in first_word for kw in (correct_starts+node_modules+other_keywords)):
        return True
    return False

# Function to check if the code compiles using package-manager
def compile_code(code: str, type: str = "build", flag: str = "") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        code_file_path = os.path.join(tmp_dir, "main.midio")
        code = code.lstrip().lstrip("\n")   # Remove leading/trailing whitespace and newlines
        warnings = []
        if code.startswith("func"):  # Remove leading whitespace and check for "func"
            # print("starts with func keyword!")
            code = "import(\"std\", Std_k98ojb)\n import(\"http\", Http_q7o96c)\nmodule() main { " + code + " }"
            warnings.append("CUSTOM WARNING: Orignal code starts with 'func' keyword, but added imports and modules manually\n")
        
        if code.startswith("module"):  # Remove leading whitespace and check for "module"
            # print("starts with module keyword!")
            code = "import(\"std\", Std_k98ojb)\n import(\"http\", Http_q7o96c)\n" + code
            warnings.append("CUSTOM WARNING: Orignal code starts with 'module' keyword, but added imports manually\n")

        # Write generated code to the main.midio file
        with open(code_file_path, "w") as temp_file:
            temp_file.write(code)

        # Write midio.json to the temp directory
        midio_json_content = '''{
            "name": "midio_example",
            "version": "0.1.0",
            "main": "main.midio",
            "include_files": [
            "main.midio"
            ],
            "dependencies": {}
        }'''
        with open(os.path.join(tmp_dir, "midio.json"), "w") as midio_json_file:
            midio_json_file.write(midio_json_content)

        # Write midio.lock.json to the temp directory
        midio_lock_json_content = '''{
        "packages": []
        }'''
        with open(os.path.join(tmp_dir, "midio.lock.json"), "w") as midio_lock_json_file:
            midio_lock_json_file.write(midio_lock_json_content)

        try:
            # Run package-manager build on the temporary directory
            if is_compile_ready(code):
                if flag:
                    commands =["package-manager", type, flag, tmp_dir]
                else:
                    commands = ["package-manager", type, tmp_dir]

                result = run_compiler_with_timeout_quiet(commands, timeout=10)
                
                # with open("temp_stdout.log", "w") as f:
                #     f.write(result.stdout)
                # with open("temp_stderr.log", "w") as f:
                #     f.write(result.stderr)
                # with open("temp_code.log", "w") as f:
                #     f.write(code)
                # print("Code is compiled")
                result.stdout += "\n".join(warnings)
            
            else:
                # print("Code is NOT compile ready for Midio")
                
                result = subprocess.CompletedProcess(
                    args = [], 
                    returncode=1, # 1 means compile build error
                    stdout="CUSTOM WARNING: Code that is not compile ready for Midio", 
                    stderr="CUSTOM WARNING: Code is not compile ready for Midio"
                )
            return result
        except FileNotFoundError:
            print("Error: package-manager not found in PATH.")
            return False


def get_errors(result: subprocess.CompletedProcess[str]) -> str:
    # Syntax Errors can be found here
    return result.stderr

def get_output(result: subprocess.CompletedProcess[str]) -> str:
    # Semantic Erros can be found here
    return result.stdout

def get_json_test_result(result: subprocess.CompletedProcess[str]) -> dict:
    output = clean_output(result.stdout)
    
    # Find the first occurrence of '{' (start of JSON)
    json_start = output.find('{')
    
    if json_start == -1:
        # print("Error: No JSON found in stdout")
        return {}

    json_text = output[json_start:]  # Extract only the JSON part

    try:
        json_result = json.loads(json_text)
        return json_result
    except json.JSONDecodeError as e:
        # print("Failed to parse JSON:", e)
        # print("Extracted JSON part:", json_text)
        return {}
    
def clean_output(text: str) -> str:
    """Remove ANSI escape codes and extra whitespace from the output."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text).strip()


def extract_errors(text: str) -> str:
    """Extracts lines that contain 'error' and returns a JSON list for structured storage."""
    if not text:
        return json.dumps([])  # Return empty list as JSON

    clean_text = clean_output(text)
    errors = [line.strip() for line in clean_text.split("\n") if "error" in line.lower()]
    
    return errors # Store as JSON string for structured retrieval

def get_test_result(json_result: dict) -> str:
    num_passed = json_result['num_passed']
    num_tests = json_result['num_tests']
    assertions = json_result['test_results']
    return f"{num_passed}/{num_tests} test passed. All tests: {assertions}"

def is_all_tests_passed(json_result: dict) -> bool:
    if not json_result:
        return False
    num_passed = json_result['num_passed']
    num_tests = json_result['num_tests']
    return num_passed == num_tests

def is_code_syntax_valid(result: subprocess.CompletedProcess[str]) -> bool:
    return result.returncode == 0

def is_code_semantically_valid(result: subprocess.CompletedProcess[str]) -> bool:
    """
    Checks if the code is semantically valid.

    - Ensures the code compiles (`returncode == 0`).
    - If `stdout` contains "error", it's a semantic issue.
    - Logs unexpected cases for debugging.
    """
    if not is_code_syntax_valid(result):  
        # The code must be compilable for semantic validation
        if "error" not in result.stdout.lower() and "error" not in result.stderr.lower():
            logging.error(f"Code does not compile, but no 'error' found in output:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        return False

    # If it compiles, check for semantic errors
    if "error" in result.stdout.lower() or "error" in result.stderr.lower():
        return False  # Semantic error detected

    return True  # No errors found

def print_compiled_output(result: subprocess.CompletedProcess[str]):
        print(f"\n\n\n\n New Output from Midio compilation of code:")
        print(f"STDOUT (semantic errors): {result.stdout}")
        print(f"STDERR (syntactical errors): {result.stderr}")

def analyze_compiler_output(stdout: str, stderr: str) -> dict:
    """
    A simplified approach:
    1) If stderr is non-empty, we consider the code to have syntax errors.
    2) If stderr is empty, but stdout has 'ERROR' lines, we consider them semantic errors.
    """
    # Trim whitespace for a clear check
    stderr_stripped = stderr.strip()
    stdout_lines = stdout.splitlines()

    syntax_errors = []
    semantic_errors = []

    # 1) If there's ANY content in stderr, treat it as syntax error
    if stderr_stripped:
        syntax_errors.append(stderr_stripped)
    else:
        # 2) If stderr is empty, look for 'ERROR' lines in stdout for semantic issues
        for line in stdout_lines:
            if "ERROR" in line:
                semantic_errors.append(line.strip())

    return {
        "syntax_errors": syntax_errors,
        "semantic_errors": semantic_errors
    }


if __name__ == "__main__":
    # Sample data
    example_stdout = r"""Installing dependencies for midio_example@0.1.0

No external dependencies

Building package...
ERROR compiler::frontend::semantic_analysis::analyzers::instance_analyzer: 83: Failed to resolve path: Failed to resolve symbol: Logic.LessThan
ERROR compiler::frontend::semantic_analysis::analyzers::instance_analyzer: 83: Failed to resolve path: Failed to resolve symbol: Logic.GreaterThan
ERROR compiler::frontend::semantic_analysis::analyzers::instance_analyzer: 83: Failed to resolve path: Failed to resolve symbol: Logic.LessThan
ERROR compiler::frontend::compiler_pass: 1341: Model has errors:
ERROR compiler::frontend::compiler_pass: 1343: SemanticAnalysisError(@78): Unable to resolve type (root.Std_k98ojb.Logic.LessThan) for instance (less_than_4c7d3b), perhaps it has been removed., backtrace:    0: <unknown>
   1: <unknown>
   2: <unknown>
   3: <unknown>
   4: <unknown>
   5: <unknown>
   6: <unknown>
   7: <unknown>
   8: <unknown>
   9: <unknown>
  10: <unknown>
  11: <unknown>
  12: <unknown>
  13: <unknown>
  14: <unknown>
  15: <unknown>
  16: <unknown>
  17: <unknown>
  18: <unknown>
  19: <unknown>
  20: <unknown>
  21: <unknown>
  22: __libc_start_main
  23: <unknown>

ERROR compiler::frontend::compiler_pass: 1343: SemanticAnalysisError(@94): Unable to resolve type (root.Std_k98ojb.Logic.GreaterThan) for instance (greater_than_2f0e6a), perhaps it has been removed., backtrace:    0: <unknown>
   1: <unknown>
   2: <unknown>
   3: <unknown>
   4: <unknown>
   5: <unknown>
   6: <unknown>
   7: <unknown>
   8: <unknown>
   9: <unknown>
  10: <unknown>
  11: <unknown>
  12: <unknown>
  13: <unknown>
  14: <unknown>
  15: <unknown>
  16: <unknown>
  17: <unknown>
  18: <unknown>
  19: <unknown>
  20: <unknown>
  21: <unknown>
  22: __libc_start_main
  23: <unknown>

ERROR compiler::frontend::compiler_pass: 1343: SemanticAnalysisError(@163): Unable to resolve type (root.Std_k98ojb.Logic.LessThan) for instance (less_than_4c7d3b), perhaps it has been removed., backtrace:    0: <unknown>
   1: <unknown>
   2: <unknown>
   3: <unknown>
   4: <unknown>
   5: <unknown>
   6: <unknown>
   7: <unknown>
   8: <unknown>
   9: <unknown>
  10: <unknown>
  11: <unknown>
  12: <unknown>
  13: <unknown>
  14: <unknown>
  15: <unknown>
  16: <unknown>
  17: <unknown>
  18: <unknown>
  19: <unknown>
  20: <unknown>
  21: <unknown>
  22: <unknown>
  23: <unknown>
  24: <unknown>
  25: __libc_start_main
  26: <unknown>

ERROR compiler::frontend::compiler_pass: 1360: Model has errors, skipping code generation
Package built successfully!

"""
    example_stderr = r"""

"""

    result = analyze_compiler_output(example_stdout, example_stderr)
    print("Syntax Errors:")
    for err in result["syntax_errors"]:
        print("  -", err)

    print("\nSemantic Errors:")
    for err in result["semantic_errors"]:
        print("  -", err)
