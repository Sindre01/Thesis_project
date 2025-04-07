import json
import logging
import subprocess
import os
import tempfile
import re
import psutil

syntax_error_indicators = [
    "parsing failed",
    "error during parsing",
    "tokenizing",
    "compilererror"
]

# semantic_error_indicators = [
#     "semantic analysis failed",
#     "semantic_analysis",
#     "semanticanalysiserror",
#     "failed to reify declaration path",
#     "symbol already exists",
#     "context_checker"
# ]


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
        stderr += "\n[TIMEOUT] Process timed out after {} seconds.".format(timeout)
        stdout += "\n[TIMEOUT] Process timed out after {} seconds.".format(timeout)
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
    node_modules = [ #Not critical if not updated. Code wil fail either way if it starts with these. Just for error messages to be applied.
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
    other_keywords = ["instance", "data_instance", "getter", "setter", "in", "out"] # sent to compiler, but will have syntax error because its not inside module or func

    first_word = code.split()[0] if code.strip() else ""

    if any(kw in first_word for kw in (correct_starts+node_modules+other_keywords)):
        return True
    
    return False # if not starting with any of the keywords, or if the code is empty

# Function to check if the code compiles using package-manager
def compile_code(code: str, type: str = "build", flag: str = "") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        code_file_path = os.path.join(tmp_dir, "main.midio")
        code = code.lstrip().lstrip("\n")   # Remove leading/trailing whitespace and newlines
        warnings = []
        if code.startswith("func"):  # Remove leading whitespace and check for "func"
            # print("starts with func keyword!")
            code = "import(\"std\", Std_k98ojb)\n import(\"http\", Http_q7o96c)\nmodule() main { \n" + code + "\n}"
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
    try:
        match = re.search(r'{.*}', output, re.DOTALL)
        if not match:
            return {}
        json_text = match.group(0)
        return json.loads(json_text)
    except json.JSONDecodeError as e:
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
    stderr_lower = result.stderr.lower()
    stdout_lower = result.stdout.lower()
    
    # Syntax-related errors typically include parsing/tokenizing issues
    
    return not any(indicator in stderr_lower or indicator in stdout_lower for indicator in syntax_error_indicators)

def is_code_semantically_valid(result: subprocess.CompletedProcess[str]) -> bool:
    """
    Checks if the code is semantically valid.

    - Ensures the code compiles (`returncode == 0`).
    - If `stdout` contains "error", it's a semantic issue.
    - Logs unexpected cases for debugging.
    """
    if not is_code_syntax_valid(result):  
        # The code must be syntactically valid for semantic validation
        return False
    
    #Must compile
    if result.returncode == 0:
        return False
    
    # If it compiles, check for semantic errors
    if "error" in result.stdout.lower() or "error" in result.stderr.lower():
        return False  # Semantic error detected

    return True  # No errors found

def is_code_semantically_valid(result: subprocess.CompletedProcess[str]) -> bool:
    if not is_code_syntax_valid(result):  
        # The code must be syntactically valid for semantic validation
        return False
    stderr_lower = result.stderr.lower()
    stdout_lower = result.stdout.lower()
    
    # If semantic indicators are found, it's semantically invalid


    return not any(indicator in stderr_lower or indicator in stdout_lower for indicator in semantic_error_indicators)


def print_compiled_output(result: subprocess.CompletedProcess[str]):
        print(f"\n\n\n\n New Output from Midio compilation of code:")
        print(f"STDOUT (semantic errors): {result.stdout}")
        print(f"STDERR (syntactical errors): {result.stderr}")

