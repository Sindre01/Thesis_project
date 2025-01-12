import subprocess
import os
import tempfile
import re

# Function to load code from a file
def load_code_from_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

# Function to check if the code compiles using package-manager
def compile_code(code: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        code_file_path = os.path.join(tmp_dir, "main.midio")

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
            result = subprocess.run(
                ["package-manager", "build", tmp_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # print("Compiler stdout:", result.stdout)
            # print("Compiler stderr:", result.stderr)


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

def is_code_syntax_valid(result: subprocess.CompletedProcess[str]) -> bool:
    return result.returncode == 0

def is_code_semantically_valid(result: subprocess.CompletedProcess[str]) -> bool:
    return "Error" not in result.stdout

def print_compiled_output(result: subprocess.CompletedProcess[str]):
        print(f"\n\n\n\n New Output from Midio compilation of code:")
        print(f"STDOUT (semantic errors): {result.stdout}")
        print(f"STDERR (syntactical errors): {result.stderr}")
def remo

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
