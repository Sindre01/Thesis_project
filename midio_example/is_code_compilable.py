import subprocess
import os
import tempfile

# Function to load code from a file
def load_code_from_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

# Function to check if the code compiles using package-manager
def is_code_compilable(code: str) -> bool:
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
            print("Compiler stdout:", result.stdout)
            print("Compiler stderr:", result.stderr)

            # Return True if the compiler exits successfully
            return result.returncode == 0
        except FileNotFoundError:
            print("Error: package-manager not found in PATH.")
            return False
        
code = load_code_from_file("./main.midio")

if is_code_compilable(code):
    print("Correct response: Code compiled successfully.")
 
else:
    print("Invalid response: Compilation failed.")
