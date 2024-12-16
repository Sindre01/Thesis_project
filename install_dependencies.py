import subprocess
import sys

def install_package(package_name):
    """
    Install a single package using pip.
    """
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}. Error: {e}")
        sys.exit(1)

def install_dependencies(requirements_file=None, dependencies=None):
    """
    Install dependencies from a requirements file or a list of dependencies.
    """
    if requirements_file:
        print(f"Installing dependencies from {requirements_file}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Successfully installed all dependencies from the requirements file!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies from {requirements_file}. Error: {e}")
            sys.exit(1)

    if dependencies:
        print("Installing individual dependencies...")
        for package in dependencies:
            install_package(package)

if __name__ == "__main__":
    # Specify the requirements file (optional)
    requirements_file = ""

    # List additional dependencies (optional)
    dependencies = [
        "scikit-learn>=1.0",
        "numpy>=1.20",
        "pandas>=1.3",
        "transformers>=4.0",
        "torch>=1.9",
        "scikit-learn",
        "iterative-stratification", 
        "matplotlib"   
    ]

    # Install dependencies
    install_dependencies(requirements_file=requirements_file, dependencies=dependencies)
