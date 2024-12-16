import subprocess
import sys

# List of required packages
REQUIRED_PACKAGES = [
    "scikit-learn",
    "iterative-stratification",  # For iterstrat.ml_stratifiers
    "matplotlib"   
]

# Function to install a package
def install_package(package_name):
    try:
        print(f"Checking installation for package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Package '{package_name}' installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package '{package_name}'. Error: {e}")
        sys.exit(1)

# Install all required packages
def install_dependencies():
    for package in REQUIRED_PACKAGES:
        install_package(package)

if __name__ == "__main__":
    install_dependencies()
