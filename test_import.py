import sys
import subprocess

# Print Python version and sys.path
print("Python version:", sys.version)
print("Sys.path:", sys.path)

# Check installed packages
installed_packages = subprocess.run(["pip", "list"], capture_output=True, text=True)
print("Installed packages:\n", installed_packages.stdout)

# Try importing transformers
try:
    from transformers import pipeline
    print("Transformers imported successfully!")
except ModuleNotFoundError as e:
    print("Error importing transformers:", e)
