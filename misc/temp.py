import sys

if sys.prefix != sys.base_prefix:
    print("Running inside a virtual environment.")
else:
    print("Running in the global Python environment.")
