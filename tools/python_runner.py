import io
import os
import sys
import shutil
import subprocess
import contextlib
import tempfile
import importlib
from tools.tools import function_tool
from conversation import Message

@function_tool
def python_runner(code: str) -> Message:
    """
    Runs Python code, installs missing dependencies temporarily, and captures output.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    confirmation = input(f"{code}\n\nAre you sure you want to run the above code? (y/n):")
    if confirmation.lower() not in {"y", "yes"}:
        return Message(
            role="tool",
            message="Python code execution cancelled by user due to safety concerns, visible bugs in the code, or other reasons.",
            summary=""
        )

    # Extract import statements to check for missing modules
    imports = set()
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("import "):
            imports.add(line.split()[1].split(".")[0])  # Get only the base module
        elif line.startswith("from "):
            imports.add(line.split()[1].split(".")[0])

    # Check for missing modules
    missing_modules = [mod for mod in imports if not importlib.util.find_spec(mod)]

    # Create a temporary directory for dependencies if needed
    temp_lib_dir = None
    if missing_modules:
        temp_lib_dir = tempfile.mkdtemp()
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_modules, "--target", temp_lib_dir])
        sys.path.insert(0, temp_lib_dir)

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, globals())
    except Exception as e:
        output = f"Error: {e}"
    else:
        out = stdout_capture.getvalue()
        err = stderr_capture.getvalue()
        output = out if out else err if err else "Command executed successfully with no output"
    finally:
        # Cleanup temporary package directory
        if os.path.exists(temp_lib_dir):
            sys.path.remove(temp_lib_dir)
            shutil.rmtree(temp_lib_dir, ignore_errors=True)

    return Message(
        role="tool",
        message=f"{code}\n\n{output}",
        summary=f'Assistant executed the python function with the following code:\n`{code}`'
    )