import importlib
import pkgutil
import os

# Get the package name
package_name = __name__
package_path = os.path.dirname(__file__)

# Import all modules in the tools package
for _, module_name, _ in pkgutil.iter_modules([package_path]):
    importlib.import_module(f"{package_name}.{module_name}")
