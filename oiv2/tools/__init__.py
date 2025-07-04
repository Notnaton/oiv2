import importlib, pkgutil, os

for _, name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    try: importlib.import_module(f"{__name__}.{name}")
    except ImportError as e: print(f"Warning: {name} - {e}")
