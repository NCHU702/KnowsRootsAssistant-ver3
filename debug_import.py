import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    import system_api.graph_manager
    print(f"system_api.graph_manager file: {system_api.graph_manager.__file__}")
    print(f"dir(system_api.graph_manager): {dir(system_api.graph_manager)}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
