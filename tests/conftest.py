import pytest
import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Base directory for tests: {BASE_DIR}")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    load_dotenv()