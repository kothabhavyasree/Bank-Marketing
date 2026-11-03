import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                      # every test runs from the project root
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from common import load_config, load_clean_raw, split_raw  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_split():
    return split_raw(load_clean_raw())


def require(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} missing - run the earlier lab pipelines first")
