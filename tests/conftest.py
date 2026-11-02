import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from solar_rl import load_config  # noqa: E402
from solar_rl.sim.weather import synthetic_year  # noqa: E402


def synth_overrides(extra=None):
    """Config overrides that force deterministic synthetic weather (tests must not depend on data/)."""
    from solar_rl.config import merge
    return merge({"data": {"dir": os.path.join(ROOT, "tests", "_no_data_"), "years": [2021]}}, extra or {})


@pytest.fixture(scope="session")
def weather_2021():
    return synthetic_year(2021, load_config("default"))


@pytest.fixture
def cfg_factory():
    def make(name="default", extra=None):
        return load_config(name, synth_overrides(extra))
    return make
