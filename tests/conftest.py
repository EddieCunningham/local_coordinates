import os

# Ensure JAX uses 64-bit precision for all tests
os.environ["JAX_ENABLE_X64"] = "true"

import jax

jax.config.update("jax_enable_x64", True)

# --- Added: CLI option and fixture for GIF output directory ---
import pytest
from pathlib import Path


def pytest_addoption(parser: pytest.Parser) -> None:
  parser.addoption(
    "--gif-dir",
    action="store",
    default=None,
    help="Directory to save optimization GIFs (defaults to pytest tmp_path)",
  )


@pytest.fixture
def gif_dir(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
  opt = request.config.getoption("--gif-dir")
  if opt:
    p = Path(opt)
    p.mkdir(parents=True, exist_ok=True)
    return p
  return tmp_path


