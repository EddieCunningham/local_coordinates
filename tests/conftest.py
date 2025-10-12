import os

# Ensure JAX uses 64-bit precision for all tests
os.environ["JAX_ENABLE_X64"] = "true"

import jax

jax.config.update("jax_enable_x64", True)


