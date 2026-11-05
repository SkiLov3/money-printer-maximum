"""Test-suite configuration.

The live ``config/trading_stack.yaml`` is private and git-ignored (it holds the
operator's tuned parameters and is rewritten by the Saturday optimizer). Tests must
never depend on it, so point every implicit ``load_stack_config()`` call at the
committed example config instead.
"""

import os
from pathlib import Path

os.environ["STACK_CONFIG_PATH"] = str(
    Path(__file__).resolve().parent.parent / "config" / "trading_stack.example.yaml"
)
