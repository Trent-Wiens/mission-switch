#!/usr/bin/env python3
"""Run the whole test suite.

    python3 run_tests.py            all tests
    python3 run_tests.py -v         verbose
    python3 run_tests.py sim        only tests/test_sim.py
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv: list[str]) -> int:
    verbosity = 2 if "-v" in argv else 1
    selectors = [a for a in argv if not a.startswith("-")]

    loader = unittest.TestLoader()
    if selectors:
        suite = unittest.TestSuite()
        for name in selectors:
            module = name if name.startswith("test_") else f"test_{name}"
            suite.addTests(loader.loadTestsFromName(f"tests.{module}"))
    else:
        suite = loader.discover(os.path.join(HERE, "tests"), top_level_dir=HERE)

    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
