"""Small callable script example.

This file exposes a `main()` function so it can be used both as a script
and imported as a module. The `test` function is intentionally small and
wrapped with CodeCarbon's `track_emissions` decorator.

Usage:
  - Run as script: python test.py
  - Import: from test import main; main()
"""

from typing import Optional, Sequence
from codecarbon import track_emissions


@track_emissions()
def test() -> None:
  """Demo function that the tracker wraps.

  Keep this small and side-effect free so it can be called from other
  modules or tests.
  """
  print("Hello world")


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Program entry point.

  Args:
    argv: Optional sequence of command-line arguments (not used yet).

  Returns:
    Exit code (0 for success).
  """
  # argv is accepted to make it easy to later add CLI flags without
  # changing the function signature when called programmatically.
  test()
  return 0


if __name__ == "__main__":
  import sys

  # Use SystemExit so callers and test harnesses can capture the exit code.
  raise SystemExit(main(sys.argv[1:]))
