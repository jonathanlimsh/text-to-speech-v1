import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

VENV_SITE_CANDIDATES = [
    ROOT / "venv" / "Lib" / "site-packages",
    ROOT / "venv" / "lib" / "python3.11" / "site-packages",
]
for candidate in VENV_SITE_CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def main() -> int:
    suite = unittest.defaultTestLoader.discover('tests')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
