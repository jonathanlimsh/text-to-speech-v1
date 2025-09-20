﻿from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tts_cli.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
