#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    scripts = sorted(
        p for p in here.glob("dataclass_to_gguf_models_*.py") if p.name != "dataclass_to_gguf_models_base.py"
    )

    if not scripts:
        print("No example scripts found.", file=sys.stderr)
        sys.exit(1)

    for script in scripts:
        print(f"Running {script.name}...")
        result = subprocess.run([sys.executable, str(script)], cwd=here)
        if result.returncode != 0:
            print(f"Script {script.name} failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

    print("All dataclass_to_gguf examples completed.")


if __name__ == "__main__":
    main()
