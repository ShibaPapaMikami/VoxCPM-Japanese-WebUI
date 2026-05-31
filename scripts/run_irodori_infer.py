from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    irodori_dir = repo_root / "external" / "Irodori-TTS"
    infer_py = irodori_dir / "infer.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"Irodori-TTS infer.py was not found: {infer_py}")

    sys.path.insert(0, str(irodori_dir))
    sys.argv = [str(infer_py), *sys.argv[1:]]
    runpy.run_path(str(infer_py), run_name="__main__")


if __name__ == "__main__":
    main()
