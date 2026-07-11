"""pytest config — put code/ on sys.path so `import agent`, `import triage`,
`import io_csv`, etc. work from tests/ without a package install.

Mirrors the `sys.path.insert(0, ...)` shim each module already uses, so
tests see the same import surface as production code.
"""

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
