from datetime import datetime

CURRENT_DEBUG_DIR = None
from pathlib import Path

def set_debug_dir(path: Path):
    global CURRENT_DEBUG_DIR
    CURRENT_DEBUG_DIR = path

def debug_log(*args):
    if not CURRENT_DEBUG_DIR:
        return

    with open(CURRENT_DEBUG_DIR / "debug.log", "a", encoding="utf-8") as f:
        for a in args:
            f.write(str(a) + "\n")
        f.write("\n")
