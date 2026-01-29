import time
from pathlib import Path
from datetime import datetime
import re

def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\d\-_. ]+", "_", s)
    s = s.strip().replace(" ", "_")
    return s[:120] or "query"

class TimingLog:
    def __init__(self, query: str):
        self.query = query

        # span timers
        self.active = {}          # name → start_time
        self.marks = {}           # name → total_time

        # nested steps
        self.submarks = {}        # group → { step → time }

    # -----------------------------
    # SPAN API (pipeline dùng)
    # -----------------------------
    def start(self, name: str):
        self.active[name] = time.time()

    def end(self, name: str):
        if name not in self.active:
            return
        dt = time.time() - self.active.pop(name)
        self.marks[name] = self.marks.get(name, 0) + dt

    # dùng cho multi-hop
    def sub(self, group: str, name: str, dt: float):
        self.submarks.setdefault(group, {})
        self.submarks[group][name] = dt
        self.marks[group] = self.marks.get(group, 0) + dt

    # -----------------------------
    # FINALIZE & WRITE FILE
    # -----------------------------
    def finish(self, enable=True):
        if not enable:
            return

        total = sum(self.marks.values())

        base = Path("debug_timing")
        base.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe = _sanitize_filename(self.query)
        path = base / f"{ts}__{safe}.txt"

        lines = []
        lines.append(f"QUERY: {self.query}\n")

        # main spans
        for name, t in self.marks.items():
            if name not in self.submarks:
                lines.append(f"- {name}: {t:.3f}s")

        # nested groups (multi hop etc)
        for group, steps in self.submarks.items():
            lines.append(f"\n[{group}]")
            for name, t in steps.items():
                lines.append(f"  - {name}: {t:.3f}s")

        lines.append(f"\nTOTAL: {total:.3f}s")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"[TIMING] log written to: {path}")
        except Exception as e:
            print("[TIMING ERROR]", e)
