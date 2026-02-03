import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from rag.logging.debug_log import debug_log


def append_t4_log_to_csv(
    *,
    run_dir: Path,
    user_query: str,
    norm_query: str,
    slot_report: dict,
    intents_executed: list,
    added_docs: list,
    ctx_docs_total: int,
    t4_docs_in_ctx: int,
):
    """
    Ghi log cho Solution Completion (T4) – mỗi dòng = 1 query
    """
    csv_path = run_dir / "debug_layer4.csv"
    print('csv_path:', csv_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    debug_log("[T4]", ts)

    # ---- thống kê từ added_docs ----
    slot_counter = {}
    tag_counter = {}

    for h in added_docs or []:
        slot = h.get("t4_intent_slot")
        if slot:
            slot_counter[slot] = slot_counter.get(slot, 0) + 1

        tags = str(h.get("tags_v2") or "")
        for t in tags.split(","):
            t = t.strip()
            if not t:
                continue
            tag_counter[t] = tag_counter.get(t, 0) + 1

    ratio = (t4_docs_in_ctx / ctx_docs_total) if ctx_docs_total else 0.0

    row = {
        "timestamp": ts,
        "user_query": user_query,
        "norm_query": norm_query,

        # Slot detection
        "missing_slots": json.dumps(slot_report.get("missing_slots", []), ensure_ascii=False),
        "slot_reason": slot_report.get("reason", ""),

        # Query planner
        "intents": json.dumps(intents_executed, ensure_ascii=False),

        # Retrieval result
        "docs_added": len(added_docs),
        "docs_by_slot": json.dumps(slot_counter, ensure_ascii=False),
        "tags_added": json.dumps(tag_counter, ensure_ascii=False),

        # Context injection
        "ctx_docs_total": ctx_docs_total,
        "t4_docs_in_ctx": t4_docs_in_ctx,
        "t4_injection_ratio": round(ratio, 4),
    }

    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)

    df.to_csv(
        csv_path,
        mode="a",
        index=False,
        header=header,
        encoding="utf-8-sig",
    )

    return csv_path
