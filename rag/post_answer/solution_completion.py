import json
from typing import List, Dict, Any
from rag.tag_filter import tag_filter_pipeline
from rag.retriever import search as retrieve_search
from rag.config import RAGConfig
from rag.logging.t4_logger import append_t4_log_to_csv

# -----------------------------
# Helpers
# -----------------------------
def _dedupe_hits(hits: List[dict], seen_ids: set) -> List[dict]:
    out = []
    for h in hits or []:
        hid = h.get("id")
        if not hid or hid in seen_ids:
            continue
        seen_ids.add(hid)
        out.append(h)
    return out

def _hits_brief(hits: List[dict], n: int = 12) -> List[dict]:
    out = []
    for h in (hits or [])[:n]:
        out.append({
            "id": h.get("id"),
            "question": h.get("question"),
            "tags": h.get("tags_v2"),
        })
    return out

def _draft_answer_stub(base_query: str, hits: List[dict]) -> str:
    if not hits:
        return "Không tìm thấy dữ liệu phù hợp trong hệ thống."
    return f"Dựa trên tài liệu hiện có cho câu hỏi '{base_query}', có thể tồn tại giải pháp phối hợp nhưng chưa đủ thành phần cụ thể."


# -----------------------------
# Slot Detector (LLM)
# -----------------------------
def detect_solution_slots(client, base_query: str, draft_answer: str, hits: List[dict]) -> Dict[str, Any]:
    sys = """
Bạn là module kiểm tra "đủ để hành động" cho hệ thống RAG nông nghiệp.

Input:
- user_query
- draft_answer (tóm tắt hiện tại)
- evidence_hits (một số tài liệu)

Nhiệm vụ:
1) Nếu draft_answer ngụ ý giải pháp dạng "kết hợp A + B" hoặc "có thể phối hợp",
   hãy liệt kê các thành phần hành động còn thiếu (slots).
2) Sinh tối đa 3 truy vấn ngắn để tìm thêm bằng chứng.

Quy tắc:
- Chỉ JSON hợp lệ.
- Không bịa sản phẩm/hoạt chất.
- Query 2–6 từ, tiếng Việt.
- Không lặp lại user_query.

Schema:
{
  "missing_slots": [string, ...],
  "search_intents": [{"slot": string, "query": string}],
  "reason": string
}

Slot gợi ý:
- need_pesticide
- need_foliar_fertilizer
- need_crop
- need_pest_or_disease
- need_mix_compatibility
- need_dosage_or_rate
- need_timing
"""
    payload = {
        "user_query": base_query,
        "draft_answer": draft_answer,
        "evidence_hits": _hits_brief(hits, n=12),
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.1,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        data.setdefault("missing_slots", [])
        data.setdefault("search_intents", [])
        data.setdefault("reason", "")

        clean = []
        for it in data.get("search_intents", [])[:3]:
            if isinstance(it, dict):
                slot = str(it.get("slot", "")).strip()
                q = str(it.get("query", "")).strip()
                if slot and q:
                    clean.append({"slot": slot, "query": q})
        data["search_intents"] = clean
        if not isinstance(data.get("missing_slots"), list):
            data["missing_slots"] = []

        return data
    except Exception:
        return {"missing_slots": [], "search_intents": [], "reason": "slot_detector_exception"}


# -----------------------------
# Validator (MVP)
# -----------------------------
def validate_answer_readiness(slot_report: Dict[str, Any]) -> Dict[str, Any]:
    missing = slot_report.get("missing_slots", []) or []
    critical = {"need_pesticide", "need_foliar_fertilizer"}
    if any(s in critical for s in missing):
        return {"status": "need_more_evidence"}
    return {"status": "ok"}


# -----------------------------
# Execute T4 retrieval
# -----------------------------
def _t4_retrieve(
    *,
    client,
    kb,
    intents: List[Dict[str, str]],
    any_tags: List[str],
    seen_ids: set,
    used_queries: set,
    top_k: int,
) -> List[dict]:

    added = []
    for it in intents[:3]:
        q = it["query"]
        if q in used_queries:
            continue
        used_queries.add(q)

        tag_result = tag_filter_pipeline(q)
        must = tag_result.get("must", [])
        any_ = tag_result.get("any", [])

        hits = retrieve_search(
            client=client,
            kb=kb,
            norm_query=q,
            top_k=top_k,
            must_tags=must,
            any_tags=any_ if (must or any_) else (any_tags or []),
        )
        unique = _dedupe_hits(hits, seen_ids)
        for h in unique:
            h["t4_origin_query"] = q
            h["t4_intent_slot"] = it.get("slot")
        added.extend(unique)
    return added


# -----------------------------
# Public API
# -----------------------------
def run_solution_completion(
    *,
    run_dir,
    client,
    kb,
    user_query: str,
    hits: List[dict],
    must_tags: List[str],
    any_tags: List[str],
    l3_missing_slots: List[str],   # 🔥 TÍN HIỆU DUY NHẤT
) -> List[dict]:

    added = []
    slot_report = {
        "missing_slots": l3_missing_slots,
        "reason": "from_L3"
    }

    # -------------------------------
    # T4 chỉ chạy khi L3 nói thiếu
    # -------------------------------
    if hits and l3_missing_slots and bool(getattr(RAGConfig, "enable_t4_solution_completion", True)):

        seen_ids = {h.get("id") for h in hits if h.get("id")}
        used_queries = {user_query}

        # Map slot → query
        intents = []
        for s in l3_missing_slots:
            if s == "need_pesticide":
                intents.append({"slot": s, "query": "thuốc trừ sâu"})
            elif s == "need_foliar_fertilizer":
                intents.append({"slot": s, "query": "phân bón lá"})
            elif s == "need_mix_compatibility":
                intents.append({"slot": s, "query": "pha chung thuốc"})
            elif s == "need_dosage_or_rate":
                intents.append({"slot": s, "query": "liều lượng pha"})
            elif s == "need_timing":
                intents.append({"slot": s, "query": "thời điểm phun"})
            elif s == "need_pest_or_disease":
                intents.append({"slot": s, "query": "đối tượng gây hại"})

        t4_top_k = int(getattr(RAGConfig, "t4_top_k", max(8, RAGConfig.multi_hop_top_k // 2)))

        added = _t4_retrieve(
            client=client,
            kb=kb,
            intents=intents,
            any_tags=any_tags,
            seen_ids=seen_ids,
            used_queries=used_queries,
            top_k=t4_top_k,
        )

    # ==============================
    # T4 LOGGING — LUÔN GHI
    # ==============================
    ctx_docs_total = min(len(hits) + len(added), RAGConfig.max_ctx_strict)
    t4_docs_in_ctx = min(len(added), ctx_docs_total)

    append_t4_log_to_csv(
        run_dir=run_dir,
        user_query=user_query,
        norm_query=user_query,
        slot_report=slot_report,
        intents_executed=[i["query"] for i in intents] if l3_missing_slots else [],
        added_docs=added,
        ctx_docs_total=ctx_docs_total,
        t4_docs_in_ctx=t4_docs_in_ctx,
    )

    return hits + added
