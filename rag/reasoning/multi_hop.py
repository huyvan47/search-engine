import json
from typing import List, Dict, Tuple, Any

from rag.config import RAGConfig
from rag.logging.multi_hop_logger import write_multi_hop_logs
from rag.retriever import search as retrieve_search
from rag.tag_filter import tag_filter_pipeline


# -----------------------------
# 1) Intent strategy (optional)
# -----------------------------
def analyze_intent_strategy(client, query: str) -> Dict[str, Any]:
    """
    Nhẹ, không bắt buộc. Dùng để tạo 'hint' cho fallback logic.
    Không dùng để hard-code tags (vì bạn đã có tagger mạnh).
    """
    sys = """
Bạn là chuyên gia nông nghiệp.

Nhiệm vụ:
- Hiểu mục tiêu thực sự của người dùng
- Xác định:
  - primary_target (mục tiêu chính)
  - fallback_targets (mục tiêu thay thế hợp lý)

Chỉ trả về JSON hợp lệ theo schema:
{
  "primary_target": string,
  "fallback_targets": [string, ...]
}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": query},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        if not isinstance(data, dict):
            return {"primary_target": "", "fallback_targets": []}
        data.setdefault("primary_target", "")
        data.setdefault("fallback_targets", [])
        if not isinstance(data.get("fallback_targets"), list):
            data["fallback_targets"] = []
        return data
    except Exception:
        return {"primary_target": "", "fallback_targets": []}


# -----------------------------------------
# 2) Decide next hop + next query (LLM-loop)
# -----------------------------------------
def analyze_need_next_hop(
    *,
    client,
    query: str,
    hits: List[dict],
    hop: int,
    max_hops: int,
) -> Tuple[bool, str, str]:
    """
    LLM chỉ được phép:
      - quyết định có cần hop tiếp không
      - nếu cần, sinh next_query (NGẮN, KHÔNG TRÙNG query cũ)
    Hệ thống vẫn có stop cứng riêng.
    """
    if hop >= max_hops:
        return False, "", "reach_max_hops"

    sys = """
Bạn là module điều phối truy vấn cho hệ thống RAG nông nghiệp.

Nhiệm vụ:
- Nhìn vào câu hỏi và danh sách tài liệu đã tìm được
- Quyết định có cần truy vấn thêm bước nữa (multi-hop) hay không
- Nếu cần, tạo ra MỘT truy vấn tiếp theo, ngắn gọn, cụ thể hơn

Quy tắc:
- Chỉ trả về JSON hợp lệ
- KHÔNG trả lời nội dung câu hỏi
- KHÔNG sinh lại chính câu query cũ
- KHÔNG tạo truy vấn quá dài
- Chỉ tạo query mới khi thực sự thiếu thông tin

Output JSON schema:
{
  "need_next_hop": boolean,
  "next_query": string,
  "reason": string
}
"""

    sample_hits = [
        {
            "id": h.get("id"),
            "question": h.get("question"),
            "tags": h.get("tags_v2"),
        }
        for h in (hits or [])[:10]
    ]

    payload = {
        "original_query": query,
        "hop_index": hop,
        "num_hits": len(hits or []),
        "sample_hits": sample_hits,
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

        need = bool(data.get("need_next_hop", False))
        nq = str(data.get("next_query", "")).strip()
        reason = str(data.get("reason", "")).strip()

        # Guard: không cho next_query rỗng hoặc trùng
        if not need or not nq:
            return False, "", reason or "llm_stop_or_empty_next_query"

        return True, nq, reason or "need_next_hop"

    except Exception:
        return False, "", "llm_exception"


# -----------------------------------------
# 3) Utility: dedupe hits + bookkeeping
# -----------------------------------------
def _dedupe_hits(hits: List[dict], seen_ids: set) -> List[dict]:
    out = []
    for h in (hits or []):
        hid = h.get("id")
        if not hid or hid in seen_ids:
            continue
        seen_ids.add(hid)
        out.append(h)
    return out


def _get_cfg(name: str, default):
    return getattr(RAGConfig, name, default)

def safe_json_array(text):
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    # fallback: extract quoted strings
    import re
    return re.findall(r'"([^"]+)"', text)

def infer_pest_from_problem(client, base_query: str) -> List[str]:
    """
    Infer likely pests/insects/mites/nematodes from user's problem.
    Output MUST be Vietnamese pest names.
    """
    prompt = f"""
You are an expert in agricultural pests (insects, mites, nematodes).

User query:
"{base_query}"

Task:
Infer the most likely pests the user is referring to OR the most common pests relevant to this situation.
Return pest names in VIETNAMESE that Vietnamese agronomists/farmers commonly use.

Rules:
- Only pest names (no products, no treatments)
- Include insects/mites if relevant; include nematodes only if strongly relevant
- 3–8 items
- Prefer specific pests over generic terms
- Output MUST be a JSON array in Vietnamese
- No explanations
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    arr = safe_json_array(resp.choices[0].message.content)

    # light normalize (optional)
    out = []
    for x in arr:
        s = str(x).strip()
        if not s:
            continue
        # avoid super generic junk
        if s.lower() in {"côn trùng", "sâu hại", "bọ", "sâu"}:
            continue
        out.append(s)

    # de-dup keep order
    seen = set()
    final = []
    for s in out:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        final.append(s)

    return final[:8]

def infer_mechanism_from_pest(client, base_query: str) -> List[str]:
    """
    Infer control mechanisms for pest problems.
    Output MUST be Vietnamese mechanism phrases usable as search queries.
    """
    prompt = f"""
You are an expert in pest control mechanisms for agriculture.

User query:
"{base_query}"

Task:
List mechanisms / mode-of-action styles typically used to control the likely pests in this query.

Return Vietnamese phrases that can be used as search queries, for example:
- "tác động tiếp xúc"
- "tác động vị độc"
- "nội hấp lưu dẫn"
- "xông hơi"
- "ức chế sinh trưởng côn trùng (IGR)"
- "tác động thần kinh côn trùng"
- "diệt trứng và ấu trùng"
- "trừ bọ trĩ nội hấp"
- "trừ rệp sáp tiếp xúc"

Rules:
- 4–8 items
- No product names, no active ingredient names
- Output MUST be a JSON array in Vietnamese
- No explanations
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    arr = safe_json_array(resp.choices[0].message.content)

    # sanitize + de-dup
    cleaned = []
    for x in arr:
        s = str(x).strip()
        if not s:
            continue
        # avoid overly generic
        if s.lower() in {"cơ chế", "biện pháp", "phòng trừ"}:
            continue
        cleaned.append(s)

    seen = set()
    final = []
    for s in cleaned:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        final.append(s)

    return final[:8]

def infer_disease_from_symptom(client, base_query: str) -> List[str]:
    prompt = f"""
You are a plant pathologist.

User described this crop problem:
"{base_query}"

Infer the most likely plant diseases or pathogens.
Return the names in VIETNAMESE that a Vietnamese agronomist would use.

Rules:
- Only disease or pathogen names (no products)
- 3–6 items
- No explanations
- Return JSON array in Vietnamese
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return safe_json_array(resp.choices[0].message.content)

def infer_mechanism_from_disease(client, base_query: str) -> List[str]:
    prompt = f"""
You are a plant pathology expert.

Given this crop problem:
"{base_query}"

List the biological or chemical control mechanisms usually involved in treating it.

Examples:
- "fungal cell wall synthesis"
- "oomycete inhibition"
- "insect nervous system"
- "viral replication"
- "systemic acquired resistance"

Rules:
- 3–6 mechanisms
- Do NOT mention any product names
- Return JSON array only
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return safe_json_array(resp.choices[0].message.content)

def infer_formula_queries(client, base_query: str) -> List[str]:
    prompt = f"""
You are designing crop treatment strategies.

User problem:
"{base_query}"

Generate high-level treatment formula concepts that would be used.
Use format like:
- "fungicide + insecticide"
- "systemic + contact"
- "oomycete control + root protection"
- "stress recovery + disease control"

Rules:
- 3–6 items
- Do NOT include product names
- Return JSON array only
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return safe_json_array(resp.choices[0].message.content)

def expand_query_with_llm(client, base_query: str) -> List[str]:
    prompt = f"""
Rewrite this crop-related question into multiple alternative search queries:

"{base_query}"

Include:
- scientific terms
- agronomy style phrasing
- common farmer phrasing

Rules:
- 4–8 variants
- Do not change meaning
- No explanations
- Return JSON array only
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return safe_json_array(resp.choices[0].message.content)

def infer_product_intent_queries(client, base_query: str) -> List[str]:
    prompt = f"""
User said:
"{base_query}"

Generate search queries that a person would use when looking for agricultural products to solve this.

Examples:
- "thuốc trị thối rễ lúa"
- "fungicide for downy mildew"
- "systemic insecticide for aphids"

Rules:
- 3–6 queries
- No brand names unless user already mentioned one
- Return JSON array only
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return safe_json_array(resp.choices[0].message.content)

def decide_recovery_branch(any_tags):
    if any(t.startswith("pest:") for t in any_tags):
        return "pest"
    if any(t.startswith("disease:") for t in any_tags):
        return "disease"
    if any(t.startswith("weed:") for t in any_tags):
        return "weed"
    return "unknown"

def no_hit_recovery_pipeline(
    *,
    client,
    kb,
    base_query: str,
    any_tags: List[str],
    top_k: int,
    max_docs: int = 32,
):
    """
    Evidence-based recovery when hop1 returns no hits.
    Run all recovery strategies in parallel and let evidence compete.
    """

    all_hits = []
    seen_ids = set()
    branch = decide_recovery_branch(any_tags)
    # ========= 1. Sinh recovery hypotheses =========
    if branch == "pest":
        recovery_queries = {
            "pest": infer_pest_from_problem(client, base_query),
            "mechanism": infer_mechanism_from_pest(client, base_query),
            "formula": infer_formula_queries(client, base_query),
            "expand": expand_query_with_llm(client, base_query),
        }

    elif branch == "disease":
        recovery_queries = {
            "disease": infer_disease_from_symptom(client, base_query),
            "mechanism": infer_mechanism_from_disease(client, base_query),
            "formula": infer_formula_queries(client, base_query),
            "expand": expand_query_with_llm(client, base_query),
        }

    else:
        recovery_queries = {
            "expand": expand_query_with_llm(client, base_query),
            "formula": infer_formula_queries(client, base_query),
        }

    # ========= 2. Thu thập bằng chứng song song =========
    for mode, queries in recovery_queries.items():
        print(f" - {mode}:")
        for q in queries:
            print("    •", q)
            tag_result = tag_filter_pipeline(q)
            print("  tags:", tag_result)
            must = tag_result.get("must", [])
            any_  = tag_result.get("any", [])
            # 🚫 Reject ungrounded queries (no ontology anchor)
            if not must and not any_:
                print("  ⚠ SKIP (no tags) →", q)
                continue
            # FORMULA: dùng tag sinh ra, không dùng any_tags gốc
            if mode == "formula":
                any_ = any_

            hits = retrieve_search(
                client=client,
                kb=kb,
                norm_query=q,
                top_k=top_k,
                must_tags=must,
                any_tags=any_,
            )
            print(f"  hits={len(hits)}")

            for h in hits:
                print(f"    + ADD doc {h.get('id')} score={h.get('score'):.4f}")
                doc_id = h.get("id")
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                h["recovery_mode"] = mode
                h["origin_query"] = q
                all_hits.append(h)

    print("\n[TOTAL RECOVERED DOCS]", len(all_hits))
    from collections import Counter
    print("By mode:", Counter(h["recovery_mode"] for h in all_hits))

    if not all_hits:
        return []

    # ========= 3. Gán điểm để chọn top 32 =========

    def score_doc(h):
        sim = float(h.get("score", 0))
        tag_count = len(str(h.get("tags_v2", "")).split(","))

        mode_weight = {
            "disease": 1.0,
            "formula": 0.9,
            "mechanism": 0.7,
            "expand": 0.6,
        }.get(h.get("recovery_mode"), 0.5)

        return 0.6 * sim + 0.3 * tag_count + 0.1 * mode_weight

    # ========= 4. Bảo vệ đa thế thoại =========
    from collections import defaultdict
    buckets = defaultdict(list)

    for h in all_hits:
        buckets[h["recovery_mode"]].append(h)
    print("\n[BUCKET SIZES]")
    for mode, docs in buckets.items():
        print(f"  {mode}: {len(docs)}")
    final = []

    # mỗi mode tối thiểu 3 doc nếu có
    for mode, docs in buckets.items():
        docs_sorted = sorted(docs, key=score_doc, reverse=True)
        final.extend(docs_sorted[:3])

    # fill phần còn lại theo score toàn cục
    remaining = [h for h in all_hits if h not in final]
    remaining_sorted = sorted(remaining, key=score_doc, reverse=True)
    print("\n[FINAL SELECTED DOCS]", len(final))
    for h in remaining_sorted:
        print(f"  {h['recovery_mode']:10} | {h.get('id')} | {h.get('score'):.4f}")
        if len(final) >= max_docs:
            break
        final.append(h)
    print("=== [NO-HIT RECOVERY END] ===\n")
    return final

# -----------------------------------------
# 4) Multi-hop controller (production-grade)
# -----------------------------------------
def multi_hop_controller(
    *,
    client,
    kb,
    base_query: str,
    must_tags: List[str],
    any_tags: List[str],
) -> List[dict]:

    # ---- Config ----
    top_k   = int(_get_cfg("multi_hop_top_k", 20))
    max_hops = int(_get_cfg("max_multi_hops", 3))
    min_docs = int(_get_cfg("min_docs_for_answer", 25))
    enable_logs = bool(_get_cfg("enable_multi_query_log", True))

    must_tags = list(must_tags or [])
    any_tags  = list(any_tags or [])

    all_hits: List[dict] = []
    seen_ids = set()
    hops_data: List[dict] = []
    used_queries = set()

    intent_hint = analyze_intent_strategy(client, base_query)
    used_queries.add(base_query)

    # =========================
    # HOP 1 — Router confidence check
    # =========================

    hop1_exists = bool(must_tags)

    if hop1_exists:
        hits1 = retrieve_search(
            client=client,
            kb=kb,
            norm_query=base_query,
            top_k=top_k,
            must_tags=must_tags,
            any_tags=[],
        )

        unique_hits1 = _dedupe_hits(hits1, seen_ids)
        all_hits.extend(unique_hits1)

        hop1_record = {
            "hop": 1,
            "query": base_query,
            "num_hits": len(unique_hits1),
            "hits": unique_hits1,
            "decision": {
                "intent_hint": intent_hint,
                "stop_reason": "",
            },
        }
        hops_data.append(hop1_record)

    else:
        unique_hits1 = []
        hops_data.append({
            "hop": 1,
            "query": base_query,
            "num_hits": 0,
            "hits": [],
            "decision": {
                "intent_hint": intent_hint,
                "stop_reason": "no_must_tags_skip_hop1",
            },
        })

    # =========================
    # HOP1 FAILED → Recovery
    # =========================
    if not unique_hits1:
        if hop1_exists:
            hops_data[-1]["decision"]["stop_reason"] = "no_hits_hop1_recovered"

        recovered_hits = no_hit_recovery_pipeline(
            client=client,
            kb=kb,
            base_query=base_query,
            any_tags=any_tags,
            top_k=top_k,
        )

        recovered_unique = _dedupe_hits(recovered_hits, seen_ids)
        all_hits.extend(recovered_unique)

        if enable_logs:
            write_multi_hop_logs(
                original_query=base_query,
                hops_data=hops_data,
                final_hits=all_hits,
            )
        return all_hits

    # =========================
    # Enough coverage after Hop1
    # =========================
    if len(all_hits) >= min_docs:
        hops_data[-1]["decision"]["stop_reason"] = f"enough_docs_after_hop1>={min_docs}"
        if enable_logs:
            write_multi_hop_logs(
                original_query=base_query,
                hops_data=hops_data,
                final_hits=all_hits,
            )
        return all_hits

    # =========================
    # HOP 2..N
    # =========================
    current_query = base_query

    for hop in range(2, max_hops + 1):

        need, next_query, llm_reason = analyze_need_next_hop(
            client=client,
            query=current_query,
            hits=all_hits,
            hop=hop,
            max_hops=max_hops,
        )

        decision = {
            "need_next_hop": need,
            "llm_reason": llm_reason,
            "stop_reason": "",
        }

        if not need or not next_query:
            decision["stop_reason"] = "llm_stop"
            hops_data.append({
                "hop": hop,
                "query": current_query,
                "num_hits": 0,
                "hits": [],
                "decision": decision,
            })
            break

        if next_query in used_queries:
            decision["stop_reason"] = "repeat_query_blocked"
            hops_data.append({
                "hop": hop,
                "query": next_query,
                "num_hits": 0,
                "hits": [],
                "decision": decision,
            })
            break

        used_queries.add(next_query)

        hits_h = retrieve_search(
            client=client,
            kb=kb,
            norm_query=next_query,
            top_k=top_k,
            must_tags=[],
            any_tags=any_tags if any_tags else [],
        )

        unique_hits_h = _dedupe_hits(hits_h, seen_ids)
        added = len(unique_hits_h)
        all_hits.extend(unique_hits_h)

        decision["added_new_docs"] = added
        decision["search_mode"] = "any_tags" if any_tags else "free_search"

        hops_data.append({
            "hop": hop,
            "query": next_query,
            "num_hits": added,
            "hits": unique_hits_h,
            "decision": decision,
        })

        if added == 0:
            decision["stop_reason"] = "no_new_docs_added"
            break

        if len(all_hits) >= min_docs:
            decision["stop_reason"] = f"enough_docs>={min_docs}"
            break

        current_query = next_query

    if enable_logs:
        write_multi_hop_logs(
            original_query=base_query,
            hops_data=hops_data,
            final_hits=all_hits,
        )

    return all_hits
