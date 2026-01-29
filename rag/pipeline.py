import json
from rag.config import RAGConfig
from rag.router import route_query
from rag.normalize import normalize_query
from rag.text_utils import is_listing_query, extract_img_keys
from rag.retriever import search as retrieve_search
from rag.scoring import fused_score
from rag.context_builder import choose_adaptive_max_ctx, build_context_from_hits
from rag.memory.conversation_manager import read_memory, log_event, build_conversation_text, write_memory
from rag.memory.summarizer import summarize_to_fact
from rag.conversation_state import conversation_state
from rag.query_rewriter import needs_rewrite, format_history, rewrite_query_with_llm
from rag.answer_modes import decide_answer_policy
from rag.generator import call_finetune_with_context
from rag.tag_filter import tag_filter_pipeline
from rag.logging.timing_logger import TimingLog
from rag.reasoning.multi_hop import multi_hop_controller
from typing import List, Tuple, Dict, Any
from rag.logging.debug_log import debug_log
from rag.post_answer.enricher import enrich_answer_if_needed


FORCE_MUST_TAGS = {
    "mechanisms:luu-dan-manh",
    "mechanisms:luu-dan",
    "mechanisms:tiep-xuc-manh",
    "mechanisms:tiep-xuc",
    "mechanisms:tiep-xuc-luu-dan-manh",
    "mechanisms:tiep-xuc-luu-dan",
    "mechanisms:xong-hoi-manh",
    "mechanisms:xong-hoi",
    "mechanisms:co-chon-loc",
    "mechanisms:khong-chon-loc",
}

FORMULA_TRIGGERS = [
    
    "công thức",
    "phối trộn",
    "phối hợp thuốc",
    "liều phối",
    "phối",
    "pha thuốc",
    "công thức trị",
    "công thức trừ",
    "công thức diệt",
    "phác đồ",
    "kết hợp thuốc",
    "hoạt chất lưu dẫn phù hợp",
]


def is_formula_query(query: str, tags: dict) -> bool:
    """
    Nhận diện truy vấn dạng phối công thức.
    """

    has_plus = "+" in query
    has_mechanisms = any(
        t.startswith("mechanisms:")
        for t in tags.get("must", []) + tags.get("soft", [])
    )

    # Nếu có nhiều hơn 1 mechanism tag -> gần như chắc chắn là phối
    num_mechs = sum(
        1 for t in tags.get("must", []) + tags.get("soft", [])
        if t.startswith("mechanisms:")
    )

    if has_mechanisms and (has_plus or num_mechs >= 2):
        return True

    return False


from rag.config import RAGConfig


def formula_mode_search(
    *,
    client,
    kb,
    norm_query: str,
    must_tags: List[str],
):
    """
    Tìm kiếm theo chế độ công thức (không dùng multi-hop).
    Budget-aware:
    - Tổng ngân sách = RAGConfig.max_ctx_soft
    - Chia đều cho mỗi must_tag
    - Nếu chưa đủ → chạy free search để bù
    """

    max_ctx = RAGConfig.max_ctx_soft

    must_tags = list(must_tags or [])
    num_tags = max(len(must_tags), 1)

    # ngân sách cho mỗi tag
    per_tag_k = max(1, max_ctx // num_tags)

    all_results = []

    # ---- ROLE 1: MUST TAG (chia ngân sách) ----
    for m in must_tags:
        print("m:", m)
        hits = retrieve_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            top_k=per_tag_k,
            must_tags=[m],
            any_tags=[]
        )
        all_results.extend(hits)

    # dedupe theo id
    unique = {}
    for h in all_results:
        hid = h.get("id")
        if hid:
            prev = unique.get(hid)
            if not prev or h.get("score", 0) > prev.get("score", 0):
                unique[hid] = h

    results = list(unique.values())

    # ---- ROLE 2: FREE SEARCH (bù slot còn thiếu) ----
    remaining = max_ctx - len(results)

    if remaining > 0:
        hits_free = retrieve_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            top_k=remaining,
            must_tags=[],
            any_tags=[]
        )

        for h in hits_free:
            hid = h.get("id")
            if hid and hid not in unique:
                unique[hid] = h
                if len(unique) >= max_ctx:
                    break

        results = list(unique.values())

    # hard cap an toàn
    return results[:max_ctx]

def preserve_search_order(hits):
    """
    Đánh dấu thứ tự gốc từ search() để pipeline KHÔNG làm xáo trộn.
    """
    for idx, h in enumerate(hits):
        h["_search_rank"] = idx
    return hits

def _count_tag_hits(h, any_tags, must_tags):
    tv2 = str(h.get("tags_v2") or "")
    score = 0
    for t in (must_tags or []):
        if t and t in tv2:
            score += 3
    for t in (any_tags or []):
        if t and t in tv2:
            score += 1
    return score

def _global_system_prompt() -> str:
    return """
Bạn là chuyên gia BVTV/nông học tại Việt Nam. Mục tiêu: cung cấp câu trả lời CHẤT LƯỢNG CAO theo phong cách giáo trình/chuyên khảo,
giải thích rõ ràng, có chiều sâu, giàu ví dụ thực tế trong canh tác Việt Nam.

TIÊU CHUẨN CHẤT LƯỢNG (BẮT BUỘC):
- Ưu tiên: chính xác, mạch lạc, có tính “giải thích được” (explainable), không nói chung chung.
- Trình bày theo cấu trúc rõ ràng, có tiêu đề; dùng bullet và bảng (nếu hữu ích).
- Luôn phân biệt: (i) điều chắc chắn/phổ quát, (ii) điều phụ thuộc bối cảnh (cây, giai đoạn, thời tiết, áp lực dịch hại), (iii) điều cần thêm dữ liệu.
- Khi thuật ngữ/đối tượng có nhiều cách gọi tại VN: nêu tên thường gọi + mô tả nhận diện; tránh bịa tên loài.
- Nếu thiếu dữ liệu để kết luận chắc: nói rõ “phụ thuộc/ cần xác minh” và đưa tiêu chí/quan sát để người dùng tự kiểm chứng.

CẤU TRÚC CÂU TRẢ LỜI CHUẨN:
1) Tóm tắt nhanh (2–4 dòng): trả lời trực diện câu hỏi.
2) Định nghĩa/khái niệm cốt lõi (ngắn gọn).
3) Đặc điểm nhận biết / điểm then chốt (3–7 bullet).
4) Cơ chế / nguyên lý (nếu liên quan): giải thích ở mức vừa đủ, tránh thuật ngữ quá hàn lâm nhưng phải đúng.
5) Phân loại (CHỈ KHI câu hỏi hỏi “gồm những loại nào/bao gồm/phân loại”): kèm tiêu chí phân biệt.
6) Ví dụ đại diện: ưu tiên nhóm/case phổ biến trong canh tác Việt Nam (nêu 3–8 ví dụ phù hợp).
7) Sai lầm thường gặp & cách tránh (2–5 ý) — chỉ nêu khi giúp ích trực tiếp.
8) Câu hỏi cần làm rõ (2–6 câu): để chốt quyết định thực tế theo bối cảnh người dùng.

QUY TẮC TRẢ LỜI:
- Không lan man sang chủ đề ngoài trọng tâm câu hỏi.
- Không “tỏ ra chắc chắn” khi thiếu cơ sở; không suy diễn vượt quá thông tin đầu vào.
- Dùng thuật ngữ BVTV quen thuộc tại Việt Nam; nếu dùng thuật ngữ quốc tế thì giải thích ngắn kèm theo.
- Văn phong chuyên nghiệp, dễ hiểu; ưu tiên ví dụ và tiêu chí phân biệt hơn là lý thuyết dài dòng.
""".strip()

def answer_with_suggestions(*, user_id, user_query, kb, client, cfg, policy):
    timer = TimingLog(user_query)

    # =====================================================
    # PRE-STEP: SHORT-TERM CONTEXT + QUERY REWRITE
    # =====================================================
    turns = conversation_state.get_turns(user_id)
    effective_query = user_query

    if turns and needs_rewrite(user_query):
        history_text = format_history(turns)
        try:
            rewritten = rewrite_query_with_llm(
                client=client,
                user_query=user_query,
                history_text=history_text,
            )
            if rewritten:
                effective_query = rewritten
        except Exception as e:
            print("[QUERY REWRITE ERROR]:", e)

    # =====================================================
    # 0) ROUTER – QUYỀN CAO NHẤT (DÙNG effective_query)
    # =====================================================
    route = route_query(client, effective_query)
    timer.start("normalize")
    norm_query = normalize_query(client, effective_query)
    norm_lower = norm_query.lower()
    timer.end("normalize")

    # =====================================================
    # MEMORY – READ (LONG-TERM, BEFORE REASONING / HOP0)
    # =====================================================
    timer.start("read_memory and check route")
    memory_facts = read_memory(
        client=client,
        user_id=user_id,
        query=norm_query,
    )

    memory_prompt = ""
    if memory_facts:
        memory_prompt = "USER MEMORY:\n" + "\n".join(
            f"- {m['fact']}" for m in memory_facts
        )

    force_rag = any(k in norm_lower for k in FORMULA_TRIGGERS)
    if force_rag:
        route = "RAG"

    print("route:", route)
    timer.end("read_memory and check route")

    # =====================================================
    # 1) NHÁNH GLOBAL
    # =====================================================
    if route == "GLOBAL":
        model = "gpt-4.1"
        resp = client.chat.completions.create(
            model=model,
            temperature=0.25,
            max_completion_tokens=3500,
            messages=[
                {"role": "system", "content": _global_system_prompt()},
                {"role": "user", "content": effective_query},
            ],
        )
        text = resp.choices[0].message.content.strip()

        # --- short-term append ---
        conversation_state.append(user_id, "user", user_query)
        conversation_state.append(user_id, "assistant", text)
        return {
            "text": text,
            "img_keys": [],
            "route": "GLOBAL",
            "norm_query": "",
            "strategy": f"GLOBAL/{model}",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    # =====================================================
    # 2) TAG FILTER
    # =====================================================
    timer.start("checking listing and tag filter")
    is_list = is_listing_query(norm_query)
    result = tag_filter_pipeline(norm_query)
    timer.end("checking listing and tag filter")

    must_tags = result.get("must", [])
    any_tags = result.get("any", [])

    # =====================================================
    # 3) RETRIEVAL
    # =====================================================
    if is_formula_query(norm_query, result):
        timer.start("is_formula_query running")
        hits = formula_mode_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            must_tags=must_tags,
        )
        timer.end("is_formula_query running")
    else:
        timer.start("multi_hop_controller running")
        hits = multi_hop_controller(
            client=client,
            kb=kb,
            base_query=norm_query,
            must_tags=must_tags,
            any_tags=any_tags,
        )
        timer.end("multi_hop_controller running")

    print("QUERY      :", norm_query)
    print("MUST TAGS  :", must_tags)
    print("ANY TAGS   :", any_tags)
    debug_log("QUERY      :", norm_query)
    debug_log("MUST TAGS  :", must_tags)
    debug_log("ANY TAGS   :", any_tags)

    timer.start("build_context running")
    hits = preserve_search_order(hits)

    if not hits:
        return {
            "text": "Không tìm thấy dữ liệu phù hợp.",
            "img_keys": [],
            "route": "RAG",
            "norm_query": norm_query,
        }

    # for h in hits:
    #     h["fused_score"] = fused_score(h)
    #     h["tag_hits"] = _count_tag_hits(h, any_tags, must_tags)

    primary_doc = hits[0]

    # =====================================================
    # 4) BUILD CONTEXT + GENERATE
    # =====================================================
    base_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)
    max_ctx = min(RAGConfig.max_ctx_strict, base_ctx)

    context = build_context_from_hits(hits[:max_ctx])
    timer.end("build_context running")

    timer.start("call_finetune_with_context running")
    policy = decide_answer_policy(
        effective_query, primary_doc, force_listing=is_list
    )
    answer_mode_final = (
        "listing" if policy.format == "listing" else policy.intent
    )

    final_answer = call_finetune_with_context(
        system_prefix=memory_prompt,
        client=client,
        user_query=effective_query,
        context=context,
        answer_mode=answer_mode_final,
        rag_mode="STRICT",
    )

    final_answer = enrich_answer_if_needed(
        client=client,
        user_query=effective_query,
        answer_text=final_answer,
        answer_mode=answer_mode_final,
        any_tags=any_tags,
        must_tags=must_tags,
        route="RAG",
    )
    timer.end("call_finetune_with_context running")
    # =====================================================
    # SHORT-TERM + LONG-TERM MEMORY WRITE
    # =====================================================
    conversation_state.append(user_id, "user", user_query)
    conversation_state.append(user_id, "assistant", final_answer)

    log_event(user_id, "user", user_query)
    log_event(user_id, "assistant", final_answer)

    try:
        conv_text = build_conversation_text(user_id)
        facts_raw = summarize_to_fact(client, conv_text)
        facts = json.loads(facts_raw)
        write_memory(client, user_id, facts)
    except Exception as e:
        print("[MEMORY WRITE ERROR]:", e)

    timer.finish(RAGConfig.enable_timing_log)

    return {
        "text": final_answer,
        "img_keys": extract_img_keys(primary_doc.get("answer", "")),
        "route": "RAG",
        "norm_query": norm_query,
        "context_build": context,
    }
