#Khối import
import json
import re
from datetime import datetime
from rag.config import RAGConfig
from rag.router import route_query
from rag.normalize import normalize_query
from rag.text_utils import is_listing_query
from rag.retriever import search as retrieve_search
from rag.scoring import fused_score
from rag.context_builder import choose_adaptive_max_ctx, build_context_from_hits
from rag.memory.conversation_manager import read_memory, log_event, build_conversation_text, write_memory
from rag.memory.summarizer import summarize_to_fact
from rag.conversation_state import conversation_state
from rag.query_rewriter import needs_rewrite, format_history, rewrite_query_with_llm
from rag.answer_modes import decide_answer_policy
from rag.generator import call_finetune_with_context_stream, call_finetune_with_context, l3_draft_fast_from_kb
from rag.tag_filter import tag_filter_pipeline
from rag.logging.timing_logger import TimingLog
from rag.logging.debug_log import set_debug_dir
from rag.reasoning.multi_hop import multi_hop_controller
from typing import List, Tuple, Dict, Any
from rag.logging.debug_log import debug_log
from rag.logging.multi_query_logger import _safe_folder_name
from httpx import RemoteProtocolError
from rag.logging.logger_csv import append_log_to_csv
from rag.post_answer.solution_completion import run_solution_completion
from rag.post_answer.l3_gap_detector import detect_l3_gaps
from rag.post_answer.t5_knowledge_fallback import t5_knowledge_fallback
# from rag.post_answer.enricher import enrich_answer_if_needed
from pathlib import Path

def make_run_dir(query: str):
    base = Path("debug_runs")
    base.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = _safe_folder_name(query)

    root = base / f"{ts}__{name}"
    root.mkdir(parents=True, exist_ok=True)

    return root

def emit_trace_snapshot(*,
    user_query,
    effective_query,
    norm_query,
    must_tags,
    any_tags,
    hits,
    base_ctx,
    context,
    l3_missing_slots,
    missing_after_t4,
    t4_report,
    need_kb_fallback,
    memory_prompt,
    final_system_override,
    answer_mode,
):
    def safe_len(x):
        try:
            return len(x)
        except:
            return 0

    print("\n================ FINAL RAG TRACE =================")
    print("USER QUERY      :", user_query)
    print("EFFECTIVE QUERY :", effective_query)
    print("NORMALIZED      :", norm_query)
    print("ANSWER MODE     :", answer_mode)
    print("--------------------------------------------------")

    print("TAGS:")
    print("  MUST :", must_tags)
    print("  ANY  :", any_tags)
    print("--------------------------------------------------")

    print("RETRIEVAL:")
    print("  total hits :", len(hits))
    print("  T4 hits    :", sum(1 for h in hits if h.get("t4_origin_query")))
    print("  KB hits    :", sum(1 for h in hits if not h.get("t4_origin_query")))
    print("--------------------------------------------------")

    print("L3 / T4:")
    print("  L3 missing slots      :", l3_missing_slots)
    print("  missing after T4     :", missing_after_t4)
    print("  T4 report present    :", bool(t4_report))
    print("--------------------------------------------------")

    print("T5 FALLBACK:")
    print("  need_kb_fallback :", need_kb_fallback)
    if need_kb_fallback:
        print("  [T5] knowledge injected")
    else:
        print("  [T5] NOT triggered")
    print("--------------------------------------------------")

    print("CONTEXT:")
    print("  base ctx length :", safe_len(base_ctx))
    print("  final ctx length:", safe_len(context))
    print("  T5 added chars  :", max(0, safe_len(context) - safe_len(base_ctx)))
    print("--------------------------------------------------")

    print("SYSTEM PROMPT:")
    print("  memory chars   :", safe_len(memory_prompt))
    print("  override chars :", safe_len(final_system_override))
    print("--------------------------------------------------")

    print("MODEL INPUT SUMMARY:")
    print("  system total   :", safe_len(memory_prompt + final_system_override))
    print("  context total  :", safe_len(context))
    print("==================================================\n")


KB_FALLBACK_SLOTS = {
    "need_pesticide",
    "need_foliar_fertilizer",
    "need_mix_compatibility",
    "need_dosage_or_rate",
    "need_timing",
    "need_crop",
    "need_pest_or_disease",
    "need_general_knowledge",
}

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

#chuẩn hóa câu hỏi của người dùng
def norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

#Nhận diện truy vấn dạng phối công thức
def is_formula_query(query: str, tags: dict) -> bool:

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

#Tìm kiếm theo chế độ công thức (không dùng multi-hop)
def formula_mode_search(
    *,
    client,
    kb,
    norm_query: str,
    must_tags: List[str],
):
    """
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

#Chuẩn hóa tag
def strip_tag_ns(s):
    if not s:
        return ""
    out = []
    for part in s.split("|"):
        if ":" in part:
            out.append(part.split(":",1)[1])
        else:
            out.append(part)
    return " ".join(out)

#Chuẩn hóa tag
def strip_ns(t):
    if ":" in t:
        t = t.split(":",1)[1]
    return t

#Filter DOC match tags/keyword
def evidence_gate_by_tags(hits, must_tags, any_tags):
    must_tags = [norm(strip_ns(t)) for t in must_tags]
    any_tags  = [norm(strip_ns(t)) for t in any_tags]

    kept = []

    for h in hits:
        blob = norm(
            (h.get("question","") or "") + " " +
            (h.get("answer","") or "") + " " +
            " ".join(h.get("alt_question",[]) or []) + " " +
            strip_tag_ns(h.get("tags_v2",""))
        )

        must_hit = sum(1 for t in must_tags if t in blob)
        any_hit  = sum(1 for t in any_tags  if t in blob)

        if must_hit >= 1 or any_hit >= 1:
            kept.append((must_hit, any_hit, h))

    kept.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [h for _,_,h in kept[:RAGConfig.max_ctx_soft]]

#Đánh dấu thứ tự gốc từ search() để pipeline KHÔNG làm xáo trộn
def preserve_search_order(hits):
    for idx, h in enumerate(hits):
        h["_search_rank"] = idx
    return hits

#Cộng điểm tag match
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

#Promt phục vụ đi nhánh GLOBAL
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

#Core GPT
def answer_with_suggestions_stream(*, user_id, user_query, kb, client, cfg, policy):
    timer = TimingLog(user_query)
    run_dir = make_run_dir(user_query)
    set_debug_dir(run_dir)

    turns = conversation_state.get_turns(user_id)
    effective_query = user_query

    # 1) rewrite
    if turns and needs_rewrite(user_query):
        history_text = format_history(turns)
        try:
            rewritten = rewrite_query_with_llm(
                client=client,
                user_query=user_query,
                history_text=history_text
            )
            if rewritten:
                effective_query = rewritten
        except Exception as e:
            print("[QUERY REWRITE ERROR]:", e)

    # 2) route + normalize
    route = route_query(client, effective_query)
    if route == "GLOBAL":
        model = "gpt-4.1"
        yield "🌍 Đang trả lời bằng tri thức tổng quát...\n\n"

        resp = client.chat.completions.create(
            model=model,
            temperature=0.25,
            messages=[
                {"role": "system", "content": _global_system_prompt()},
                {"role": "user", "content": effective_query},
            ],
            stream=True,
        )

        parts = []
        for chunk in resp:
            try:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    parts.append(delta.content)
                    yield delta.content
            except Exception:
                continue

        final_text = "".join(parts)

        conversation_state.append(user_id, "user", user_query)
        conversation_state.append(user_id, "assistant", final_text)
        log_event(user_id, "user", user_query)
        log_event(user_id, "assistant", final_text)

        timer.finish(RAGConfig.enable_timing_log)
        return

    timer.start("normalize")
    norm_query = normalize_query(client, effective_query)
    norm_lower = norm_query.lower()
    timer.end("normalize")

    # 3) read memory + force_rag
    timer.start("read_memory and check route")
    memory_facts = read_memory(client=client, user_id=user_id, query=norm_query)
    memory_prompt = ""
    if memory_facts:
        memory_prompt = "USER MEMORY:\n" + "\n".join(f"- {m['fact']}" for m in memory_facts)

    force_rag = any(k in norm_lower for k in FORMULA_TRIGGERS)
    if force_rag:
        route = "RAG"
    timer.end("read_memory and check route")

    # UI status
    yield "⏳ Đang truy vấn dữ liệu...\n\n"

    # 4) tag filter
    timer.start("tag_filter_pipeline running")
    result = tag_filter_pipeline(norm_query)
    must_tags = result.get("must", [])
    any_tags  = result.get("any", [])
    timer.end("tag_filter_pipeline running")

    # 5) retrieval
    timer.start("retrieval")
    is_list = is_listing_query(norm_query)
    if is_formula_query(norm_query, {"must": must_tags, "soft": any_tags}):
        hits = formula_mode_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            must_tags=must_tags
        )
    else:
        hits = multi_hop_controller(
            client=client,
            kb=kb,
            base_query=norm_query,
            must_tags=must_tags,
            any_tags=any_tags,
        )
    timer.end("retrieval")

    print("QUERY      :", norm_query)
    print("MUST TAGS  :", must_tags)
    print("ANY TAGS   :", any_tags)
    debug_log("QUERY      :", norm_query)
    debug_log("MUST TAGS  :", must_tags)
    debug_log("ANY TAGS   :", any_tags)

    if not hits:
        yield "Không tìm thấy dữ liệu phù hợp."
        return

    # fused score, tag hits, ordering
    for h in hits:
        h["fused_score"] = fused_score(h)
        h["tag_hits"] = _count_tag_hits(h, any_tags, must_tags)

    hits = preserve_search_order(hits)
    primary_doc = hits[0]

    # 6) build context (pre T3/T4/T5)
    timer.start("build_context")
    base_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)
    max_ctx = min(RAGConfig.max_ctx_strict, base_ctx)

    policy = decide_answer_policy(effective_query, primary_doc, force_listing=is_list)

    off_filter_tag_on_doc = policy.intent not in {"disease"}
    if not off_filter_tag_on_doc:
        base_hits = [h for h in hits if not h.get("t4_origin_query")]
        t4_hits   = [h for h in hits if h.get("t4_origin_query")]
        base_hits = evidence_gate_by_tags(base_hits, must_tags=must_tags, any_tags=any_tags)
        hits = t4_hits + base_hits

    context = build_context_from_hits(hits[:max_ctx])
    timer.end("build_context")

    # ===========================
    # L3 — KB Gap Detector
    # ===========================
    t4_report = None

    timer.start("l3_draft")
    # IMPORTANT: draft trung thực từ KB (không dùng LLM để tránh che lỗ hổng KB)
    kb_draft = l3_draft_fast_from_kb(hits)
    timer.end("l3_draft")

    timer.start("l3_gap")
    gap = detect_l3_gaps(client, norm_query, kb_draft)
    timer.end("l3_gap")

    l3_missing_slots = gap.get("missing_slots", []) or []

    # ===========================
    # T4 — Solution Completion
    # ===========================
    if l3_missing_slots:
        timer.start("t4_retrieval")
        # NOTE: run_solution_completion phải return (hits, t4_report)
        hits, t4_report = run_solution_completion(
            run_dir=run_dir,
            client=client,
            kb=kb,
            user_query=norm_query,
            hits=hits,
            must_tags=must_tags,
            any_tags=any_tags,
            l3_missing_slots=l3_missing_slots,
        )
        timer.end("t4_retrieval")

        # ưu tiên doc T4
        hits.sort(key=lambda h: 1 if h.get("t4_origin_query") else 0, reverse=True)

        timer.start("build_context_t4")
        context = build_context_from_hits(hits[:max_ctx])
        timer.end("build_context_t4")

    # ===========================
    # T5 — Knowledge Fallback
    # ===========================
    # 1) Nếu T4 report nói KB không có solution → bật T5
    # 2) Hoặc nếu L3 có need_general_knowledge mà T4_report None (case bypass) → cũng bật T5
    need_kb_fallback = False

    missing_after_t4 = []
    if t4_report is not None:
        raw = t4_report.get("l3_missing_slots") or t4_report.get("missing_slots") or []
        missing_after_t4 = raw if isinstance(raw, list) else []
    else:
        missing_after_t4 = l3_missing_slots if isinstance(l3_missing_slots, list) else []

    # Chỉ cần còn need_general_knowledge là bật T5
    need_kb_fallback = any(
        slot in KB_FALLBACK_SLOTS
        for slot in missing_after_t4
    )

    final_system_override = ""
    if need_kb_fallback:
        timer.start("t5_fallback")
        kb_fallback_text = t5_knowledge_fallback(
            client=client,
            user_query=norm_query,
            missing_slots=missing_after_t4,
            context=context,
        )
        context += "\n\n[KIẾN THỨC NỀN]\n" + (kb_fallback_text or "")
        timer.end("t5_fallback")
        final_system_override = """
        ⚠️ INTERNAL DATA IS INSUFFICIENT.

        You MUST use the [KIẾN THỨC NỀN] section
        to complete the user's objective.
        Do not answer using only internal data.
        """
    yield "✍️ Đang tổng hợp câu trả lời...\n\n"

    # 7) generate streaming (FINAL)
    answer_mode_final = (
        "formula" if is_formula_query(norm_query, {"must": must_tags, "soft": any_tags}) else "default"
    )

    emit_trace_snapshot(
        user_query=user_query,
        effective_query=effective_query,
        norm_query=norm_query,
        must_tags=must_tags,
        any_tags=any_tags,
        hits=hits,
        base_ctx=build_context_from_hits(hits[:max_ctx]),  # context trước T4/T5
        context=context,
        l3_missing_slots=l3_missing_slots,
        missing_after_t4=missing_after_t4,
        t4_report=t4_report,
        need_kb_fallback=need_kb_fallback,
        memory_prompt=memory_prompt,
        final_system_override=final_system_override,
        answer_mode=answer_mode_final,
    )

    timer.start("final_gpt_ttft")
    timer.start("final_gpt_total")
    first_tok = True
    parts = []
    stream_failed = False

    try:
        for tok in call_finetune_with_context_stream(
            system_prefix=memory_prompt + final_system_override,
            client=client,
            user_query=effective_query,
            context=context,  # context đã có T4/T5 nếu có
            answer_mode=answer_mode_final,
            must_tags=must_tags,
            any_tags=any_tags,
        ):
            if first_tok:
                timer.end("final_gpt_ttft")
                first_tok = False

            parts.append(tok)
            yield tok

    except RemoteProtocolError as e:
        print("[STREAM DROPPED] OpenAI closed connection mid-stream:", e)
        stream_failed = True

    except Exception as e:
        print("[STREAM ERROR] Unknown streaming error:", e)
        stream_failed = True

    finally:
        # luôn end tổng thời gian stream
        timer.end("final_gpt_total")

    # -------------------------------------------------
    # Fallback: re-run in NON-STREAM mode if stream died
    # -------------------------------------------------
    if stream_failed:
        print("[STREAM RECOVERY] Re-running in non-stream mode")
        try:
            final_answer = call_finetune_with_context(
                client=client,
                user_query=user_query,
                context=context,
                answer_mode=answer_mode_final,
                rag_mode="STRICT",
            )
            if parts:
                yield "\n\n[⚠ Kết nối bị gián đoạn – tiếp tục kết quả đầy đủ]\n\n"
            yield final_answer
            final_answer = "".join(parts) + final_answer
        except Exception as e:
            print("[STREAM RECOVERY FAILED]:", e)
            final_answer = "".join(parts)
    else:
        final_answer = "".join(parts)

    # 8) log CSV
    try:
        append_log_to_csv(
            run_dir,
            user_query,
            norm_query,
            context,
            {
                "text": final_answer,
                "route": route,
                "norm_query": norm_query,
                "missing_slots": l3_missing_slots,
            },
            route
        )
    except Exception as e:
        print("[RAG CSV LOG ERROR]:", e)

    # 9) memory/log
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
