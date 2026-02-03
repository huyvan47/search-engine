import re
from rag.post_answer.chemical_extractor import (
    extract_chemicals_from_matched_tags,
    extract_primary_chemical_from_answer,
)
from rag.post_answer.global_knowledge import query_global_knowledge
from rag.post_answer.detectors import should_enrich_post_answer
from rag.retriever import search as retrieve_search
from rag.tag_filter import normalize_entity, PRODUCT_ALIASES, tag_filter_pipeline
from typing import List, Tuple, Dict, Any



def enrich_answer_if_needed(
    *,
    client,
    user_query: str,
    answer_text: str,
    answer_mode: str,
    any_tags: list,
    must_tags: list,
    route: str = "RAG",
) -> str:

    if not should_enrich_post_answer(
        user_query=user_query,
        answer_text=answer_text,
        answer_mode=answer_mode,
        route=route,
    ):
        return answer_text

    blocks = []

    # =========================
    # MODE 1: FORMULA → enrich theo CHEMICAL (giữ logic cũ)
    # =========================
    if answer_mode == "formula":
        # 1️⃣ ƯU TIÊN: chemical từ tags
        chemicals = extract_chemicals_from_matched_tags(
            any_tags=any_tags,
            must_tags=must_tags,
        )

        # 2️⃣ Fallback: từ answer text
        if not chemicals:
            c = extract_primary_chemical_from_answer(answer_text)
            if c:
                chemicals = [c]

        if not chemicals:
            return answer_text

        for chem in chemicals[:1]:  # formula → 1 chemical là đủ
            try:
                info = query_global_knowledge(
                    client=client,
                    chemical=chem,
                    user_query=user_query,
                    model="gpt-4.1",
                    max_tokens=900,
                )
                if info:
                    blocks.append(
                        f"### Kiến thức nền về hoạt chất {chem.upper()} (ngoài tài liệu nội bộ)\n"
                        f"{info.strip()}\n"
                    )
            except Exception:
                continue

    # =========================
    # MODE 2: PRODUCT → enrich theo CÂU HỎI (không gắn nhãn chemical)
    # =========================
    elif answer_mode == "product":
        try:
            info = query_global_knowledge(
                client=client,
                chemical=user_query,   # không ép chemical
                user_query=user_query,
                model="gpt-4.1",
                max_tokens=900,
            )
            if info:
                blocks.append(info.strip())
        except Exception:
            pass

    # =========================
    # Không có gì để enrich
    # =========================
    if not blocks:
        return answer_text

    # =========================
    # Appendix chung
    # =========================
    appendix = (
        "\n\n---\n\n"
        "## BỔ SUNG KIẾN THỨC NỀN (tham khảo khoa học, ngoài tài liệu nội bộ)\n"
        "Lưu ý: Phần dưới đây là kiến thức nền mang tính tham khảo, "
        "không thay thế hướng dẫn trên nhãn/khuyến cáo nhà sản xuất.\n\n"
        + "\n".join(blocks)
    )

    return answer_text.rstrip() + appendix

def detect_products_from_text(text: str) -> List[str]:
    """
    Trả về list product keys (canonical ids) xuất hiện trong answer
    """
    norm_text = normalize_entity(text)

    hits = []

    for product_key, aliases in PRODUCT_ALIASES.items():
        for a in aliases:
            na = normalize_entity(a)
            if na and na in norm_text:
                hits.append(product_key)
                break

    return list(set(hits))

def summarize_product_from_hits(product, hits):
    """
    Tìm hoạt chất + cơ chế từ các doc trong hits.
    """
    actives = set()
    mechs = set()

    for h in hits:
        blob = " ".join([
            str(h.get("question","")),
            str(h.get("answer","")),
            str(h.get("tags_v2","")),
        ]).lower()

        if product.lower() in blob:
            tv2 = str(h.get("tags_v2",""))
            for part in tv2.split("|"):
                if part.startswith("chemical:"):
                    actives.add(part.split(":",1)[1])
                if part.startswith("mechanisms:"):
                    mechs.add(part.split(":",1)[1])

    return list(actives), list(mechs)


def enrich_answer_with_product_knowledge(
    *, 
    client,
    kb,
    answer_text: str,
    hits: list,
    top_k: int = 5
) -> str:
    """
    Hậu xử lý câu trả lời: gắn hoạt chất + cơ chế cho sản phẩm nếu thiếu.
    """

    products = detect_products_from_text(answer_text)
    if not products:
        return answer_text

    enriched = answer_text

    for p in products:
        actives, mechs = summarize_product_from_hits(p, hits)

        # Nếu chưa đủ → search bổ sung riêng cho product
        if not actives and not mechs:
            tag_result = tag_filter_pipeline(p)
            hits_p = retrieve_search(
                client=client,
                kb=kb,
                norm_query=p,
                top_k=top_k,
                must_tags=tag_result.get("must", []),
                any_tags=tag_result.get("any", []),
            )
            actives, mechs = summarize_product_from_hits(p, hits_p)

        if actives or mechs:
            desc = []
            if actives:
                desc.append(", ".join(actives))
            if mechs:
                desc.append("cơ chế: " + ", ".join(mechs))

            enrich_str = f"{p} ({' – '.join(desc)})"

            # thay thế nếu trong answer có đoạn mơ hồ
            enriched = re.sub(
                rf"{re.escape(p)}\s*\([^)]*không ghi rõ[^)]*\)",
                enrich_str,
                enriched,
                flags=re.IGNORECASE
            )

    return enriched