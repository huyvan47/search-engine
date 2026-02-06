from rag.config import RAGConfig

def build_context_from_hits(hits_for_ctx: list) -> str:
    blocks = []
    for i, h in enumerate(hits_for_ctx, 1):
        block = (
            f"[DOC {i}]\n"
            f"CÂU HỎI: {h.get('question','')}\n"
            f"HỎI KHÁC: {h.get('alt_question','')}\n"
            f"NỘI DUNG:\n{h.get('answer','')}"
        )
        # print('block: ', block)
        blocks.append(block)
    return "\n\n--------------------\n\n".join(blocks)
