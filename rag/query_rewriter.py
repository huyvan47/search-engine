# rag/query_rewriter.py
from __future__ import annotations
from typing import List
import re

REFERENCE_PATTERNS = [
    r"\bđó\b", r"\bnày\b", r"\bkia\b",
    r"\bcái đó\b", r"\bcái này\b",
    r"\bloại đó\b", r"\bloại này\b",
    r"\bthuốc đó\b", r"\bsản phẩm đó\b",
    r"\bhoạt chất đó\b", r"\bhoạt chất này\b",
    r"\bvậy\b",  # thường mở đầu câu nối tiếp
]

def needs_rewrite(user_query: str) -> bool:
    q = user_query.strip().lower()
    if len(q) < 4:
        return False
    # Nếu có pattern tham chiếu thì rewrite
    return any(re.search(p, q) for p in REFERENCE_PATTERNS)

def format_history(turns) -> str:
    # turns: list[Turn]
    lines = []
    for t in turns[-6:]:
        role = "USER" if t.role == "user" else "ASSISTANT"
        lines.append(f"{role}: {t.content}")
    return "\n".join(lines)

def rewrite_query_with_llm(client, user_query: str, history_text: str) -> str:
    """
    Rewrite câu hỏi hiện tại thành câu ĐỘC LẬP, rõ đối tượng tham chiếu.
    Không trả lời nội dung, chỉ rewrite câu hỏi.
    """
    system = (
        "Bạn là bộ máy viết lại truy vấn (query rewriter).\n"
        "Nhiệm vụ: dựa trên hội thoại gần nhất, viết lại câu hỏi cuối thành một câu hỏi ĐỘC LẬP.\n"
        "Quy tắc:\n"
        "- Chỉ trả về 1 câu hỏi duy nhất (tiếng Việt).\n"
        "- Thay mọi đại từ tham chiếu (đó/này/cái này/hoạt chất đó...) bằng thực thể cụ thể.\n"
        "- Không trả lời, không giải thích.\n"
        "- Nếu không đủ thông tin để thay thế, giữ nguyên câu hỏi ban đầu.\n"
    )
    user = f"HỘI THOẠI GẦN NHẤT:\n{history_text}\n\nCÂU HỎI HIỆN TẠI:\n{user_query}"

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        max_completion_tokens=120,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    # Fallback nếu model trả rỗng
    return rewritten if rewritten else user_query
