import json
from typing import Dict, Any


def detect_l3_gaps(client, user_query: str, answer_text: str) -> Dict[str, Any]:
    """
    L3 Gap Detector:
    Nhận câu hỏi + câu trả lời hiện tại
    Trả về missing_slots nếu chưa đủ để hành động
    """

    sys = """
Bạn là bộ kiểm tra "đủ để hành động" cho hệ thống RAG nông nghiệp.

Input:
- user_query: câu hỏi của người dùng
- answer_text: câu trả lời hiện tại

Nhiệm vụ:
1) Nếu answer_text đã trả lời đúng và đủ theo user_query (ví dụ: user chỉ hỏi danh sách sản phẩm và answer_text đã liệt kê rõ ràng),
   thì trả is_complete=true và missing_slots=[].

2) Nếu user_query yêu cầu hành động (ví dụ: "công thức", "nên phun thế nào", "xử lý ra sao", "liều bao nhiêu")
   mà answer_text thiếu thông tin để hành động (liều, phối, thời điểm, sản phẩm cụ thể, ...),
   thì trả is_complete=false và liệt kê missing_slots.

Slots cho phép:
- need_pesticide
- need_foliar_fertilizer
- need_mix_compatibility
- need_dosage_or_rate
- need_timing
- need_crop
- need_pest_or_disease

Chỉ trả JSON hợp lệ theo schema:
{
  "is_complete": boolean,
  "missing_slots": [string, ...],
  "reason": string
}
"""

    payload = {
        "user_query": user_query,
        "answer_text": answer_text
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )

        data = json.loads(resp.choices[0].message.content)
        data.setdefault("is_complete", True)
        data.setdefault("missing_slots", [])
        data.setdefault("reason", "")

        if not isinstance(data["missing_slots"], list):
            data["missing_slots"] = []
        print("l3-data: ", data)
        return data

    except Exception:
        return {"is_complete": True, "missing_slots": [], "reason": "gap_detector_exception"}
