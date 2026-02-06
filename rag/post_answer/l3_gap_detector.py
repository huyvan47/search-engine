import json
from typing import Dict, Any


def detect_l3_gaps(client, user_query: str, answer_text: str) -> Dict[str, Any]:
    """
    L3 Gap Detector:
    Nhận câu hỏi + câu trả lời hiện tại
    Trả về missing_slots nếu chưa đủ để hành động
    """

    sys = """
Bạn là bộ đánh giá "ĐỦ ĐỂ GIẢI QUYẾT MỤC TIÊU" cho hệ thống RAG nông nghiệp.

Input:
- user_query: câu hỏi của người dùng
- answer_text: câu trả lời hiện tại do hệ thống tạo ra (dựa trên KB)

Nhiệm vụ của bạn KHÔNG phải là kiểm tra tính đúng sai của dữ liệu.
Nhiệm vụ của bạn là đánh giá:
→ "Sau khi đọc answer_text, người dùng có đạt được MỤC TIÊU THỰC TẾ mà họ đang tìm hay chưa?"

=====================================
CÁC QUY TẮC BẮT BUỘC
=====================================

1) Nếu answer_text chỉ nói rằng:
   - "tài liệu không đề cập"
   - "không có thông tin"
   - "không có sản phẩm"
   - "chưa ghi nhận"
   - hoặc chỉ phủ định trạng thái dữ liệu
   mà KHÔNG đưa ra giải pháp thay thế hoặc hướng dẫn thực tế,
   thì coi là:
   → is_complete = false

2) Nếu user_query là câu hỏi KIẾN THỨC (ví dụ: "khi nào", "vì sao", "giai đoạn nào", "tác hại", "cơ chế"),
   mà answer_text chỉ trả lời "không có trong tài liệu",
   thì người dùng vẫn CHƯA đạt được mục tiêu.
   → is_complete = false
   → missing_slots phải chứa: "need_general_knowledge"

3) Nếu user_query có dạng:
   - "vừa A vừa B"
   - "kết hợp"
   - "công thức"
   - "nên làm gì"
   - "xử lý thế nào"
   thì answer_text phải chứa ÍT NHẤT MỘT trong các loại:
   - sản phẩm cụ thể
   - cách kết hợp
   - phương án thực hiện
   Nếu không có → is_complete = false

4) Chỉ khi answer_text giúp người dùng:
   - biết dùng gì
   - hoặc làm gì
   - hoặc có phương án thay thế khả thi
   thì mới được coi là is_complete = true.

=====================================
SLOTS BẠN ĐƯỢC PHÉP DÙNG
=====================================
- need_herbicides
- need_pesticide
- need_foliar_fertilizer
- need_mix_compatibility
- need_dosage_or_rate
- need_timing
- need_crop
- need_pest_or_disease
- need_general_knowledge   ← (bắt buộc dùng khi KB không có kiến thức sinh học)

=====================================
VÍ DỤ
=====================================

User: "Sâu lông bùng phát mạnh vào giai đoạn nào?"
Answer: "Tài liệu không đề cập."
→ is_complete = false
→ missing_slots = ["need_general_knowledge"]

User: "Thuốc nào vừa trị nhện vừa làm phân bón lá?"
Answer: "Không có sản phẩm nào trong dữ liệu."
→ is_complete = false
→ missing_slots = ["need_foliar_fertilizer", "need_mix_compatibility"]

User: "Sản phẩm trị ruồi vàng"
Answer: "A, B, C"
→ is_complete = true
→ missing_slots = []

=====================================
CHỈ TRẢ JSON THEO SCHEMA
=====================================
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
