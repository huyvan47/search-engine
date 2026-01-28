- Gọi router.py lấy đầu ra GLOBAL, RAG. 
 + Các tín hiệu mạnh đi RAG: tên công ty, từ khóa thuộc nhóm product_context, treatment_intent, definition_signals, các tag"if force_rag_by_tags", rồi mới qua khâu xét "GLOBAL", nếu khâu này chưa đạt thì hệ thống mặc định dùng "RAG"
 + Các tín hiệu mạnh đi GLOBAL: có tên hoạt chất nhưng không đi kèm các tín hiệu sản phẩm trong product_intent, vượt qua khâu kiểm tra tín hiệu giáo trình "definition_signals"

- Chuẩn hóa câu hỏi -> không dấu
- Nếu router == "GLOBAL" đi GLOBAL set model "gpt-4.1" -> return kết quả 

- So khớp cứng với "FORMULA_TRIGGERS" nếu True -> RAG

- Kiểm tra cứng "is_listing_query" xem có phải listing không: Kiểm tra tín hiệu người dùng muốn liệt kê nhiều sản phẩm hay không.

- Chạy phân tích tag "tag_filter_pipeline" -> chuẩn hóa query với normalize -> phân tách tag với KB tag .json. Với các product, brand, formulation, formula phải vào must, còn lại vào any. Còn soft_tags dành riêng cho các hoạt chất (phục vụ các câu hỏi tìm công thức, ví dụ "tiếp xúc + lưu dẫn").
 + Chuẩn hóa câu hỏi
 + Phân tách tags với "extract_tags", nhận về must tag, any tag khớp với KB tag .json

- Kiểm tra xem có phải câu hỏi dạng công thức hay không với "is_formula_query" nếu đúng -> chạy mode tìm hits với "formula_mode_search" đặc thù (quét tags must), nếu không gọi "multi_hop_controller". Thiết kế đặc thù khi phát hiện tín hiệu công thức, các tính chất thuốc như "lưu dẫn mạnh, tiếp xúc, ..." các tag này vào must để bắt đầu quét doc liên quan đến tag với retrieve_search (chỉ lấy PASS) lần lượt (được chia đều ngân sách "per_tag_k" tổng doc sẽ đi LLM suy luận)
 + Các hits từ các tag riêng biệt sẽ được trả về và gom vào list.
 + Kiểm tra số lượng doc nếu quét toàn bộ tag rồi vẫn thiếu thì cho quét vòng cuối với free tag nhằm quét các doc có điểm SIM mạnh và lắp đầy danh sách DOC gửi LLM suy luận.

- Chạy "multi_hop_controller" quét các hits với cơ chế multi hop, max 3 lần gọi hop, max 25 doc tìm được.

- Pick doc chính
- Building context:
 + Quyết định bao nhiêu doc gửi LLM để build với khung "build_context_from_hits", trước đó check số lượng với "choose_adaptive_max_ctx" ,...
 + Gọi call_finetune_with_context
=> Làm giàu thông tin cho câu trả lời nếu thiếu thông tin với "enrich_answer_if_needed"