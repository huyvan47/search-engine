import openai
import os
import sys, json
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policies.v7_policy import PolicyV7
from flask import Flask, session, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
from flask import render_template
from run.main import BASE_DIR, KB
from rag.logging.logger_csv import append_log_to_csv  # Import CSV logging function
from logger_img_csv.logger_img import log_image_analysis  # Import log_image_analysis function
from rag.config import RAGConfig
from rag.kb_loader import load_npz
from rag.pipeline import answer_with_suggestions
openai.api_key = '...'

# thêm path để import module rag
CSV_PATH = "D:/Huy/Project/programing/search-engine/search-engine/rag_logs.csv"
BASE_DIR = Path(__file__).resolve().parent
# Cấu hình Flask
app = Flask(__name__)

app.secret_key = "BmcVN!2023#"

# Thư mục lưu ảnh tải lên
UPLOAD_FOLDER = BASE_DIR / 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}
# Giới hạn file upload tối đa 16MB (có thể tăng tuỳ nhu cầu)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Tạo thư mục uploads nếu chưa có
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

def get_user_id():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]

# Hàm kiểm tra định dạng tệp hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def build_product_query_from_diag(diag: dict) -> str:
    """
    Chuyển kết quả JSON diagnose thành câu hỏi chuẩn cho RAG
    """

    crop = diag.get("crop", "cây trồng")
    diagnoses = diag.get("top_diagnoses", [])

    if not diagnoses:
        return f"Cây này là {crop}, vậy sản phẩm nào phù hợp để bảo vệ cây?"

    disease_name = diagnoses[0].get("name", "sâu bệnh")

    query = f"Cây này có khả năng là {crop} bị {disease_name}, vậy sản phẩm nào có thể trị được {disease_name} hiệu quả?"

    return query


def analyze_image_with_gpt4v(image_path): # Phân tích ảnh với GPT-4 Vision
    import base64
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # prompt hướng dẫn GPT-4 Vision phân tích ảnh
    prompt = """
    Bạn là CHUYÊN GIA NÔNG NGHIỆP & BẢO VỆ THỰC VẬT tại Việt Nam.

    NHIỆM VỤ CỦA BẠN:
    - Tôi cung cấp cho bạn MỘT HÌNH ẢNH cây trồng do khách hàng gửi.
    - Bạn chỉ được quan sát trực tiếp nội dung hình ảnh để suy luận.
    - Dựa trên những gì NHÌN THẤY, hãy đưa ra dự đoán hợp lý nhất về:
    + cây trồng trong ảnh
    + loại sâu hoặc bệnh có khả năng đang xảy ra

    NGUYÊN TẮC BẮT BUỘC:

    1) CHỈ được dựa trên dấu hiệu quan sát được trong ảnh.
    - Không suy diễn ngoài hình ảnh.
    - Không bịa thêm thông tin.

    2) CHỈ đưa ra TỐI ĐA 1 chẩn đoán có khả năng cao nhất.
    - Không liệt kê nhiều khả năng.
    - Không đưa ra danh sách dài.

    3) Nếu không chắc chắn về cây trồng:
    - Được phép dùng cụm: "có khả năng là cây ..."

    4) TUYỆT ĐỐI KHÔNG:
    - Không hỏi thêm câu hỏi phụ.
    - Không đặt câu hỏi khảo sát người dùng.
    - Không đưa ra hướng dẫn kỹ thuật hay liều lượng.

    5) QUAN TRỌNG NHẤT – NGÔN NGỮ:
    - BẮT BUỘC trả lời hoàn toàn bằng TIẾNG VIỆT.
    - Tất cả tên sâu bệnh phải được diễn giải sang tiếng Việt phổ biến tại Việt Nam.
    - KHÔNG ĐƯỢC giữ nguyên bất kỳ cụm từ tiếng Anh nào trong câu trả lời.
    - Nếu trong chuyên môn tồn tại nhiều cách gọi, hãy dùng tên gọi tiếng Việt thông dụng nhất.

    QUY ĐỊNH VỀ ĐỊNH DẠNG TRẢ LỜI – PHẢI TUÂN THỦ 100%:

    Bạn CHỈ được trả lời đúng 1 câu duy nhất theo đúng khuôn sau, không thêm bớt:

    "Cây này có khả năng là [tên cây bằng tiếng Việt] bị [tên sâu hoặc bệnh bằng tiếng Việt], vậy sản phẩm nào có thể trị được [đúng tên sâu hoặc bệnh đó bằng tiếng Việt] hiệu quả?"

    YÊU CẦU QUAN TRỌNG:
    - Câu trả lời phải luôn kết thúc bằng cụm: 
    "vậy sản phẩm nào có thể trị được ... hiệu quả?"

    - Không được tự ý đổi cấu trúc câu.
    - Không được thêm bất kỳ câu hỏi phụ nào khác.
    - Không được sử dụng tiếng Anh.

    VÍ DỤ HỢP LỆ:

    "Cây này có khả năng là cây bắp cải bị sâu ăn lá, vậy sản phẩm nào có thể trị được sâu ăn lá hiệu quả?"

    "Cây này có khả năng là cây ớt bị bệnh thán thư, vậy sản phẩm nào có thể trị được bệnh thán thư hiệu quả?"
    """


    # Gọi GPT-4 Vision API
    response = openai.chat.completions.create(
        model="gpt-4.1",  # Dùng mô hình 4.1 để bắt hình ảnh tốt hơn
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content # Trả về mô tả phân tích ảnh

DIAG_SCHEMA_HINT = """{
  "crop":"unknown",
  "organ":"unknown",
  "symptoms":[],
  "top_diagnoses":[
    {
      "name":"",
      "pathogen_type":"unknown",
      "confidence":0.0,
      "rationale":[],
      "differentials":[]
    }
  ],
  "questions_to_confirm":[],
  "red_flags":[]
}"""


def diagnose_from_image(client, q: str, image_base64: str) -> dict:
    """
    Hàm mới: dùng khi có đồng thời cả câu hỏi và hình ảnh
    Trả về JSON chẩn đoán theo schema
    """

    system_prompt = (
        "Bạn là chuyên gia chẩn đoán bệnh cây dựa trên hình ảnh (triage).\n"
        "Nhiệm vụ: đề xuất các chẩn đoán có khả năng nhất và thông tin cần hỏi thêm để xác nhận.\n\n"
        "Quy tắc:\n"
        "- ĐƯỢC phép nêu tên bệnh cụ thể và loại tác nhân (nấm/vi khuẩn/sâu...).\n"
        "- KHÔNG tư vấn thuốc, sản phẩm, liều lượng, thời gian phun, hay quy trình xử lý.\n"
        "- Trả về DUY NHẤT JSON theo schema yêu cầu, không kèm chữ thừa.\n"
        "- Đưa ra tối đa 3 chẩn đoán, sắp theo độ tin cậy giảm dần.\n"
        "- Nếu không đủ chắc, hạ confidence và đưa câu hỏi xác nhận.\n\n"
        f"Schema mẫu (chỉ để tham chiếu):\n{DIAG_SCHEMA_HINT}"
    )

    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
    )

    text = response.choices[0].message.content.strip()
    return json.loads(text)

 # Gọi hệ thống RAG để trả lời câu hỏi dựa trên mô tả hình ảnh
def query_rag_system(user_id, kb, API_KEY, query): 
    """
    Gọi vào hệ thống RAG thực tế để trả lời câu hỏi dựa trên mô tả hình ảnh.
    Hệ thống RAG có thể truy vấn cơ sở dữ liệu hoặc sử dụng các mô hình tạo câu trả lời.
    """
    # Tích hợp openai.OpenAI với hệ thống RAG thực tế
    user_query = query
    openai.api_key = API_KEY
    client = openai
    KB = "D:/Huy/Project/programing/search-engine/search-engine/data-kd-1-4-25-1-2026-focus-product.npz"
    kb = load_npz(KB)   
    cfg = RAGConfig()
    policy = PolicyV7()
    answer = answer_with_suggestions(
        user_id=user_id,
        user_query=user_query,
        kb=kb,
        client=client,
        cfg=cfg,
        policy=policy,
        )
    return answer # Trả về kết quả từ hệ thống RAG


# Gửi kết quả phân tích vào RAG để trả lời
def process_with_rag( user_id, description):
    if not description:
        return {"error": "Không có mô tả để xử lý."}

    result_from_rag = query_rag_system(user_id, KB, openai.api_key, description)

    # Bảo vệ nếu RAG trả về None
    if result_from_rag is None:
        return {"error": "Hệ thống RAG không trả về kết quả."}

    return result_from_rag

# Ghi log vào CSV cho phân tích ảnh
def log_image_analysis_result(image_path, image_description):
    log_image_analysis(image_path, image_description)  # Ghi mô tả ảnh vào file CSV

# Ghi log vào CSV cho kết quả RAG
def log_to_csv(image_path, image_description, rag_result, user_query, norm_query, context_build, strategy, profile):
    append_log_to_csv(
        'analyze_img_log.csv', 
        image_path, 
        image_description, 
        rag_result, 
        user_query,
        norm_query,
        context_build, # Thêm context_build trường mới
        strategy,
        profile
    )

def encode_image_to_base64(image_path):
    import base64
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode("utf-8")

@app.route('/upload', methods=['POST'])
def upload_image():          # Xử lý upload ảnh và câu hỏi từ người dùng
    user_id = get_user_id()
    result = None
    user_query = request.form.get('query', '').strip() if 'query' in request.form else ''
    print("[DEBUG] request.files:", request.files)
    print("[DEBUG] request.form:", request.form)
    if 'file' not in request.files and not user_query:
        print("[ERROR] No file and no query provided.")
        return jsonify({"error": "Bạn cần nhập nội dung hoặc tải lên hình ảnh."}), 400 # Kiểm tra nếu không có file và không có câu hỏi
    has_file = 'file' in request.files and request.files['file'].filename != ''
    image_description = None
    file_path = None
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            print("[ERROR] No file selected.")
            if not user_query:
                return jsonify({"error": "Không có file hình ảnh được chọn."}), 400
        elif not allowed_file(file.filename):
            print("[ERROR] File type not allowed.")
            return jsonify({"error": "Định dạng file không hợp lệ. Chỉ chấp nhận png, jpg, jpeg, gif."}), 400
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_description = analyze_image_with_gpt4v(file_path)  # Phân tích ảnh với GPT-4 Vision
            log_image_analysis_result(file_path, image_description)  # Ghi log mô tả ảnh vào CSV
        if has_file and user_query:
            # Encode ảnh sang base64
            image_base64 = encode_image_to_base64(file_path)

            # Gọi luồng chẩn đoán chuyên biệt khi có cả text + ảnh
            diag_json = diagnose_from_image(
                client=openai,
                q=user_query,
                image_base64=image_base64
            )

            product_query = build_product_query_from_diag(diag_json)

            print("[DEBUG] Product Query:", product_query)

            result = process_with_rag(user_id, product_query)

    elif has_file:
        result = process_with_rag(user_id, image_description)
    elif user_query:
        result = process_with_rag(user_id, user_query)
    else:
        return jsonify({"error": "Bạn cần nhập nội dung hoặc tải lên hình ảnh."}), 400
    try:
        if not result:
            result = {}

        if isinstance(result, dict):
            profile = result.get('profile', {})
            norm_query = result.get('norm_query', "")
            strategy = result.get('strategy', "")
            route = result.get('route', "RAG")
            context_build = result.get('context_build', "")
        else:
            profile = {}
            norm_query = ""
            strategy = ""
            route = "RAG"
        context_build = ""
        # Lấy context_build nếu có, nếu không thì truyền rỗng
        context_build = result.get('context_build', "") if isinstance(result, dict) else ""
        append_log_to_csv( # Ghi log kết quả RAG vào CSV
            CSV_PATH,
            user_query if user_query else (image_description if image_description else ""),
            norm_query,
            context_build,
            strategy,
            profile,
            result,
            route,
        )
        return jsonify({"result": result})
    except Exception as e:
        print(f"[LOG CSV ERROR]: {e}")

        if result is None:
            result = "Không thể xử lý yêu cầu do lỗi hệ thống."

        return jsonify({"result": result})

@app.route('/')
def index():
    return render_template('./index.html') 

if __name__ == '__main__':
    app.run(debug=True) # Chạy Flask ở chế độ debug
    
    
    