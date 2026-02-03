import openai
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import os
BASE_DIR = Path(os.environ["BMCVN_BASE"])
import sys, json
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policies.v7_policy import PolicyV7
from flask import Flask, session, request, jsonify, Response, stream_with_context, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
from run.main import BASE_DIR, KB
from rag.logging.logger_csv import append_log_to_csv  # Import CSV logging function
from logger_img_csv.logger_img import log_image_analysis  # Import log_image_analysis function
from rag.config import RAGConfig
from rag.kb_loader import load_npz
from rag.pipeline import answer_with_suggestions_stream
openai.api_key = '...'

# thêm path để import module rag
BASE_DIR = Path(os.getenv("BMCVN_BASE", Path(__file__).parent.parent))

UPLOAD_FOLDER = BASE_DIR / "uploads"
CSV_PATH = BASE_DIR / "rag_logs.csv"
KB_PATH = BASE_DIR / "data-kd-1-4-1-2-2026-focus-product.npz"

# Cấu hình Flask
app = Flask(__name__)

app.secret_key = "BmcVN!2023#"

# Thư mục lưu ảnh tải lên
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

def process_with_rag_stream(user_id, description):
    if not description:
        yield "Không có mô tả để xử lý."
        return

    user_query = description
    openai.api_key = openai.api_key
    client = openai
    kb = load_npz(str(KB_PATH))
    cfg = RAGConfig()
    policy = PolicyV7()

    full_text = ""

    # gọi streaming
    for chunk in answer_with_suggestions_stream(
        user_id=user_id,
        user_query=user_query,
        kb=kb,
        client=client,
        cfg=cfg,
        policy=policy,
    ):
        full_text += chunk
        yield chunk

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

@app.route('/upload_stream', methods=['POST'])
def upload_stream():
    user_id = get_user_id()
    user_query = request.form.get('query', '').strip() if 'query' in request.form else ''

    if 'file' not in request.files and not user_query:
        return Response("Bạn cần nhập nội dung hoặc tải lên hình ảnh.", status=400, mimetype="text/plain; charset=utf-8")

    has_file = 'file' in request.files and request.files['file'].filename != ''
    image_description = None
    file_path = None

    # --- xử lý file giống /upload ---
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            if not user_query:
                return Response("Không có file hình ảnh được chọn.", status=400, mimetype="text/plain; charset=utf-8")
        elif not allowed_file(file.filename):
            return Response("Định dạng file không hợp lệ. Chỉ chấp nhận png, jpg, jpeg, gif.", status=400, mimetype="text/plain; charset=utf-8")
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_description = analyze_image_with_gpt4v(file_path)
            log_image_analysis_result(file_path, image_description)

    # --- build query cuối giống /upload ---
    final_query = None
    if has_file and user_query:
        image_base64 = encode_image_to_base64(file_path)
        diag_json = diagnose_from_image(client=openai, q=user_query, image_base64=image_base64)
        product_query = build_product_query_from_diag(diag_json)
        final_query = product_query
    elif has_file:
        final_query = image_description
    else:
        final_query = user_query
    @stream_with_context
    def generate():
        # headers “đẩy stream” nhanh (quan trọng nếu reverse proxy)
        yield ""  # kickstart

        # stream từ pipeline
        for chunk in process_with_rag_stream(user_id, final_query):
            yield chunk

    resp = Response(generate(), mimetype="text/plain; charset=utf-8")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"  # nginx: tắt buffer nếu có
    return resp

@app.route('/')
def index():
    return render_template('./index.html') 

if __name__ == '__main__':
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True,
        threaded=True
    )
    
    
    