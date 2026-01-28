
let chatBox = document.getElementById("chatBox");
let userInput = document.getElementById("userInput");

// Tự động mở rộng textarea khi nhập nhiều dòng
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Xử lý Enter để gửi, Shift+Enter để xuống dòng
userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('chatForm').dispatchEvent(new Event('submit', {cancelable: true, bubbles: true}));
    }
});
let loadingText = document.getElementById("loadingText");
let filePreviewContainer = document.getElementById("filePreviewContainer");
let fileInput = document.getElementById("fileInput");
let sendBtn = document.querySelector(".send-btn");
let uploadLabel = document.querySelector(".upload-label");
let uploadedFile = null;
let isLoading = false;

function appendMessage(message, sender, imageUrl=null, fileName=null) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
    let html = `<p>${message}</p>`;
    if (imageUrl) {
        html += `
            <div style="margin-top: 8px; display: flex; align-items: center;">
                <img src="${imageUrl}" alt="${fileName}" style="max-width: 120px; max-height: 120px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); margin-right: 8px;" />
                <span style="font-size: 13px; color: #555;">${fileName}</span>
            </div>
        `;
    }
    messageDiv.innerHTML = html;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

fileInput.addEventListener('change', function() {
    filePreviewContainer.innerHTML = "";
    if (!fileInput.files[0]) return;
    uploadedFile = fileInput.files[0];
    const filePreview = document.createElement("div");
    filePreview.classList.add("file-preview");
    const img = document.createElement("img");
    img.src = URL.createObjectURL(uploadedFile);
    const fileInfo = document.createElement("span");
    fileInfo.classList.add("file-info");
    fileInfo.textContent = uploadedFile.name;
    const deleteButton = document.createElement("span");
    deleteButton.classList.add("delete-file");
    deleteButton.textContent = "✕";
    deleteButton.onclick = function() {
        fileInput.value = "";
        uploadedFile = null;
        filePreviewContainer.innerHTML = "";
    };
    filePreview.appendChild(img);
    filePreview.appendChild(fileInfo);
    filePreview.appendChild(deleteButton);
    filePreviewContainer.appendChild(filePreview);
});

document.getElementById('chatForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    if (isLoading) return;
    const userMessage = userInput.value.trim();
    if (userMessage === "" && !uploadedFile) return;

    let imageUrl = null, fileName = null;
    if (uploadedFile) {
        imageUrl = URL.createObjectURL(uploadedFile);
        fileName = uploadedFile.name;
    }
    // Tạo formData và append file trước khi reset uploadedFile
    let formData = new FormData();
    formData.append("query", userMessage);
    if (uploadedFile) {
        formData.append("file", uploadedFile);
    }
    appendMessage(userMessage, "user", imageUrl, fileName);
    userInput.value = "";
    fileInput.value = "";
    filePreviewContainer.innerHTML = "";
    uploadedFile = null;
    loadingText.style.display = "block";
    isLoading = true;
    if (sendBtn) sendBtn.disabled = true;
    if (uploadLabel) uploadLabel.style.pointerEvents = "none";

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();
        loadingText.style.display = "none";
        let botMsg = result.result;
        // Nếu là object, chỉ lấy trường trả lời chính (ưu tiên content, answer, text, output...)
        if (typeof botMsg === 'object' && botMsg !== null) {
            if (botMsg.content) {
                botMsg = botMsg.content;
            } else if (botMsg.answer) {
                botMsg = botMsg.answer;
            } else if (botMsg.text) {
                botMsg = botMsg.text;
            } else if (botMsg.output) {
                botMsg = botMsg.output;
            } else {
                // Nếu không có trường nào, lấy giá trị đầu tiên dạng string
                const firstStr = Object.values(botMsg).find(v => typeof v === 'string');
                botMsg = firstStr || "Không thể xử lý yêu cầu của bạn. Vui lòng thử lại.";
            }
        }
        // Chỉ in đậm các mục chính như 1), 2), ... ở đầu dòng
        function formatBotMsg(msg) {
            if (!msg || typeof msg !== 'string') return msg;
            return msg.split(/\r?\n/).map(line => {
                // In đậm nếu là dòng bắt đầu bằng số và dấu ) (ví dụ: 1) 2) ...)
                if (/^\s*\d+\)/.test(line)) {
                    return '<b>' + line + '</b>';
                }
                return line;
            }).join('<br>');
        }
        appendMessage(formatBotMsg(botMsg) || "Không thể xử lý yêu cầu của bạn. Vui lòng thử lại.", "bot");
    } catch (error) {
        loadingText.style.display = "none";
        appendMessage("Đã xảy ra lỗi khi xử lý yêu cầu. Vui lòng thử lại.", "bot");
    }
    isLoading = false;
    if (sendBtn) sendBtn.disabled = false;
    if (uploadLabel) uploadLabel.style.pointerEvents = "auto";
});
