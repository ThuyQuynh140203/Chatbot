from flask import Flask, request, jsonify
from chatbot import (
    get_answer,
    initialize_model,
    initialize_chat_messages,
    load_vector_store,
    vintern_from_image,
    extract_from_pdf_as_images,
    read_docx,
    create_temp_vectorstore,
)
import tempfile
import os

app = Flask(__name__)

# Khởi tạo mô hình và vector store 1 lần khi start server
model, tokenizer = initialize_model()
vector_store = load_vector_store()
chat_history = initialize_chat_messages()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Chatbot API is running. Available endpoints: /chat, /upload, /ask-from-content"
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Thiếu nội dung message người dùng"}), 400
    response = response = get_answer(user_message, chat_history, vector_store)

    return jsonify({"answer": response})


@app.route("/ask_from_file", methods=["POST"])
def ask_from_file():
    if "file" not in request.files or "question" not in request.form:
        return jsonify({"error": "Thiếu file hoặc câu hỏi"}), 400

    file = request.files["file"]
    question = request.form["question"]
    filename = file.filename.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        file.save(temp_file.name)
        file_path = temp_file.name

    try:
        if filename.endswith(".pdf"):
            content = extract_from_pdf_as_images(model, tokenizer, file_path, show_image=False)
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            content = vintern_from_image(model, tokenizer, file_path, show_image=False)
        elif filename.endswith(".docx"):
            content = read_docx(file_path)
        else:
            return jsonify({"error": "Định dạng file không hỗ trợ"}), 400
    finally:
        os.remove(file_path)

    temp_vectorstore = create_temp_vectorstore(content)
    docs = temp_vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    full_prompt = f"""
    Bạn là một trợ lý AI thông minh, hãy trả lời câu hỏi sau dựa trên nội dung đã cho:
    {context}
    Câu hỏi: {question}
    """

    response = get_answer(full_prompt, chat_history, vector_store)
    return jsonify({"answer": response})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
