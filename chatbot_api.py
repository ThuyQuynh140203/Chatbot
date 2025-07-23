from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
from datetime import datetime
import os
import tempfile

from chatbot import (
    get_answer,
    initialize_chat_messages,
    load_vector_store,
    vintern_ocr_cached,
    read_docx,
    create_temp_vectorstore,
)
from database import session_collection, user_collection
from pdf2image import convert_from_path
from auth import create_access_token, verify_token, verify_password, hash_password

app = FastAPI()

# === OAuth2 config ===
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# === Dependency xác thực ===
def get_current_user(token: str = Depends(oauth2_scheme)):
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc đã hết hạn",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id

# === Khởi tạo ===
chat_history = initialize_chat_messages()
vector_store = load_vector_store()
POPPLER_PATH = r"poppler-24.08.0\Library\bin"  # Điều chỉnh đúng đường dẫn poppler

@app.get("/")
def home():
    return {"message": "Chatbot API is running. Endpoints: /chat, /ask_from_file"}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Demo: kiểm tra user mẫu hoặc từ DB
    user = user_collection.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Sai username hoặc password")
    access_token = create_access_token({"sub": str(user["_id"])})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/chat")
async def chat(message: str = Form(...), user_id: str = Depends(get_current_user)):
    if not message:
        raise HTTPException(status_code=400, detail="Thiếu message")

    response = get_answer(message, chat_history, vector_store)
    session_collection.insert_one({
        "user": user_id,
        "type": "text",
        "message": message,
        "response": response,
        "timestamp": datetime.utcnow()
    })
    return {"answer": response}


# @app.post("/ask_from_file")
# async def ask_from_file(file: UploadFile = File(...), question: str = Form(...), user_id: str = Depends(get_current_user)):
#     filename = file.filename.lower()
#     suffix = os.path.splitext(filename)[1]

#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
#         temp_path = temp_file.name
#         temp_file.write(await file.read())

#     try:
#         if filename.endswith(".pdf"):
#             pages = convert_from_path(temp_path, poppler_path=POPPLER_PATH)
#             texts = []
#             for idx, page in enumerate(pages):
#                 tmp_image = f"temp_page_{idx}.png"
#                 page.save(tmp_image)
#                 texts.append(vintern_ocr_cached(tmp_image))
#                 os.remove(tmp_image)
#             content = "\n\n".join(texts)

#         elif filename.endswith((".png", ".jpg", ".jpeg")):
#             content = vintern_ocr_cached(temp_path)

#         elif filename.endswith(".docx"):
#             content = read_docx(temp_path)

#         else:
#             raise HTTPException(status_code=400, detail="Định dạng file không hỗ trợ")

#     finally:
#         os.remove(temp_path)

#     # ✅ In nội dung sau OCR để kiểm tra
#     print("=== 📄 OCR OUTPUT ===")
#     print(content[:2000], "...\n")  # in 2000 ký tự đầu

#     temp_vectorstore = create_temp_vectorstore(content)
#     docs = temp_vectorstore.similarity_search(question, k=5)
#     context = "\n\n".join([doc.page_content for doc in docs])

#     full_prompt = f"""
#     Bạn là một trợ lý AI thông minh, hãy trả lời câu hỏi dựa trên nội dung sau:
#     {context}
#     Câu hỏi: {question}
#     """

#     response = get_answer(full_prompt, chat_history, temp_vectorstore)


#     session_collection.insert_one({
#         "user": user_id,
#         "type": "file",
#         "question": question,
#         "file_name": filename,
#         "file_type": filename.split(".")[-1],
#         "file_content": content,
#         "response": response,
#         "timestamp": datetime.utcnow()
#     })

#     return {"answer": response}

@app.post("/ask_from_file")
async def ask_from_file(file: UploadFile = File(...), question: str = Form(...), user_id: str = Depends(get_current_user)):
    filename = file.filename.lower()
    suffix = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await file.read())

    try:
        # === Trích xuất nội dung từ file ===
        if filename.endswith(".pdf"):
            pages = convert_from_path(temp_path, poppler_path=POPPLER_PATH)
            texts = []
            for idx, page in enumerate(pages):
                tmp_image = f"temp_page_{idx}.png"
                page.save(tmp_image)
                texts.append(vintern_ocr_cached(tmp_image))
                os.remove(tmp_image)
            content = "\n\n".join(texts)

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            content = vintern_ocr_cached(temp_path)

        elif filename.endswith(".docx"):
            content = read_docx(temp_path)

        else:
            raise HTTPException(status_code=400, detail="Định dạng file không hỗ trợ")

    finally:
        os.remove(temp_path)

    print("=== 📄 OCR OUTPUT ===")
    print(content[:2000], "...\n")

    # === Tạo vectorstore tạm thời từ nội dung ===
    temp_vectorstore = create_temp_vectorstore(content)

    # Gọi mô hình RAG
    response = get_answer(question, chat_history, temp_vectorstore)

    # === Ghi log vào MongoDB ===
    session_collection.insert_one({
        "user": user_id,
        "type": "file",
        "question": question,
        "file_name": filename,
        "file_type": filename.split(".")[-1],
        "file_content": content,
        "response": response,
        "timestamp": datetime.utcnow()
    })

    return {"answer": response}