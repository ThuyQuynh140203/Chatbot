# Chatbot AI Trợ Lý Tài Liệu (RAG + OCR + LLM)

## Giới thiệu

Đây là hệ thống chatbot AI hỗ trợ trả lời câu hỏi dựa trên tài liệu (PDF, DOCX, hình ảnh) và hội thoại tự do. Chatbot sử dụng pipeline RAG (Retrieval-Augmented Generation) kết hợp OCR (Vintern-1B-v3_5), tìm kiếm ngữ nghĩa (FAISS + BAAI/bge-m3), và mô hình ngôn ngữ lớn (Ollama/vistral-7b-chat). Hệ thống hỗ trợ API (FastAPI), giao diện web (Streamlit), lưu lịch sử vào MongoDB, cache bằng Redis.

## Tính năng chính
- Trả lời câu hỏi dựa trên tài liệu PDF, DOCX, PNG, JPG (hỗ trợ tiếng Việt).
- Chat hội thoại tự do với AI.
- Tìm kiếm thông tin theo ngữ nghĩa (vector search FAISS).
- Tích hợp OCR mạnh mẽ (Vintern-1B-v3_5) cho tài liệu scan/hình ảnh.
- Lưu lịch sử hội thoại vào MongoDB.
- Hỗ trợ API (FastAPI) và giao diện web (Streamlit).
- Cơ chế cache kết quả OCR và LLM bằng Redis giúp tăng tốc độ xử lý.
- Xác thực JWT cho API và giao diện web.

## Pipeline xử lý tài liệu
1. **OCR (Vintern-1B-v3_5):** Trích xuất text từ PDF scan/hình ảnh.
2. **Tách đoạn (Text Splitter):** Chia nhỏ nội dung thành các chunk.
3. **Embedding (BAAI/bge-m3):** Mã hóa các đoạn thành vector.
4. **Lưu trữ (FAISS):** Lưu vector vào FAISS để tìm kiếm ngữ nghĩa.
5. **RAG (Ollama LLM):** Trả lời dựa trên context tìm được từ vectorstore.

## Cài đặt

### 1. Yêu cầu hệ thống
- Python >= 3.10
- MongoDB (mặc định: localhost:27017)
- Redis (mặc định: localhost:6379)
- Poppler (đã kèm thư mục `poppler-24.08.0/`)
- Ollama (cài và pull model vistral-7b-chat hoặc model khác)

### 2. Cài đặt thư viện Python
```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu
- Đặt các file PDF, DOCX, hình ảnh vào thư mục `data/`.
- Chạy lệnh sau để tạo vector database cho tìm kiếm ngữ nghĩa:
```bash
python prepare_vector_db.py
```

### 4. Tạo user đăng nhập (nếu chưa có)
```bash
python database.py
```
User mặc định: **admin / admin123**

## Sử dụng

### 1. Chạy API server (FastAPI)
```bash
uvicorn chatbot_api:app --reload
```
- Truy cập docs: http://127.0.0.1:8000/docs

### 2. Chạy giao diện web (Streamlit)
```bash
streamlit run app.py
```
- Đăng nhập bằng tài khoản đã tạo để sử dụng chatbot.

### 3. Chạy chatbot CLI (nếu muốn)
```bash
python chatbot.py
```

## API endpoints
- `POST /login` : Đăng nhập, trả về JWT token.
    - Body: `username`, `password` (form-data)
- `POST /chat` : Gửi câu hỏi hội thoại tự do, trả về câu trả lời. (Yêu cầu Bearer Token)
    - Body: `message` (str)
- `POST /ask_from_file` : Gửi file (PDF, DOCX, PNG, JPG) và câu hỏi, trả về câu trả lời dựa trên nội dung file. (Yêu cầu Bearer Token)
    - Body: `file` (UploadFile), `question` (str)

## Cấu trúc thư mục chính
```
├── app.py                # Giao diện web Streamlit
├── chatbot.py            # Chatbot CLI
├── chatbot_api.py        # FastAPI endpoints
├── prepare_vector_db.py  # Tạo vector DB từ tài liệu
├── database.py           # Kết nối MongoDB, tạo user mẫu
├── auth.py               # Xác thực JWT
├── data/                 # Thư mục chứa tài liệu nguồn
├── vectorstores/         # Lưu trữ vector DB FAISS
├── poppler-24.08.0/      # Thư viện Poppler cho PDF
```

## Lưu ý
- Để sử dụng OCR, nên có GPU để tăng tốc (nếu không sẽ chạy trên CPU, sẽ chậm).
- Lịch sử hội thoại và truy vấn được lưu vào MongoDB.
- Kết quả OCR và LLM được cache bằng Redis để tăng hiệu năng.
- Đường dẫn Poppler đã cấu hình sẵn cho Windows, cần điều chỉnh nếu chạy trên OS khác.
- Ollama cần pull model trước khi sử dụng: `ollama pull vistral-7b-chat`

