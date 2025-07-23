
import os
import hashlib
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import redis

# === CẤU HÌNH ===
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# ==== REDIS CACHE THUẦN CHO OCR ==== #
redis_client = redis.Redis.from_url("redis://localhost:6379")

def get_ocr_cache(image_path: str) -> str | None:
    with open(image_path, "rb") as f:
        file_bytes = f.read()
        content_hash = hashlib.md5(file_bytes).hexdigest()
    key = "ocr:" + content_hash + ":" + os.path.basename(image_path)
    result = redis_client.get(key)
    if result:
        print("✅ [OCR CACHE HIT]", image_path)
        return result.decode("utf-8")
    return None

def set_ocr_cache(image_path: str, value: str):
    with open(image_path, "rb") as f:
        file_bytes = f.read()
        content_hash = hashlib.md5(file_bytes).hexdigest()
    key = "ocr:" + content_hash + ":" + os.path.basename(image_path)
    redis_client.setex(key, 21600, value)  # TTL 6h
    print("💾 [OCR CACHE SAVE]", image_path)


# ==== Cấu hình thiết bị và kiểu dữ liệu ==== #
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ==== Biến toàn cục ==== #
_model_vintern = None
_tokenizer_vintern = None

# ==== Tiền xử lý ảnh ==== #
_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ==== Khởi tạo mô hình Vintern ==== #
def init_vintern():
    global _model_vintern, _tokenizer_vintern
    if _model_vintern and _tokenizer_vintern:
        return

    print("🔄 Tải mô hình Vintern-1B-v3_5...")
    _model_vintern = AutoModel.from_pretrained(
        "5CD-AI/Vintern-1B-v3_5",
        torch_dtype=_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).to(_device).eval()

    _tokenizer_vintern = AutoTokenizer.from_pretrained(
        "5CD-AI/Vintern-1B-v3_5", trust_remote_code=True, use_fast=False
    )
    print("Sẵn sàng dùng Vintern.")

# ==== Hàm OCR có cache ==== #
def vintern_ocr_cached(image_path: str) -> str:
    init_vintern()

    cached = get_ocr_cache(image_path)
    if cached:
        return cached

    print(f"🔍 OCR: {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = _transform(image).unsqueeze(0).to(device=_device, dtype=_dtype)

    prompt = "<image>\nMô tả hình ảnh một cách chi tiết trả về dạng markdown."
    raw_response = _model_vintern.chat(
        _tokenizer_vintern,
        pixel_values,
        prompt,
        generation_config={
            "max_new_tokens": 2048,
            "do_sample": False,
            "num_beams": 3,
            "repetition_penalty": 3.5
        }
    )

    if isinstance(raw_response, list) and isinstance(raw_response[0], dict):
        response = raw_response[0]["generated_text"]
    else:
        response = str(raw_response)

    # Loại bỏ dòng trùng
    seen, lines = set(), []
    for line in response.strip().splitlines():
        if line not in seen:
            lines.append(line)
            seen.add(line)
    response = "\n".join(lines)

    set_ocr_cache(image_path, response)
    return response


# === TẠO VECTORSTORE TỪ FILE PDF/ẢNH ===
def create_db_from_all_files():
    documents = []

    # Bước 1: Load các PDF có thể đọc text trực tiếp
    print("Đang đọc các file PDF có thể trích xuất văn bản...")
    text_pdf_loader = DirectoryLoader(
        path=pdf_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    text_docs = text_pdf_loader.load()
    documents.extend(text_docs)

    # Lưu tên các file đã xử lý bằng PyPDFLoader
    loaded_files = {os.path.basename(doc.metadata.get("source", "")) for doc in text_docs}

    # Bước 2: Xử lý các file còn lại bằng OCR
    for fname in sorted(os.listdir(pdf_data_path)):
        fpath = os.path.join(pdf_data_path, fname)
        if fname in loaded_files:
            continue

        print(f"Đang xử lý (OCR): {fname}")

        # Nếu là PDF → OCR từng trang
        if fname.lower().endswith(".pdf"):
            pages = convert_from_path(fpath, poppler_path=r"poppler-24.08.0\Library\bin")
            for idx, page_img in enumerate(pages):
                tmp_img = f"temp_{os.path.splitext(fname)[0]}_page_{idx}.png"
                page_img.save(tmp_img)
                content = vintern_ocr_cached(tmp_img)
                os.remove(tmp_img)
                documents.append(Document(
                    page_content=content,
                    metadata={"source": fname, "page": idx + 1}
                ))

        # Nếu là ảnh đơn
        elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
            content = vintern_ocr_cached(fpath)
            documents.append(Document(page_content=content, metadata={"source": fname}))

    if not documents:
        print("Không có nội dung nào để xử lý.")
        return

    # Bước 3: Tách văn bản thành các đoạn nhỏ để embedding
    print("Đang chia văn bản thành các đoạn...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"Số đoạn sau khi chia: {len(chunks)}")

    # Bước 4: Tạo vectorstore
    print("Đang tính embedding và lưu vector store...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(vector_db_path)

    print(f"Vector DB đã lưu tại: {vector_db_path}")


if __name__ == "__main__":
    create_db_from_all_files()
