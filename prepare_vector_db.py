from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from PIL import Image
from pdf2image import convert_from_path
from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# def create_db_from_file():
#     # Tải dữ liệu từ thư mục chứa các file PDF

#     loader = DirectoryLoader(pdf_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
#     documents = loader.load()

#     # Chia nhỏ văn bản thành các đoạn văn bản nhỏ hơn
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=512,
#         chunk_overlap=50,
#         length_function=len
#     )

#     chunks = text_splitter.split_documents(documents)

#     # HuggingFace BGE-M3 embeddings
#     embedding = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-m3",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     # Tạo vector store FAISS từ các đoạn văn bản
#     vector_store = FAISS.from_documents(chunks, embedding)

#     # Lưu vector store
#     vector_store.save_local(vector_db_path)

#     print("Done! Vector DB đã lưu tại:", vector_db_path)
# if __name__ == "__main__":
#     create_db_from_file()

# Cấu hình Vintern
vintern_model_id = "5CD-AI/Vintern-1B-v3_5"
vintern_model = AutoModel.from_pretrained(
    vintern_model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()
vintern_tokenizer = AutoTokenizer.from_pretrained(
    vintern_model_id, trust_remote_code=True
)


def vintern_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(torch.bfloat16)
    question = "<image>\nMô tả nội dung ảnh một cách chi tiết."
    config = dict(max_new_tokens=2048, do_sample=False)
    return vintern_model.chat(vintern_tokenizer, tensor, question, config)


def create_db_from_all_files():
    documents = []

    # Load PDF dạng text bằng DirectoryLoader + PyPDFLoader
    text_pdf_loader = DirectoryLoader(
        path=pdf_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    text_docs = text_pdf_loader.load()
    documents.extend(text_docs)

    # Duyệt toàn bộ thư mục để tìm file ảnh hoặc PDF cần OCR bằng Vintern
    loaded_files = {doc.metadata.get("source", "") for doc in text_docs}

    for fname in sorted(os.listdir(pdf_data_path)):
        fpath = os.path.join(pdf_data_path, fname)
        if fname in loaded_files:
            continue  # đã xử lý bằng PyPDFLoader rồi

        if fname.lower().endswith(".pdf"):
            pages = convert_from_path(
                fpath, poppler_path=r"poppler-24.08.0\\Library\\bin"
            )
            for idx, page_img in enumerate(pages):
                tmp_img = f"temp_page_{idx}.png"
                page_img.save(tmp_img)
                content = vintern_from_image(tmp_img)
                os.remove(tmp_img)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": fname, "page": idx + 1},
                    )
                )

        elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
            content = vintern_from_image(fpath)
            documents.append(Document(page_content=content, metadata={"source": fname}))

    if len(documents) == 0:
        print("Không có nội dung nào được trích xuất.")
        return

    print(
        f"Đã trích xuất {len(documents)} đoạn. Bắt đầu chia đoạn và tạo vectorstore..."
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(vector_db_path)

    print("Hoàn tất. Vector DB đã được lưu tại:", vector_db_path)


if __name__ == "__main__":
    create_db_from_all_files()
