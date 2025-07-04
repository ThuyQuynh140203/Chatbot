from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_db_from_file():
    # Tải dữ liệu từ thư mục chứa các file PDF
    loader = DirectoryLoader(pdf_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Chia nhỏ văn bản thành các đoạn văn bản nhỏ hơn
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    # HuggingFace BGE-M3 embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Tạo vector store FAISS từ các đoạn văn bản
    vector_store = FAISS.from_documents(chunks, embedding)

    # Lưu vector store 
    vector_store.save_local(vector_db_path)         
    
    print("✅ Done! Vector DB đã lưu tại:", vector_db_path)
if __name__ == "__main__":
    create_db_from_file()