# get_ipython().run_line_magic('pip', 'install timm einops')
# get_ipython().run_line_magic('pip', 'install --no-dependencies --upgrade flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl')

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from docx import Document
from PIL import Image
from pdf2image import convert_from_path
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_temp_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.from_documents(docs, embedding)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def initialize_model():
    model_name = "5CD-AI/Vintern-1B-v3_5"

    try:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval()
    except:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )

    return model, tokenizer


def initialize_embeddings():
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding


def load_vector_store():
    embedding = initialize_embeddings()
    vector_store = FAISS.load_local(
        "vectorstores/db_faiss",
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def vintern_from_image(model, tokenizer, file_path: str, question: str = None, show_image: bool = True) -> str:
    if show_image:
        plt.figure(figsize=(8, 8))
        plt.imshow(Image.open(file_path))
        plt.axis("off")
        plt.show()
    pixel_values = load_image(file_path, max_num=6).to(torch.bfloat16)
    if question is None:
        question = "<image>\nMô tả hình ảnh một cách chi tiết trả về dạng markdown."
    generation_config = dict(
        max_new_tokens=2048, do_sample=False, num_beams=3, repetition_penalty=2.5
    )
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f"User: {question}\nAssistant: {response}")
    return response


def extract_from_pdf_as_images(
    model, tokenizer, pdf_path, question=None, show_image=True
):
    images = convert_from_path(pdf_path, poppler_path=r"poppler-24.08.0\Library\bin")
    all_contents = []
    for idx, img in enumerate(images):
        temp_path = f"temp_page_{idx}.png"
        img.save(temp_path)
        print(f"Trang {idx+1}:")
        content = vintern_from_image(
            model, tokenizer, temp_path, question=question, show_image=show_image
        )
        all_contents.append(f"--- Trang {idx+1} ---\n{content}")
        os.remove(temp_path)
    return "\n\n".join(all_contents)


def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


# def read_pdf(file_path: str) -> str:
#     return extract_text(file_path)


def get_answer(prompt: str, messages: list, vector_store) -> str:
    messages.append(HumanMessage(content=prompt))

    # Truy vấn vector store
    docs = vector_store.similarity_search(prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Bổ sung context vào câu hỏi
    rag_prompt = f"""
    Bối cảnh tham khảo:
    {context}
    
    Câu hỏi:
    {prompt}
    """

    llm = ChatOllama(
        model="vistral-7b-chat", temperature=0.7, base_url="http://localhost:11434"
    )

    full_response = ""
    for chunk in llm.stream(messages[:-1] + [HumanMessage(content=rag_prompt)]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
    print()

    messages.append(AIMessage(content=full_response))
    return full_response


def initialize_chat_messages():
    return [
        SystemMessage(
            content="Bạn là một trợ lý AI thông minh và hữu ích. Bạn có thể trả lời các câu hỏi, giải thích khái niệm, và hỗ trợ người dùng với nhiều loại yêu cầu khác nhau."
        )
    ]


def print_welcome_message():
    print("=== CHATBOT KHỞI ĐỘNG ===")
    print("Gõ 'exit' để thoát, 'clear' để xóa lịch sử chat")
    print("Gõ 'file:đường_dẫn_tệp' để đính kèm tệp (hỗ trợ .pdf, .docx, .png, .jpg)")
    print("=========================")


def handle_file_input(file_path: str, model, tokenizer, messages: list):
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy tệp {file_path}")
        return

    file_content = ""
    file_type = file_path.lower()

    if file_type.endswith((".pdf", ".png", ".jpg", ".jpeg")):
        if file_type.endswith(".pdf"):
            file_content = extract_from_pdf_as_images(model, tokenizer, file_path)
            print(f"Đã trích xuất thông tin từ file PDF: {file_path}")
            print("Nội dung đã trích xuất từ PDF:")
        else:
            file_content = vintern_from_image(
                model, tokenizer, file_path, show_image=True
            )
            print(f"Đã xử lý hình ảnh: {file_path}")
            print("Nội dung mô tả ảnh:")

        print(file_content)
        handle_file_questions(file_content, messages)

    elif file_type.endswith(".docx"):
        file_content = read_docx(file_path)
        print(f"Đã đọc tệp DOCX: {file_path}")
        print(file_content)

    else:
        print("Lỗi: Định dạng tệp không được hỗ trợ. Hỗ trợ: .pdf, .docx, .png, .jpg")


def handle_file_questions(file_content: str, messages: list):
    while True:
        follow_up = (
            input("Bạn có muốn hỏi thêm về nội dung file này không? (y/n): ")
            .strip()
            .lower()
        )
        if follow_up != "y":
            break

        user_question = input("Nhập câu hỏi của bạn về tệp này: ")

        temp_vectorstore = create_temp_vectorstore(file_content)
        docs = temp_vectorstore.similarity_search(user_question, k=3)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])

        user_input = f"""
            Bối cảnh từ nội dung tệp:
            {retrieved_context}

            Câu hỏi: {user_question}
            """
        print("AI: ", end="")
        response = get_answer(user_input, messages, load_vector_store())
        print()


def run_chat():
    # Khởi tạo các thành phần
    model, tokenizer = initialize_model()
    vector_store = load_vector_store()
    messages = initialize_chat_messages()

    print_welcome_message()

    while True:
        user_input = input("Bạn: ")

        if user_input.lower() == "exit":
            print("=== KẾT THÚC CHAT ===")
            break

        if user_input.lower() == "clear":
            messages = [messages[0]]
            print("Đã xóa lịch sử chat!")
            continue

        if user_input.startswith("file:"):
            file_path = user_input[5:].strip()
            handle_file_input(file_path, model, tokenizer, messages)
            continue

        # Nếu không phải tệp, thì là một câu hỏi thông thường
        print("AI: ", end="")
        get_answer(user_input, messages, vector_store)
        continue


def main():
    run_chat()


if __name__ == "__main__":
    main()
