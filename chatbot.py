# get_ipython().run_line_magic('pip', 'install timm einops')
# get_ipython().run_line_magic('pip', 'install --no-dependencies --upgrade flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl')

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from docx import Document
from pdfminer.high_level import extract_text
from PIL import Image
from pdf2image import convert_from_path
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# In[49]:


# vintern_model_id = "5CD-AI/Vintern-1B-v3_5"
# vintern_processor = AutoProcessor.from_pretrained(vintern_model_id, trust_remote_code=True)

# from transformers import AutoModelForCausalLM
# vintern_model = AutoModelForCausalLM.from_pretrained(vintern_model_id, trust_remote_code=True)

# print(type(vintern_processor))


# In[50]:


import os
import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height 

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
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
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# In[51]:


model_name = "5CD-AI/Vintern-1B-v3_5"


# In[52]:


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
      trust_remote_code=True
  ).eval()


# In[53]:


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

# Khởi tạo embedding y hệt lúc train
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load FAISS vector store
vector_store = FAISS.load_local(
    "vectorstores/db_faiss",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

# In[54]:


import matplotlib.pyplot as plt

def vintern_from_image(file_path: str, question: str = None, show_image: bool = True) -> str:
    if show_image:
        plt.figure(figsize=(8, 8))
        plt.imshow(Image.open(file_path))
        plt.axis('off')
        plt.show()
    pixel_values = load_image(file_path, max_num=6).to(torch.bfloat16)
    if question is None:
        question = "<image>\nMô tả hình ảnh một cách chi tiết trả về dạng markdown."
    generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')
    return response


# In[55]:


def extract_from_pdf_as_images(pdf_path, question=None, show_image=True):
    images = convert_from_path(pdf_path, poppler_path=r'poppler-24.08.0\Library\bin')
    all_contents = []
    for idx, img in enumerate(images):
        temp_path = f"temp_page_{idx}.png"
        img.save(temp_path)
        print(f"Trang {idx+1}:")
        content = vintern_from_image(temp_path, question=question, show_image=show_image)
        all_contents.append(f"--- Trang {idx+1} ---\n{content}")
        os.remove(temp_path)
    return "\n\n".join(all_contents)


# In[56]:


# === HÀM ĐỌC DOCX ===
def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# === HÀM ĐỌC PDF ===
def read_pdf(file_path: str) -> str:
    return extract_text(file_path)


# In[57]:


# === HÀM CHAT VISTRAL ===
# def get_answer(prompt: str, messages: list) -> str:
#     messages.append(HumanMessage(content=prompt))
    
#     llm = ChatOllama(
#         model="vistral-7b-chat",
#         temperature=0.7,
#         base_url="http://localhost:11434"
#     )
    
#     full_response = ""
#     for chunk in llm.stream(messages):
#         if chunk.content:
#             print(chunk.content, end="", flush=True)
#             full_response += chunk.content
#     print()
    
#     messages.append(AIMessage(content=full_response))
#     return full_response

def get_answer(prompt: str, messages: list) -> str:
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
        model="vistral-7b-chat",
        temperature=0.7,
        base_url="http://localhost:11434"
    )
    
    full_response = ""
    for chunk in llm.stream(messages[:-1] + [HumanMessage(content=rag_prompt)]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
    print()
    
    messages.append(AIMessage(content=full_response))
    return full_response


# In[58]:


def run_chat():
    messages = [
        SystemMessage(content="Bạn là một trợ lý AI thông minh và hữu ích. Bạn có thể trả lời các câu hỏi, giải thích khái niệm, và hỗ trợ người dùng với nhiều loại yêu cầu khác nhau.")
    ]
    
    print("=== CHATBOT KHỞI ĐỘNG ===")
    print("Gõ 'exit' để thoát, 'clear' để xóa lịch sử chat")
    print("Gõ 'file:đường_dẫn_tệp' để đính kèm tệp (hỗ trợ .pdf, .docx, .png, .jpg)")
    print("=========================")
    
    while True:
        user_input = input("Bạn: ")
        
        if user_input.lower() == 'exit':
            print("=== KẾT THÚC CHAT ===")
            break
            
        if user_input.lower() == 'clear':
            messages = [messages[0]]  # Giữ lại system message
            print("Đã xóa lịch sử chat!")
            continue
        
        # Kiểm tra nếu người dùng đính kèm tệp
        if user_input.startswith('file:'):
            file_path = user_input[5:].strip()
            
            if not os.path.exists(file_path):
                print(f"Lỗi: Không tìm thấy tệp {file_path}")
                continue
                
            file_content = ""
            file_type = file_path.lower()
            
            try:
                if file_type.endswith('.pdf'):
                    file_content = extract_from_pdf_as_images(file_path)
                    print(f"Đã trích xuất thông tin từ file PDF: {file_path}")
                    print("Nội dung đã trích xuất từ PDF:")
                    print(file_content)
                    # Hỏi tiếp về ảnh hay không?
                    follow_up = input("Bạn có muốn hỏi thêm về nội dung file này không? (y/n): ").strip().lower()
                    if follow_up != 'y':
                        continue
                    user_question = input("Nhập câu hỏi của bạn về tệp này: ")
                    user_input = f"Tệp đính kèm: {file_content}\nCâu hỏi của tôi: {user_question}"
                elif file_type.endswith('.docx'):
                    file_content = read_docx(file_path)
                    print(f"Đã đọc tệp DOCX: {file_path}")
                elif file_type.endswith(('.png', '.jpg', '.jpeg')):
                    # Hiển thị ảnh và mô tả nội dung
                    file_content = vintern_from_image(file_path, show_image=True)
                    print(f"Đã xử lý hình ảnh: {file_path}")
                    print("Nội dung mô tả ảnh:")
                    print(file_content)
                    # Hỏi tiếp về ảnh hay không?
                    follow_up = input("Bạn có muốn hỏi thêm về nội dung ảnh này không? (y/n): ").strip().lower()
                    if follow_up != 'y':
                        continue
                    user_question = input("Nhập câu hỏi của bạn về tệp này: ")
                    user_input = f"Tệp đính kèm: {file_content}\nCâu hỏi của tôi: {user_question}"
                else:
                    print(f"Lỗi: Định dạng tệp không được hỗ trợ. Hỗ trợ: .pdf, .docx, .png, .jpg")
                    continue
                    
            except Exception as e:
                print(f"Lỗi khi xử lý tệp: {str(e)} ")
                continue
        
        print("AI: ", end="")
        response = get_answer(user_input, messages)
        print()  # Thêm dòng trống sau phản hồi
    
    return messages


# # In[59]:


run_chat()

