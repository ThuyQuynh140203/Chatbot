
import streamlit as st
import requests
import time

st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Trợ lý AI - Chatbot")

# Khởi tạo các session state cần thiết
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = ""

# ==== Đăng nhập ==== #
if st.session_state.token is None:
    st.subheader(":lock: Đăng nhập để sử dụng Chatbot")
    with st.form("login_form"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập")
    if submitted:
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/login",
                data={"username": username, "password": password},
                headers={"accept": "application/json"},
            )
            if resp.status_code == 200:
                token = resp.json()["access_token"]
                st.session_state.token = token
                st.session_state.username = username
                st.success("Đăng nhập thành công!")
                st.rerun()
            else:
                st.error("Sai tài khoản hoặc mật khẩu!")
        except Exception as e:
            st.error(f"Lỗi kết nối: {e}")
    st.stop()

st.markdown(f"**Đã đăng nhập:** `{st.session_state.username}`")
if st.button("Đăng xuất"):
    st.session_state.token = None
    st.session_state.username = ""
    st.rerun()

# ==== Hỏi từ tệp ==== #
st.markdown("## Hỏi từ nội dung tệp")

uploaded_file = st.file_uploader("Tải tệp (PDF, DOCX, PNG, JPG)", type=["pdf", "docx", "png", "jpg"])
question = st.text_input("Câu hỏi của bạn về tệp:")

if uploaded_file and question:
    st.markdown(f"**Tệp:** {uploaded_file.name}")
    st.markdown(f"**Câu hỏi:** {question}")

    file = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {"question": question}
    headers = {"Authorization": f"Bearer {st.session_state.token}"}

    with st.spinner("Đang xử lý nội dung tệp..."):
        start_time = time.time()
        response = requests.post(
            "http://127.0.0.1:8000/ask_from_file",
            files=file,
            data=data,
            headers=headers,
            timeout=2400,
        )
        elapsed_time = time.time() - start_time
        answer = response.json().get("answer", "Không tìm thấy câu trả lời.")

    st.markdown(f"**Trả lời:** {answer}")
    st.markdown(f"**Thời gian phản hồi:** {elapsed_time:.2f} giây")
    st.session_state.messages.append({
        "user": f"[Tệp: {uploaded_file.name}] {question}",
        "bot": answer,
    })

st.divider()

# ==== Chat thông thường ==== #
st.markdown("## 💬 Hỏi đáp tự do")

prompt = st.chat_input("Nhập câu hỏi tại đây...")

if prompt:
    st.markdown(f"**Bạn:** {prompt}")
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    with st.spinner("Đang trả lời..."):
        start_time = time.time()
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            data={"message": prompt},
            headers=headers,
            timeout=2400,
        )
        answer = response.json().get("answer", "Không tìm thấy câu trả lời.")
        elapsed_time = time.time() - start_time
    st.markdown(f"**AI:** {answer}")
    st.markdown(f"**Thời gian phản hồi:** {elapsed_time:.2f} giây")
    st.session_state.messages.append({"user": prompt, "bot": answer})

# ==== Lịch sử ==== #
if st.session_state.messages:
    st.markdown("## Lịch sử trò chuyện")
    for msg in st.session_state.messages:
        st.markdown(f"**Bạn:** {msg['user']}")
        st.markdown(f"**AI:** {msg['bot']}")
