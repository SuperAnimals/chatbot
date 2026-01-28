import streamlit as st
import os
from engine import OfflineChatEngine

st.set_page_config(page_title="Hệ thống Nội quy Offline", layout="wide")

# Khởi tạo hoặc lấy lại Engine từ bộ nhớ phiên
if "engine" not in st.session_state:
    st.session_state.engine = OfflineChatEngine()
    st.session_state.chat_history = []

st.title("🤖 Chatbot Nội quy & An toàn Công ty (100% Offline)")

# Sidebar: Quản lý File (Giới hạn 5 file)
with st.sidebar:
    st.header("📁 Quản lý tài liệu")
    files = [f for f in os.listdir("data") if f.endswith(('.pdf', '.docx'))]
    st.write(f"Đang có: {len(files)}/5 file")
    
    uploaded_file = st.file_uploader("Thêm tài liệu mới", type=['pdf', 'docx'])
    if uploaded_file and len(files) < 5:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Đang phân tích tài liệu..."):
            st.session_state.engine.add_file(os.path.join("data", uploaded_file.name))
        st.rerun()

    if st.button("🗑️ Xóa toàn bộ dữ liệu"):
        for f in files: os.remove(os.path.join("data", f))
        st.session_state.engine.delete_all()
        st.rerun()

# Khu vực Chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Hỏi về nội quy an toàn..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # Tìm dữ liệu liên quan
        context_docs = st.session_state.engine.retriever.get_relevant_documents(prompt)
        context_text = "\n\n".join([d.page_content for d in context_docs])
        
        # Prompt chuyên sâu dành cho Qwen 2.5 14B
        full_prompt = f"""Bạn là chuyên gia về nội quy công ty. Trả lời dựa TRỰC TIẾP vào tài liệu dưới đây.
        Nếu không có thông tin, hãy nói 'Tôi không tìm thấy quy định này'. 
        KHÔNG tự ý bịa đặt.
        
        Tài liệu gốc:
        {context_text}
        
        Câu hỏi của nhân viên: {prompt}"""
        
        # Hiển thị kết quả ngay lập tức (Streaming)
        response = st.write_stream(st.session_state.engine.llm.stream(full_prompt))
        st.session_state.chat_history.append({"role": "assistant", "content": response})