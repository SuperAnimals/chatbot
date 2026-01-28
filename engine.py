import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.storage import LocalFileStore
from langchain.storage import EncoderBackedStore
import pickle

# Ép toàn bộ hệ thống Python chạy ở chế độ Offline tuyệt đối
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class OfflineChatEngine:
    def __init__(self):
        # 1. Kết nối với Qwen 2.5 14B qua Ollama chạy cục bộ
        self.llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model="qwen2.5:14b-instruct",
            temperature=0.1
        )
        
        # 2. Cấu hình mô hình nhúng Offline từ folder models
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./models/embedding_model",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 3. Cấu hình tách văn bản: Giữ nguyên cấu trúc Điều/Khoản
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        # 4. Lưu trữ Vector cục bộ
        self.vectorstore = Chroma(
            collection_name="company_rules",
            embedding_function=self.embeddings,
            persist_directory="./vector_db"
        )
        
        # 5. Lưu trữ văn bản gốc cục bộ (Thay thế cho create_kv_docstore cũ)
        fs = LocalFileStore("./vector_db/docstore")
        self.store = EncoderBackedStore(
            store=fs,
            encoder=lambda d: pickle.dumps(d),
            decoder=lambda b: pickle.loads(b),
            key_encoder=lambda k: k
        )
        
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

    def add_file(self, file_path):
        """Đọc và nạp file vào trí nhớ"""
        loader = PyMuPDFLoader(file_path) if file_path.endswith('.pdf') else Docx2txtLoader(file_path)
        docs = loader.load()
        self.retriever.add_documents(docs)

    def delete_all(self):
        """Xóa sạch kho dữ liệu để cập nhật mới"""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name="company_rules",
            embedding_function=self.embeddings,
            persist_directory="./vector_db"
        )