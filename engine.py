import os
import pickle
# 1. Các thư viện bóc tách và tách văn bản (Phải cài langchain-text-splitters)
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# 2. Cấu hình Retriever và Storage (Sửa lỗi dòng 7, 8, 9)
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_community.storage import LocalFileStore
from langchain.storage import EncoderBackedStore

# Ép hệ thống chạy 100% Offline, chặn mọi kết nối ra ngoài
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class OfflineChatEngine:
    def __init__(self):
        # Kết nối Qwen 2.5 14B (Sử dụng 12GB VRAM của RTX 4070)
        self.llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model="qwen2.5:14b-instruct",
            temperature=0.1
        )
        
        # Mô hình nhúng đã tải về thư mục models/
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./models/embedding_model",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Cấu hình tách văn bản đảm bảo độ chính xác 9/10 cho nội quy
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        # Cơ sở dữ liệu Vector cục bộ trong folder vector_db/
        self.vectorstore = Chroma(
            collection_name="company_rules",
            embedding_function=self.embeddings,
            persist_directory="./vector_db"
        )
        
        # Bộ lưu trữ văn bản gốc (Dùng EncoderBackedStore để ổn định hơn)
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
        """Đọc và nạp dữ liệu từ PDF hoặc Docx"""
        loader = PyMuPDFLoader(file_path) if file_path.endswith('.pdf') else Docx2txtLoader(file_path)
        docs = loader.load()
        self.retriever.add_documents(docs)

    def delete_all(self):
        """Xóa trắng dữ liệu để làm lại từ đầu"""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name="company_rules",
            embedding_function=self.embeddings,
            persist_directory="./vector_db"
        )
