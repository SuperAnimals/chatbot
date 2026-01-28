import os
import pickle
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.storage import LocalFileStore
from langchain.storage import EncoderBackedStore

# Ép hệ thống chạy Offline tuyệt đối
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class OfflineChatEngine:
    def __init__(self):
        # Ket noi Qwen 2.5 14B qua Ollama (Tan dung 12GB VRAM RTX 4070)
        self.llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model="qwen2.5:14b-instruct",
            temperature=0.1
        )
        
        # Embedding Model tu folder models/
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./models/embedding_model",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Thiet lap chia van ban de dat do chinh xac 9/10
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        # Vector Database cuc bo
        self.vectorstore = Chroma(
            collection_name="company_rules",
            embedding_function=self.embeddings,
            persist_directory="./vector_db"
        )
        
        # Luu tru van ban goc dung Encoder
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
        """Doc va nap file vao kho tri thuc"""
        loader = PyMuPDFLoader(file_path) if file_path.endswith('.pdf') else Docx2txtLoader(file_path)
        docs = loader.load()
        self.retriever.add_documents(docs)

    def delete_all(self):
        """Xoa trang kho du lieu"""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name="company_rules",
            embedding_function=self.embeddings,
            persist_directory="./vector_db"
        )
