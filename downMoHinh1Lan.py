from sentence_transformers import SentenceTransformer
# Tải và lưu cục bộ vào folder models
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.save('./models/embedding_model')
print("Đã tải xong mô hình nhúng về máy!")