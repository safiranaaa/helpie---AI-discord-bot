import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.config import VECTOR_DB_URL, VECTOR_DB_API_KEY
import pytesseract

# Beritahu Python di mana Tesseract terinstal di Windows-mu
# Sesuaikan jalurnya jika kamu menginstalnya di folder yang berbeda!
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\MSI\Tesseract-OCR\tesseract.exe'

def load_documents(data_dir):
    documents = []
    print(f"[*] Memindai folder: {data_dir}")
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            try:
                # 1. Menangani Teks Murni
                if ext in ['.md', '.txt']:
                    print(f"  -> Membaca teks: {file}")
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                
                # 2. Menangani Dokumen PDF
                elif ext == '.pdf':
                    print(f"  -> Mengekstrak PDF: {file}")
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                
                # 3. Menangani Gambar (OCR)
                elif ext in ['.png', '.jpg', '.jpeg']:
                    print(f"  -> Menjalankan OCR pada gambar: {file}")
                    loader = UnstructuredImageLoader(file_path)
                    documents.extend(loader.load())
                    
            except Exception as e:
                print(f"  [!] Gagal memproses {file}: {e}")
                
    return documents

def build_knowledge_base():
    print("=== PABRIK INGATAN MULTIMODAL MEMULAI PROSES ===")
    
    # Memanggil fungsi loader cerdas kita
    documents = load_documents('./data/knowledge_base')
    
    if not documents:
        print("Folder kosong atau format tidak didukung. Masukkan file .md, .pdf, atau .png/.jpg")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"\nBerhasil memecah dokumen menjadi {len(chunks)} potongan informasi.")

    print("Mengunggah data ke Vector DB (Qdrant)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=VECTOR_DB_URL,
        api_key=VECTOR_DB_API_KEY,
        collection_name="personal_context",
        force_recreate=True
    )
    print("Data berhasil diunggah! Otak AI sekarang sudah terisi info dari teks, PDF, dan Gambar.")

if __name__ == "__main__":
    build_knowledge_base()