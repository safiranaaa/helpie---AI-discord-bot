from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import GOOGLE_API_KEY, VECTOR_DB_URL, VECTOR_DB_API_KEY

def start_chat():
    print("Menginisialisasi Agen Modern (Jalur Native Google)...")
    
    # 1. Hubungkan ke memori vektor cloud
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="personal_context",
        url=VECTOR_DB_URL,
        api_key=VECTOR_DB_API_KEY,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
   # 2. Siapkan Otak Gemini (Sesuai instruksi paksa dari server Google)
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest", 
        google_api_key=GOOGLE_API_KEY,
        max_retries=0
    )

    # 3. Buat Instruksi Khusus (Prompt Template)
    template = """Gunakan informasi berikut untuk menjawab pertanyaan.
    Jika jawabannya tidak ada di dalam informasi ini, jawab saja sesuai pengetahuanmu.
    
    Konteks Personal:
    {context}

    Pertanyaan: {question}
    Jawaban:"""
    prompt = PromptTemplate.from_template(template)

    # 4. Format hasil pencarian dokumen
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Rangkai menjadi Rantai LCEL
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Agen siap! Ketik 'keluar' untuk berhenti.\n")
    
    # 6. Looping Terminal dengan Error Handling
    while True:
        try:
            user_input = input("Kamu: ")
            if user_input.lower() == 'keluar':
                break
                
            response = qa_chain.invoke(user_input)
            print(f"Agen: {response}\n")
            
        except Exception as e:
            # Menangkap error jika sewaktu-waktu kena limit lagi di terminal
            print(f"[ERROR] Waduh kena limit atau error API cuy: {e}\nTunggu sebentar lalu ketik lagi.\n")

if __name__ == "__main__":
    start_chat()