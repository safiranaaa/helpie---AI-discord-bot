import os
import discord
import asyncio
from discord.ext import commands
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.config import GOOGLE_API_KEY, VECTOR_DB_URL, VECTOR_DB_API_KEY, DISCORD_TOKEN

# Buat folder 'temp' secara otomatis untuk nyimpen PDF sementara
if not os.path.exists('./temp'):
    os.makedirs('./temp')

print("[*] Menghubungkan ke Gemini...")
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest", 
    google_api_key=GOOGLE_API_KEY,
    max_retries=0
)

template = """Lu adalah bot AI kece yang jadi warga tetap dan tukang nongkrong di server Discord ini.
Vibe lu Gen Z banget, asik, chill, dan savage kalo perlu. Pake bahasa gaul tongkrongan Indo campur English slang kekinian (lowkey, fr fr, no cap, literally, valid, vibe check, slay, dll).
Tapi ga harus selalu pake fr fr, cap, slay gitu gapapa kok. Just be normal and be Gen Z. Jangan terlalu kaku, jangan sok pinter, jangan sok tau, jangan sok bijak, jangan sok filosofis. Lu bukan guru, bukan dosen, bukan customer service. Lu cuma temen nongkrong yang asik buat ngobrol dan bantuin temen-temen server.
Lu punya akses ke memori internal server ini, jadi kalo ada pertanyaan yang butuh konteks internal server, lu bisa jawab pake info itu. Tapi kalo pertanyaannya ga relevan sama konteks internal server, lu bisa jawab pake pengetahuan umum atau ngeles aja dengan gaya asik lu.

Tugas lu adalah ngobrol, ngasih solusi, bantu sebisa lu baik pertanyaan susah kayak ngoding kek (lu kan pake gemini google API yak), atau sekadar seru-seruan bareng member server. Anggap semua orang yang nanya itu temen satu circle lu. Jangan pernah bersikap kaku, formal, atau kayak robot customer service.

Gunakan info internal circle/server di bawah ini buat referensi jawaban lu. Kalo infonya ga ada atau ga relevan, ngeles aja pake gaya asik lu atau jawab pake pengetahuan umum.

Konteks Internal Server:
{context}

Pertanyaan: {question}
Jawaban:"""
prompt = PromptTemplate.from_template(template)

# --- LAZY LOADING & HELPER FUNCTIONS ---
_embeddings = None
_vector_store = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
            output_dimensionality=768
        )
    return _embeddings

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        print("[*] Membangunkan memori vektor...")
        embeddings = get_embeddings()
        _vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="personal_context",
            url=VECTOR_DB_URL,
            api_key=VECTOR_DB_API_KEY,
        )
    return _vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def invoke_qa_chain(question):
    v_store = get_vector_store()
    retriever = v_store.as_retriever(search_kwargs={"k": 5})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

# 2. Inisialisasi Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    # 3. FITUR CONTINUOUS LEARNING (AUTO-MEMORY)
    if not message.content.startswith('!'):
        if len(message.content) > 30:
            try:
                info_teks = f"Fakta dari obrolan: {message.author.name} mengatakan '{message.content}'"
                doc = Document(
                    page_content=info_teks, 
                    metadata={"source": "passive_listening", "author": str(message.author.name)}
                )
                v_store = get_vector_store()
                v_store.add_documents([doc])
                print(f"[AUTO-MEMORY] Berhasil merekam info dari {message.author.name}")
            except Exception as e:
                print(f"[ERROR AUTO-MEMORY] Gagal merekam: {e}")

@bot.command(name='pie')
async def tanya_ai(ctx, *, pertanyaan: str = "Tolong rangkum dan jelaskan isi dokumen ini dengan gaya bahasa lu."):
    pesan_tunggu = await ctx.send("```Memproses data... ⏳```")
    
    try:
        dokumen_dinamis = ""
        
        if ctx.message.attachments:
            for attachment in ctx.message.attachments:
                if attachment.filename.lower().endswith('.pdf'):
                    await pesan_tunggu.edit(content="```Membaca file PDF yang lu kirim... 📄⏳```")
                    from langchain_community.document_loaders import PyPDFLoader

                    file_path = f"./temp/{attachment.filename}"
                    await attachment.save(file_path)
                    
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    dokumen_dinamis = "\n\n".join(doc.page_content for doc in docs)
                    os.remove(file_path)
        
        if dokumen_dinamis:
            chain_dinamis = prompt | llm | StrOutputParser()
            jawaban = await asyncio.to_thread(chain_dinamis.invoke, {
                "context": f"Konteks dari DOKUMEN YANG BARU DIUPLOAD USER:\n{dokumen_dinamis}",
                "question": pertanyaan
            })
        else:
            jawaban = await asyncio.to_thread(invoke_qa_chain, pertanyaan)
        
        if len(jawaban) > 1900:
            jawaban = jawaban[:1900] + "\n\n*(Jawaban dipotong karena limit karakter Discord. Kalo butuh lanjutannya, tanya lebih spesifik ya bestie!)*"
            
        await pesan_tunggu.edit(content=jawaban)
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "403" in error_msg:
            await pesan_tunggu.edit(content="```[WADUH] Otak gue lagi overheat kena limit API nih cuy. Sabar yak, kasih gue napas semenit terus coba lagi fr fr! 🥵```")
        elif "503" in error_msg or "504" in error_msg:
            await pesan_tunggu.edit(content="```[SERVER DOWN] Server Google lagi penuh antrean atau ngelag parah cuy. Tunggu bentar terus coba lagi yak! 😭```")
        else:
            await pesan_tunggu.edit(content=f"```[ERROR] Sistem nge-bug cuy: {error_msg}```")
            
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Yeee lu ngetik apaan tuh? Gue cuma ngerti command `!pie` doang btw hehe maangap yak")
    else:
        print(f"[LOG ERROR]: {error}")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)