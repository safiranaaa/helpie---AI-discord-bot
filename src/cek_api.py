import google.generativeai as genai
from src.config import GOOGLE_API_KEY

# 1. Cek apakah API Key beneran masuk atau malah kosong (None)
api_awalan = str(GOOGLE_API_KEY)[:5] if GOOGLE_API_KEY else "KOSONG/NONE"
print(f"[*] Status API Key lu: {api_awalan}********")

# 2. Tarik data dari server Google
print("[*] Nanya ke server Google model apa aja yang kebuka buat API Key ini...")
genai.configure(api_key=GOOGLE_API_KEY)

try:
    ada_model = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" ✅ {m.name}")
            ada_model = True
            
    if not ada_model:
        print(" ❌ API Key lu valid, tapi ga punya akses ke model teks (generateContent).")
except Exception as e:
    print(f" ❌ GAGAL KONEK: {e}")