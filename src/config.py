import os
from dotenv import load_dotenv

# Memuat variabel rahasia dari file .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
VECTOR_DB_API_KEY = os.getenv("VECTOR_DB_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Pengecekan keamanan: hentikan program jika ada API Key yang lupa diisi
if not all([GOOGLE_API_KEY, VECTOR_DB_URL, VECTOR_DB_API_KEY]):
    raise ValueError("Kredensial API tidak lengkap! Pastikan file .env sudah diisi dengan benar.")