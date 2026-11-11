import os
from dotenv import load_dotenv

# Memuat variabel rahasia dari file .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
VECTOR_DB_API_KEY = os.getenv("VECTOR_DB_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Hentikan program lebih awal jika ada environment variable wajib yang belum diisi.
missing_variables = [
    name
    for name, value in {
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "VECTOR_DB_URL": VECTOR_DB_URL,
        "VECTOR_DB_API_KEY": VECTOR_DB_API_KEY,
        "DISCORD_TOKEN": DISCORD_TOKEN,
    }.items()
    if not value
]

if missing_variables:
    raise ValueError(
        "Environment variables belum diisi: " + ", ".join(missing_variables)
    )