# 🥧 Helpie — AI Discord Bot

> A RAG-based AI assistant built for Discord, powered by LangChain, Google Gemini, and Qdrant Vector Store.

---

## 🚀 Features
- **RAG Architecture**: Retrieves context dynamically from local knowledge bases using Qdrant.
- **Google Gemini Integration**: Powered by lightweight and fast Gemini models for intelligent natural language processing.
- **Discord Integration**: Fully interactive Discord bot built using `discord.py`.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Frameworks:** Discord.py, LangChain, Flask
- **AI & Vector DB:** Google Gemini API, Qdrant Cloud

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/safiranaa/helpie---AI-discord-bot.git
   cd helpie

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate

3. **Install depedencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the bot locally:**
   ```bash
   python -m src.discord_bot

## 📦 Deployment
Easily deployable on cloud platforms like **Railway** by configuring the start command to:

```bash
python -m src.discord_bot
```

## 👩‍💻 Author
Safira Nabila Putri
