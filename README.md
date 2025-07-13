# 🎥 YouTube Video QA Assistant

This project is an **AI-powered assistant** that can answer your questions and summarize content from any YouTube video — as long as it has captions (subtitles) enabled.

It uses the video transcript to build a **semantic search knowledge base**, and then answers your question using a powerful LLM (Gemini).

---

## 🚀 Features

✅ Retrieve and process YouTube video transcripts (captions)  
✅ Ask custom questions about the video content  
✅ Summarize the entire video in a single click  
✅ Simple and interactive Streamlit web UI  
✅ Error handling for videos without captions

---

## 🛠️ Tech Stack

- **Streamlit** — For building an easy-to-use web interface
- **YouTube Transcript API** — To extract video captions
- **LangChain** — To manage document splitting, embeddings, retrieval, and chain execution
- **Google Generative AI (Gemini)** — For generating natural language answers
- **FAISS** — For vector-based similarity search
- **Python** — Core language

---

## 💡 How it works

1️⃣ You enter a YouTube video ID (e.g., `Gfr50f6ZBvo`).  
2️⃣ The app fetches the video transcript using `youtube-transcript-api`.  
3️⃣ Transcript text is split into chunks using LangChain's recursive splitter.  
4️⃣ Each chunk is converted into vector embeddings (Google Gemini embeddings).  
5️⃣ Your question is matched to the most relevant transcript parts using FAISS vector search.  
6️⃣ A prompt is created to guide the LLM to answer **only from the context**.  
7️⃣ The Gemini LLM generates the final answer or summary.

---

## ⚙️ Setup

### 1️⃣ Clone this repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit youtube-transcript-api langchain langchain-google-genai faiss-cpu python-dotenv
```

### 3️⃣ Set up your environment

Create a `.env` file if needed (for example, to store API keys if required by Google GenAI).

### 4️⃣ Run the app

```bash
streamlit run app.py
```

---

## 💬 Usage

- **Video ID field**: Enter the YouTube video ID (the string after `v=` in the URL).  
- **Question field**: Type your question related to the video.  
- **Get Answer button**: Retrieves context and answers your question.  
- **Summarize Entire Video button**: Summarizes the full content of the video.

---

## ⚠️ Important

- This app only works with videos that have **captions enabled** (either uploaded or auto-generated).
- The more accurate the captions, the better the results.

---

## 👨‍💻 Created by

**Vishwait VM**  
📧 vishwaitmall50@gmail.com

---

## ⭐ Contributions

Feel free to open issues or submit pull requests to improve the app!

---

## 📄 License

This project is open-source and free to use under [MIT License](LICENSE).
