# 🎥 YouTube RAG Assistant

### Ask Questions About Any YouTube Video Using AI

An AI-powered **Retrieval-Augmented Generation (RAG)** application that transforms any YouTube video into an interactive knowledge system.
Simply paste a video URL, and the system allows you to ask questions based on the video’s transcript with precise timestamp-backed answers.

---


## 🧠 How It Works

This project implements a **full RAG pipeline**:

1. **Transcript Extraction**
   Fetches subtitles from YouTube using `youtube-transcript-api`.

2. **Document Processing**
   Converts transcript into a structured document with timestamp metadata.

3. **Chunking Strategy**
   Splits text into overlapping chunks while preserving time references.

4. **Embeddings Generation**
   Uses HuggingFace models to convert text into vector representations.

5. **Vector Database (FAISS)**
   Stores embeddings for fast similarity-based retrieval.

6. **Retrieval + LLM (Groq)**
   Retrieves relevant chunks and generates answers using LLM.

7. **Answer + Video Sync**
   Displays answer along with the exact video segment.

---

## ✨ Features

* 🔍 Ask questions about any YouTube video
* ⏱️ Timestamp-aware answers
* 🧠 Semantic search using embeddings
* 📊 Intelligent document chunking
* ⚡ Fast retrieval using FAISS
* 🎥 Embedded video playback from relevant timestamp
* 🖥️ Interactive UI with Streamlit

---

## 🛠️ Tech Stack

| Category      | Technology             |
| ------------- | ---------------------- |
| Frontend UI   | Streamlit              |
| Backend Logic | Python                 |
| RAG Framework | LangChain              |
| Embeddings    | HuggingFace            |
| Vector DB     | FAISS                  |
| LLM           | Groq                   |
| Data Source   | YouTube Transcript API |

---


## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/youtube-rag-assistant.git
cd youtube-rag-assistant
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# OR CMD
venv\Scripts\activate.bat
```

---

### 3️⃣ Install Dependencies

manually:

```bash
pip install streamlit youtube-transcript-api langchain langchain-core langchain-community langchain-huggingface langchain-groq faiss-cpu sentence-transformers python-dotenv
```

---

### 4️⃣ Setup Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Get your API key from: https://console.groq.com

---

### 5️⃣ Run the Application

```bash
streamlit run VideoChatting_RAG.py
```

---

## 🎯 Usage

1. Paste any YouTube video URL
2. Wait for processing pipeline to complete
3. Ask questions about the video
4. Get answers with timestamps + video playback

---

## 📊 Example Query

```
What are the document chunking strategies?
```

✔️ Output:

* Detailed explanation
* Source timestamps
* Embedded video segment

---


## 🔮 Future Improvements

* 🎙️ Support for audio transcription (no subtitles required)
* 🌐 Multi-language support
* 📄 Upload custom documents (PDF, DOCX)
* 🧠 Improved semantic chunking strategies
* ☁️ Deployment (Streamlit Cloud / AWS)
* 📱 Mobile-friendly UI

---

## 🧩 Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector Search & Embeddings
* Semantic Information Retrieval
* LLM-based Question Answering
* Real-time AI Applications

---


## 👩‍💻 Author

**Meghana L**
AI Enthusiast

---

## ⭐ Show Your Support

If you found this project useful, please ⭐ the repository!

---
