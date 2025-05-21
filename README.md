# 📺 YouTube Video Q&A using Gemini + RAG

This Streamlit app allows users to ask questions about a YouTube video. It fetches the transcript, indexes it using FAISS, and answers questions using Google's Gemini (via LangChain).

---

## 🚀 Features

- Enter any YouTube video URL or ID
- Automatically fetch the transcript (English)
- Perform semantic search on transcript chunks
- Use Google's Gemini (`gemini-2.0-flash`) to answer based on context
- Display context used for the answer (optional toggle)

---

## 🧠 Technologies Used

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/youtube-gemini-qa.git
cd youtube-gemini-qa
