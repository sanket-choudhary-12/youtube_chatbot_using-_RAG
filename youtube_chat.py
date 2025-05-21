import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Set your Google API key
from dotenv import load_dotenv
load_dotenv()

# Load Gemini and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Relaxed Prompt Template
prompt = PromptTemplate(
    template="""
    You are a helpful assistant. Use the context from the YouTube video transcript to answer the question.
    If the context is insufficient, say you don't know, but try to be helpful.

    Context:
    {context}

    Question: {question}
    """,
    input_variables=['context', 'question']
)

def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    elif "/shorts/" in url:
        return url.split("/shorts/")[-1].split("?")[0]
    return url.strip()  # assume raw ID

def get_transcript(video_id: str) -> str:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    return " ".join(chunk["text"] for chunk in transcript_list)

def create_retriever_from_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Streamlit App UI
st.set_page_config(page_title="YouTube Video Q&A", layout="centered")
st.title("📺 YouTube Video Q&A using Gemini + RAG")

youtube_url = st.text_input("Enter YouTube Video URL or ID:")

if youtube_url:
    video_id = extract_video_id(youtube_url)
    with st.spinner("🔍 Fetching transcript..."):
        try:
            transcript = get_transcript(video_id)
            st.success("✅ Transcript fetched successfully!")
            st.caption("ℹ️ Transcript (first 500 chars):")
            st.code(transcript[:500], language="text")

            retriever = create_retriever_from_transcript(transcript)

            # Build Main Chain
            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            main_chain = parallel_chain | prompt | llm | StrOutputParser()

            st.subheader("Ask a question about the video:")
            user_question = st.text_input("Your question")

            if st.button("Ask") and user_question:
                with st.spinner("💬 Generating answer..."):
                    answer = main_chain.invoke(user_question)
                    st.markdown(f"### ✅ Answer:\n{answer}")

                    # Optional: Show retrieved context for debugging
                    if st.checkbox("🔍 Show retrieved context"):
                        retrieved = retriever.invoke(user_question)
                        st.code(format_docs(retrieved), language="text")

        except Exception as e:
            st.error(f"❌ Error fetching transcript: {e}")
