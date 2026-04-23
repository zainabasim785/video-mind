import os
import re
from crewai.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

# Load API key: use Streamlit secrets on cloud, dotenv locally
try:
    import streamlit as st
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        from dotenv import load_dotenv
        load_dotenv()
except Exception:
    from dotenv import load_dotenv
    load_dotenv()

CHROMA_DB_DIR = "chroma_db"

def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    # Get the first available transcript (e.g. English, or auto-generated)
    transcript = next(iter(transcript_list))
    data = transcript.fetch()
    return " ".join([entry.text for entry in data])

def build_vector_store(transcript_text: str, video_id: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([Document(page_content=transcript_text, metadata={"video_id": video_id})])
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print(f"Vector store built with {len(chunks)} chunks.")
    return vector_store

@tool("VideoRAGTool")
def rag_search_tool(query: str) -> str:
    """Search the YouTube video transcript for information related to the query."""
    
    if not os.path.exists(CHROMA_DB_DIR):
        return "No video loaded yet. Please process a video first."
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    results = vector_store.similarity_search(query, k=4)
    
    if not results:
        return "No relevant information found in the video."
    
    output = "Relevant sections from the video transcript:\n"
    for i, doc in enumerate(results, 1):
        output += f"\n--- Section {i} ---\n{doc.page_content}\n"
    
    return output