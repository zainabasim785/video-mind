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

import urllib.request
import random

def get_free_proxies():
    try:
        url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=all&ssl=all&anonymity=all"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        proxies = response.read().decode('utf-8').strip().split('\r\n')
        return [p for p in proxies if p]
    except Exception:
        return []

def get_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    
    try:
        transcript_list = api.list(video_id)
        transcript = next(iter(transcript_list))
        data = transcript.fetch()
        return " ".join([entry['text'] for entry in data])
    except Exception as e:
        error_msg = str(e)
        if "blocking requests from your IP" in error_msg or "YouTubeRequestFailed" in str(type(e).__name__):
            raise Exception(
                "❌ **YouTube IP Blocked** \n\n"
                "YouTube has permanently blocked Streamlit Cloud's servers from downloading transcripts. "
                "Because free proxies cause the app to freeze/hang for 10+ minutes, we have disabled them. \n\n"
                "**To fix this, you must run this app locally on your computer:**\n"
                "1. Download the code\n"
                "2. Run `streamlit run app.py`\n"
                "Your local internet IP is not blocked by YouTube, so it will work perfectly!"
            )
        else:
            raise e

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