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
import concurrent.futures
import requests

def get_free_proxies():
    try:
        url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=all&ssl=all&anonymity=all"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=5)
        proxies = response.read().decode('utf-8').strip().split('\r\n')
        return [p for p in proxies if p]
    except Exception:
        return []

class TimeoutSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault('timeout', 3)
        return super().request(*args, **kwargs)

def try_fetch_with_proxy(video_id, proxy):
    from youtube_transcript_api.proxies import GenericProxyConfig
    try:
        proxy_config = GenericProxyConfig(http_url=proxy, https_url=proxy)
        http_client = TimeoutSession()
        api_with_proxy = YouTubeTranscriptApi(proxy_config=proxy_config, http_client=http_client)
        transcript_list = api_with_proxy.list(video_id)
        transcript = next(iter(transcript_list))
        data = transcript.fetch()
        return " ".join([entry['text'] for entry in data])
    except Exception:
        return None

def get_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    
    try:
        transcript_list = api.list(video_id)
        transcript = next(iter(transcript_list))
        data = transcript.fetch()
        return " ".join([entry['text'] for entry in data])
    except Exception as e:
        error_msg = str(e)
        if "blocking requests from your IP" not in error_msg and "YouTubeRequestFailed" not in str(type(e).__name__):
            raise e
            
    # 2. If IP is blocked, try using fast parallel proxies
    proxies_list = get_free_proxies()
    random.shuffle(proxies_list)
    
    if proxies_list:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(try_fetch_with_proxy, video_id, proxy): proxy for proxy in proxies_list[:20]}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    return result

    raise Exception("YouTube blocked the server IP, and all free proxy attempts failed or timed out. Please try again later or run locally.")

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