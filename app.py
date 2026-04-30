import streamlit as st
import shutil
import os

# --- SQLite Fix for ChromaDB on Streamlit Cloud ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from rag_tool import extract_video_id, get_transcript, build_vector_store
from crew import run_crew

# --- Page Config ---
st.set_page_config(
    page_title="VideoMind",
    page_icon="🎬",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 400;
    }

    .stApp {
        background-color: #1c1a19; /* Warm dark espresso */
        color: #b5aca3; /* Soft beige-grey */
    }

    .main-header {
        text-align: center;
        padding: 4rem 0 2.5rem 0;
        border-bottom: 1px solid rgba(230, 222, 195, 0.05);
        margin-bottom: 3rem;
    }

    .main-header h1 {
        font-family: 'Cormorant Garamond', serif;
        font-size: 3.8rem;
        font-weight: 400;
        color: #e6dec3; /* Light cream/beige */
        letter-spacing: 0.1em;
        margin-bottom: 0.2rem;
    }

    .main-header p {
        color: #9c9288; /* Muted beige */
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    /* Clean transparent container */
    .url-box {
        background-color: transparent;
        padding: 0;
        margin: 2rem 0;
    }

    /* Answer box as elegant card */
    .answer-box {
        background-color: #262322; /* Warm dark grey card */
        border: 1px solid rgba(230, 222, 195, 0.08);
        border-left: 4px solid #cca47c; /* Chic tan accent */
        border-radius: 8px;
        padding: 1.8rem 2.5rem;
        margin-top: 1rem;
        color: #e6dec3; /* Light cream */
        font-size: 1.05rem;
        line-height: 1.7;
        font-weight: 400;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    /* Status indicator */
    .status-pill {
        display: inline-block;
        background-color: rgba(204, 164, 124, 0.08); /* Tan tint */
        color: #cca47c; /* Tan */
        border: 1px solid rgba(204, 164, 124, 0.2);
        border-radius: 20px;
        padding: 0.4rem 1.2rem;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* High-end minimalist inputs */
    .stTextInput > div > div > input {
        background-color: #262322 !important; 
        border: 1px solid rgba(230, 222, 195, 0.15) !important;
        border-radius: 8px !important;
        color: #e6dec3 !important; /* Light cream */
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        padding: 0.8rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #cca47c !important; /* Tan accent */
        background-color: #2e2a29 !important;
        box-shadow: 0 0 0 2px rgba(204, 164, 124, 0.15) !important;
    }

    /* Minimalist buttons */
    .stButton > button {
        background-color: #cca47c !important; /* Chic tan */
        color: #1f1915 !important; /* Dark brown text */
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 2.5rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .stButton > button:hover {
        background-color: #e3bc94 !important; /* Lighter tan */
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Sidebar */
    div[data-testid="stSidebarContent"] {
        background-color: #161413; /* Deep espresso */
        border-right: 1px solid rgba(230, 222, 195, 0.05);
    }

    .sidebar-section {
        background-color: transparent;
        border: none;
        padding: 0;
        margin-bottom: 2.5rem;
        font-size: 0.9rem;
        color: #9c9288;
        line-height: 1.6;
        font-weight: 400;
    }

    hr {
        border-color: rgba(230, 222, 195, 0.05) !important;
    }

    .stSpinner > div {
        border-top-color: #cca47c !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>VideoMind</h1>
    <p>Intelligent Audio & Visual Analysis</p>
</div>
""", unsafe_allow_html=True)

# --- Session State ---
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "response_style" not in st.session_state:
    st.session_state.response_style = "Concise & Clear"

# --- Sidebar ---
with st.sidebar:
    st.markdown("### VideoMind")
    st.markdown('<div class="sidebar-section">Paste a YouTube URL, load the video, then ask questions about it.</div>', unsafe_allow_html=True)

    if st.session_state.video_loaded:
        st.markdown(f'<div class="status-pill">✓ Video loaded</div>', unsafe_allow_html=True)
        st.caption(st.session_state.video_url[:50] + "...")
        if st.button("Load new video"):
            st.session_state.video_loaded = False
            st.session_state.video_url = ""
            st.session_state.chat_history = []
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")
            st.rerun()
            
        st.markdown("---")
        with st.expander("⚙️ Style Settings", expanded=True):
            st.session_state.response_style = st.selectbox(
                "Agent Personality", 
                ["Concise & Clear", "Detailed & Analytical", "Explain like I'm 5", "High-End Consultant", "Like a Pirate"]
            )
            st.caption("Tweaking this alters how the AI writes its answers.")
    
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Built with LangChain · CrewAI · Gemini · ChromaDB</div>', unsafe_allow_html=True)

# --- Main Area ---
if not st.session_state.video_loaded:
    st.markdown('<div class="url-box">', unsafe_allow_html=True)
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Load Video"):
        if not url:
            st.warning("Please enter a YouTube URL.")
        else:
            video_id = extract_video_id(url)
            if not video_id:
                st.error("Invalid YouTube URL. Please check and try again.")
            else:
                with st.spinner("Fetching transcript and building knowledge base..."):
                    try:
                        transcript = get_transcript(video_id)
                        build_vector_store(transcript, video_id)
                        st.session_state.video_loaded = True
                        st.session_state.video_url = url
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    with st.expander("📺 View Source Video", expanded=False):
        st.video(st.session_state.video_url)

    # --- Chat Interface ---
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item.get("display_question", item.get("question", "")))
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask anything about the video...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Agents working..."):
                # Inject style into the prompt
                styled_question = f"{question}\n\n(IMPORTANT INSTRUCTION: You MUST write your final answer adopting this specific Persona/Style: {st.session_state.response_style})"
                answer = run_crew(styled_question)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        
        st.session_state.chat_history.append({
            "display_question": question,
            "question": styled_question,
            "answer": answer
        })