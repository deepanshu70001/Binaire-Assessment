"""
app.py — Streamlit Web Application for Pico-8 RAG Code Assistant.

Provides an interactive fantasy console coding assistant powered by
FAISS vector retrieval and Groq LLM inference.
"""

import os
# Prevent OpenMP multiple runtime conflict warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
from dotenv import load_dotenv, set_key
from src.data_loader import load_pico8_csv
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Page Configuration
st.set_page_config(
    page_title="Pico-8 RAG Code Assistant",
    layout="wide",
    page_icon="🕹️",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
    }
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# --- Cached Vector Store & RAG Initializers ---
@st.cache_resource(show_spinner="Loading Vector Store & Embedding Model...")
def get_vector_store(persist_dir: str = "faiss_store") -> FaissVectorStore:
    store = FaissVectorStore(persist_dir)
    if store.is_built():
        store.load()
    return store


@st.cache_resource(show_spinner=False)
def get_rag_engine(persist_dir: str, model_name: str, api_key: str):
    store = get_vector_store(persist_dir)
    os.environ["GROQ_API_KEY"] = api_key
    return RAGSearch(persist_dir=persist_dir, llm_model=model_name, vector_store=store)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    # Model Selection
    available_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3.8-27b",
        "mixtral-8x7b-32768",
    ]
    selected_model = st.selectbox(
        "Groq Model",
        available_models,
        index=0,
        help="Select the LLM used for generating Pico-8 Lua code."
    )

    # API Key Management
    current_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=current_key,
        type="password",
        placeholder="gsk_..."
    )
    if st.button("💾 Save Key", use_container_width=True):
        if api_key.strip():
            set_key(".env", "GROQ_API_KEY", api_key.strip())
            os.environ["GROQ_API_KEY"] = api_key.strip()
            st.success("API Key saved to .env!")
        else:
            st.warning("Please enter a valid key.")

    st.markdown("---")

    # Index Status
    store = get_vector_store("faiss_store")
    if store.is_built() and store.index is not None:
        st.success(f"🟢 Database Ready: **{store.index.ntotal:,}** vectors")
    else:
        st.warning("🟠 Database not found. Upload CSV below.")

    # CSV Upload & Rebuild
    st.markdown("### 📂 Dataset Ingestion")
    uploaded_file = st.file_uploader("Upload Pico-8 CSV", type=["csv"])
    if uploaded_file and st.button("🔨 Build / Rebuild Database", use_container_width=True):
        with st.spinner("Processing CSV and generating vector embeddings..."):
            os.makedirs("data", exist_ok=True)
            save_path = os.path.join("data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            docs = load_pico8_csv(save_path)
            store.build_from_documents(docs)
            st.cache_resource.clear()
            st.success(f"Built FAISS store with {store.index.ntotal} vectors!")
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #64748b;'>"
        "<b>Pico-8 Console Specs:</b><br>"
        "• Resolution: 128×128 px<br>"
        "• Palette: 16 colors<br>"
        "• Audio: 4 channels, 64 SFX<br>"
        "• Code limit: 8192 tokens"
        "</div>",
        unsafe_allow_html=True
    )

# --- Main Layout ---
st.markdown('<div class="main-header">🕹️ Pico-8 RAG Code Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Generate authentic Pico-8 Lua code conditioned on 100+ real games using FAISS semantic search and Groq high-speed LLMs.'
    '</div>',
    unsafe_allow_html=True
)

# Badges
st.markdown("""
<div>
    <span class="badge-pill">⚡ Sub-20ms Retrieval</span>
    <span class="badge-pill">🎮 100 Curated Games</span>
    <span class="badge-pill">🔍 11,000+ Vector Chunks</span>
    <span class="badge-pill">🚀 Groq LLM Inference</span>
</div>
<br>
""", unsafe_allow_html=True)

# Prompt Pre-fill buttons
st.markdown("**Quick Prompts:**")
col1, col2, col3, col4 = st.columns(4)
query_seed = ""
if col1.button("🏃 Platformer Physics"):
    query_seed = "Create a simple platformer with jumping, gravity, and ground collision"
if col2.button("🚗 Car Drift Steering"):
    query_seed = "Implement top-down car acceleration, turning, and drift skidmarks"
if col3.button("💥 Particle Explosion"):
    query_seed = "Create a particle explosion effect when an object is destroyed"
if col4.button("🏓 Bouncing Ball & Paddle"):
    query_seed = "Create a breakout or pong bouncing ball with paddle collision and score"

# Query Input
effective_query = st.text_area(
    "Describe what you want to build in Pico-8:",
    value=query_seed,
    placeholder="e.g., Create a player character that shoots laser projectiles with screen bounds check...",
    height=100
)

col_ctrl1, col_ctrl2 = st.columns([1, 4])
with col_ctrl1:
    top_k = st.slider("Reference Games (top_k)", min_value=1, max_value=8, value=3)

# Execution
if st.button("🚀 Generate Pico-8 Code", type="primary", use_container_width=True):
    if not effective_query.strip():
        st.warning("Please enter a query or select one of the quick prompts above.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ Please provide a Groq API Key in the sidebar.")
    elif not store.is_built():
        st.error("⚠️ FAISS vector store is not built. Please upload a dataset in the sidebar.")
    else:
        with st.spinner(f"Searching {store.index.ntotal} vectors & generating code with {selected_model}..."):
            try:
                rag = get_rag_engine("faiss_store", selected_model, os.getenv("GROQ_API_KEY"))
                result_data = rag.query_with_sources(effective_query, top_k=top_k)

                # Output code
                st.markdown("### 🎮 Generated Pico-8 Code")
                st.markdown(result_data["response"])

                # Output retrieved references
                sources = result_data.get("sources", [])
                if sources:
                    st.markdown("---")
                    st.markdown(f"### 📚 Retrieved Context ({len(sources)} Reference Games)")
                    for i, src in enumerate(sources, 1):
                        score_pct = int(src["score"] * 100)
                        with st.expander(f"#{i}: {src['game_name']} (by {src['author']}) — Similarity Match: {score_pct}% | ❤️ {src['like_count']} likes"):
                            if src.get("description"):
                                st.markdown(f"**Description:** {src['description']}")
                            if src.get("source_url"):
                                st.markdown(f"**URL:** [{src['source_url']}]({src['source_url']})")
                            st.code(src["snippet"], language="lua")

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")