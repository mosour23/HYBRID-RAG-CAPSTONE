import sys
import types
import importlib
import importlib.util
import time
import os
import json
import streamlit as st
from html import escape

# Compatibility hack for Python 3.12 if some libs import 'imp'
class MockImp:
    @staticmethod
    def find_module(name, path=None):
        spec = importlib.util.find_spec(name, path)
        if spec is None:
            raise ImportError(f"No module named {name}")
        return None, spec.origin, None

    @staticmethod
    def load_module(name, file, pathname, description):
        return importlib.import_module(name)

sys.modules.setdefault('imp', MockImp())
try:
    sys.modules['imp'].new_module = types.ModuleType
except Exception:
    pass

# Import project components (use the real class names)
from src.router.fuzzy_controller import FuzzyLogicController
from src.analyzer.query_analyzer import QueryAnalyzer
from src.pipelines.pipeline_factory import PipelineFactory

# Streamlit page config
st.set_page_config(page_title="Hybrid-RAG AI", page_icon="✨", layout="centered")

# Modern CSS
st.markdown(
    """
    <style>
    /* Hide default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg,#0f1724 0%,#071327 100%);
        color: #e6eef8;
        min-height: 100vh;
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }

    .chat-card {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(2,6,23,0.6);
        max-width: 900px;
        margin: 24px auto;
    }

    .bubble {
        display: flex;
        gap: 12px;
        margin: 8px 0;
        align-items: flex-start;
    }
    .avatar {
        min-width: 40px;
        height: 40px;
        border-radius: 12px;
        display:flex;align-items:center;justify-content:center;
        font-size:18px;
    }
    .bubble .content {
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 78%;
        line-height: 1.45;
        box-shadow: 0 6px 18px rgba(2,6,23,0.5);
    }
    .user .avatar { background: linear-gradient(90deg,#334155,#0ea5a2); }
    .assistant .avatar { background: linear-gradient(90deg,#7c3aed,#06b6d4); }
    .user .content { background: linear-gradient(180deg, rgba(4,120,87,0.12), rgba(2,6,23,0.06)); color: #dff7ef; }
    .assistant .content { background: linear-gradient(180deg, rgba(124,58,237,0.12), rgba(2,6,23,0.06)); color: #f1e9ff; }

    .meta { font-size: 12px; opacity: 0.7; margin-top:6px; }

    .stSidebar .sidebar-content {
        background: linear-gradient(180deg,#021124,#042033);
        border-radius: 12px;
        padding: 12px;
    }

    .stChatInput>div>div>div>div {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.03) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"route": "-", "ttft": "-", "complexity": "-"}

@st.cache_resource
def load_system():
    analyzer = QueryAnalyzer()
    fuzzy = FuzzyLogicController()

    class RouterWrapper:
        def __init__(self, analyzer, fuzzy):
            self.analyzer = analyzer
            self.fuzzy = fuzzy

        def evaluate(self, prompt_text: str):
            sd = self.analyzer.calculate_semantic_density(prompt_text)
            mhr = self.analyzer.calculate_multihop_requirement(prompt_text)
            result = self.fuzzy.decide_route(sd, mhr)
            return result['decision'], result['score']

    return RouterWrapper(analyzer, fuzzy)

router = load_system()

# Sidebar
with st.sidebar:
    st.title("⚙️ Model Diagnostics")
    st.caption("Backend Telemetry")
    st.divider()
    st.metric(label="🛤️ Active Route", value=st.session_state.metrics["route"])
    st.metric(label="⏱️ Latency (TTFT)", value=st.session_state.metrics["ttft"])
    st.metric(label="🧠 Density Score", value=st.session_state.metrics["complexity"])

    # RAG index info + preload
    try:
        rag = PipelineFactory.get_pipeline("OP-RAG")
        idx_count = getattr(rag.index, 'ntotal', 0)
    except Exception:
        idx_count = 0
    st.caption(f"RAG index documents: {idx_count}")
    if st.button("Preload dataset into RAG"):
        dataset_path = "data/test_dataset.json"
        if os.path.exists(dataset_path):
            with st.spinner("Preloading dataset into RAG index..."):
                try:
                    with open(dataset_path, 'r', encoding='utf-8') as fh:
                        ds = json.load(fh)
                        corpus = ds.get('corpus', [])
                        if corpus:
                            rag.ingest_documents(corpus)
                            st.success(f"Preloaded {len(corpus)} documents into RAG index.")
                        else:
                            st.warning("Dataset found but no corpus entries present.")
                except Exception as e:
                    st.error(f"Failed to preload dataset: {e}")
        else:
            st.warning("data/test_dataset.json not found. Run generate_benchmark_dataset.py first.")

# Welcome
if not st.session_state.messages:
    st.markdown("<h1 style='text-align: center; color: #ECECFA; padding-top: 50px;'>How can I help you today?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8E8EA0;'>Ask any complex networking query, and our Hybrid-RAG model will route it dynamically.</p>", unsafe_allow_html=True)

# Chat area
st.markdown('<div class="chat-card">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role = message.get("role", "user")
    avatar = "🧑‍💻" if role == "user" else "✨"
    css_role = "user" if role == "user" else "assistant"
    content = message.get("content", "")
    meta = message.get("meta", "")

    safe_content = escape(str(content)).replace('\n', '<br>')
    safe_meta = escape(str(meta)).replace('\n', '<br>')

    html = f"""
    <div class='bubble {css_role}'>
      <div class='avatar'>{avatar}</div>
      <div class='content'>
        <div>{safe_content}</div>
        {f"<div class='meta'>{safe_meta}</div>" if meta else ''}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Message Hybrid-RAG..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(escape(prompt).replace('\n', '<br>'), unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            try:
                route_decision, complexity_score = router.evaluate(prompt)
                pipeline = PipelineFactory.get_pipeline(route_decision)
                # pipelines implement generate(query, **kwargs) -> (text, ttft)
                response_text, ttft_out = pipeline.generate(prompt)
            except Exception as e:
                response_text = f"⚠️ Error processing query: {str(e)}"
                route_decision = "Error"
                complexity_score = "N/A"
                ttft_out = 0.0

            ttft_value = round((time.time() - start_time) * 1000, 2)

            st.session_state.metrics["route"] = str(route_decision)
            st.session_state.metrics["ttft"] = f"{ttft_value} ms"
            st.session_state.metrics["complexity"] = str(complexity_score)

            st.markdown(escape(response_text).replace('\n', '<br>'), unsafe_allow_html=True)

            meta_text = f"⚡ Routed via {route_decision} in {ttft_value}ms"
            st.caption(f"_{meta_text}_")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "meta": meta_text,
            })

        st.experimental_rerun()
