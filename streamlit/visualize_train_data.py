import streamlit as st
import json
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page config with custom styling
st.set_page_config(
    page_title="VLSP 2025 Traffic Sign Viewer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fancy styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: #fdfdfd;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    .question-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .question-card h3 {
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
    }

    .question-text {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        font-size: 1.1rem;
        font-weight: 500;
        color: #1c1c1c;
    }

    .image-container {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
        border: 2px solid #0369a1;
    }

    .image-container h3 {
        color: #0369a1;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .stats-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        color: #f0f0f0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 150px;
        text-align: center;
    }

    .stats-card p1 {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 2.0rem;
        font-weight: 700;
    }

    .stats-card p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.0rem;
        font-weight: 500;
    }

    .nav-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    .choice-correct {
        background: #ffffff;
        color: #1e1b4b;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #22c55e;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .choice-normal {
        background: #ffffff;
        color: #1e1b4b;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #e5e7eb;
        font-weight: 500;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .answer-box {
        background: #ffffff;
        color: #1e1b4b;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: 2px solid #8b5cf6;
    }

    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #fefefe;
    }

    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 8px;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .article-tag {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: #fdfdfd;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Data paths
TRAIN_DIR = "D:/VLSP-MLQA-TSR/data/processed/train_data"
# Path to legal document file
LAW_DIR = "D:/VLSP-MLQA-TSR/data/processed/law_db"

@st.cache_data
def load_law():
    with open(f"{LAW_DIR}/vlsp2025_law.json", encoding="utf-8") as f:
        return json.load(f)

try:
    law_data = load_law()
except FileNotFoundError:
    st.error("❌ Could not find the legal documents file.")
    law_data = []

@st.cache_data
def load_data():
    with open(f"{TRAIN_DIR}/vlsp_2025_train.json", encoding="utf-8") as f:
        return json.load(f)

try:
    data = load_data()
except FileNotFoundError:
    st.error("❌ Could not find the dataset. Please check the path.")
    st.stop()

# ==== Render logic for different question types ====

def get_article_text(law_id, article_id):
    for law in law_data:
        if law.get("id") == law_id:
            for article in law.get("articles", []):
                if article.get("id") == article_id:
                    title = article.get("title", "")
                    text = article.get("text", "")
                    return f"**{title}**\n\n{text.strip()}"
    return "⚠️ Article not found."

def render_multiple_choice(q):
    choices = q.get("choices", {})
    correct_key = q.get("answer", "")
    
    for key, val in choices.items():
        if key == correct_key:
            st.markdown(f"""
            <div class="choice-correct">
                ✅ <strong>{key}. {val}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="choice-normal">
                {key}. {val}
            </div>
            """, unsafe_allow_html=True)

def render_yes_no(q):
    answer = q.get("answer", "")
    st.markdown(f"""
    <div class="answer-box">
        📝 <strong>Answer:</strong> {answer}
    </div>
    """, unsafe_allow_html=True)

# ==== Session state ====
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# ==== Fancy Header ====
st.markdown("""
<div class="main-header">
    <h1>🚦 VLSP 2025 Traffic Sign Viewer</h1>
    <p>Explore traffic sign questions with interactive visualizations and smart filtering</p>
</div>
""", unsafe_allow_html=True)

# ==== Progress indicator ====
progress = (st.session_state.current_index + 1) / len(data)
st.progress(progress)
st.markdown(f"<div style='text-align: center; color: #666; margin-bottom: 2rem;'>Question {st.session_state.current_index + 1} of {len(data)} ({progress:.1%} complete)</div>", unsafe_allow_html=True)

# ==== Navigation with fancy buttons ====
st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1], gap="medium")

with col1:
    if st.button("⬅ Previous", disabled=(st.session_state.current_index == 0), key="prev_btn", use_container_width=True):
        st.session_state.current_index = max(0, st.session_state.current_index - 1)

with col2:
    st.session_state.current_index = st.slider(
        "🎯 Jump to question",
        min_value=0,
        max_value=len(data) - 1,
        value=st.session_state.current_index,
        key="question_slider"
    )

with col3:
    if st.button("Next ➡", disabled=(st.session_state.current_index == len(data) - 1), key="next_btn", use_container_width=True):
        st.session_state.current_index = min(len(data) - 1, st.session_state.current_index + 1)

# ==== Current question ====
current_q = data[st.session_state.current_index]

# ==== Main content layout ====
col_img, col_content = st.columns([1.2, 1.8], gap="large")

with col_img:
    st.markdown("""
    <div class="question-card">
        <h3>📸 Traffic Sign Image</h3>
    </div>
    """, unsafe_allow_html=True)
    
    img_path = f"{TRAIN_DIR}/train_images/{current_q['image_id']}.jpg"
    
    try:
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption=f"Image ID: {current_q['image_id']}", use_container_width=True)
        else:
            st.error(f"Image not found: {img_path}")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

with col_content:
    st.markdown("""
    <div class="question-card">
        <h3>❓ Question & Answer</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="question-text">
        {current_q['question']}
    </div>
    """, unsafe_allow_html=True)
    
    q_type = current_q.get("question_type", "Multiple choice")
    if q_type.lower() == "multiple choice":
        st.markdown("##### 🧩 **Options**")
        render_multiple_choice(current_q)
    else:
        st.markdown("##### ✅ **Yes/No:**")
        render_yes_no(current_q)
    
# References with fancy styling
# References with expandable law content
st.markdown("---")
st.markdown("##### 📚 **Related Legal Documents**")
related = current_q.get("relevant_articles", [])

if related:
    for article in related:
        if isinstance(article, dict) and "law_id" in article and "article_id" in article:
            label = f"{article['law_id']} - Article {article['article_id']}"
            with st.expander(label):
                content = get_article_text(article["law_id"], article["article_id"])
                st.markdown(content)
else:
    st.markdown("*No related articles*")

# ==== Interactive Statistics with Plotly ====
st.markdown("---")
col_stats1, col_stats2, col_stats3 = st.columns(3)

mc_count = sum(1 for q in data if q.get("question_type", "").lower() == "multiple choice")
yes_no_count = len(data) - mc_count

with col_stats1:
    st.markdown(f"""
    <div class="stats-card">
        <p1>{len(data)}</p1>
        <p><strong>Total Questions</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col_stats2:
    st.markdown(f"""
    <div class="stats-card">
        <p1>{mc_count}</p1>
        <p><strong>Multiple Choice</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col_stats3:
    st.markdown(f"""
    <div class="stats-card">
        <p1>{yes_no_count}</p1>
        <p><strong>Yes/No</strong></p>
    </div>
    """, unsafe_allow_html=True)

# ==== Enhanced Sidebar Tools ====
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h2>🔧 Smart Tools</h2>
    </div>
    """, unsafe_allow_html=True)

    # Search by image ID with autocomplete
    st.markdown("### 🔍 **Search by Image ID**")
    all_image_ids = [q.get('image_id', '') for q in data]
    search_id = st.selectbox("Select or type Image ID:", [""] + sorted(set(all_image_ids)))
    
    if search_id:
        matches = [i for i, q in enumerate(data) if search_id == q.get('image_id', '')]
        if matches:
            st.success(f"✅ Found {len(matches)} match(es)")
            for idx in matches:
                if st.button(f"🎯 Go to Q{idx + 1}", key=f"search_{idx}"):
                    st.session_state.current_index = idx
                    st.rerun()

    # Enhanced filter
    st.markdown("### 🔍 **Smart Filters**")
    filter_type = st.selectbox(
        "Question Type:",
        ["All", "Multiple choice", "Yes/No"]
    )
    
    # Filter by answer key for multiple choice
    if filter_type == "Multiple choice":
        answer_keys = sorted(set(q.get('answer', '') for q in data if q.get('question_type', '').lower() == 'multiple choice'))
        selected_answer = st.selectbox("Filter by Answer Key:", ["All"] + answer_keys)
        
        if selected_answer != "All":
            filtered = [
                i for i, q in enumerate(data) 
                if q.get("question_type", "").lower() == "multiple choice" and q.get('answer', '') == selected_answer
            ]
        else:
            filtered = [
                i for i, q in enumerate(data) 
                if q.get("question_type", "").lower() == "multiple choice"
            ]
    elif filter_type == "Yes/No":
        filtered = [
            i for i, q in enumerate(data) 
            if q.get("question_type", "").lower() != "multiple choice"
        ]
    else:
        filtered = list(range(len(data)))
    
    if filter_type != "All":
        st.info(f"📋 {len(filtered)} question(s) match your criteria")
        
        if filtered:
            # Show random sample button
            if st.button("🎲 Random Sample"):
                import random
                st.session_state.current_index = random.choice(filtered)
                st.rerun()
            
            # Show first few filtered results
            st.markdown("**Quick Access:**")
            for idx in filtered[:8]:
                col_btn, col_info = st.columns([1, 2])
                with col_btn:
                    if st.button(f"Q{idx + 1}", key=f"filter_{idx}"):
                        st.session_state.current_index = idx
                        st.rerun()
                with col_info:
                    st.markdown(f"<small>{data[idx]['image_id']}</small>", unsafe_allow_html=True)

    # Quick navigation
    st.markdown("### ⚡ **Quick Navigation**")
    col_first, col_last = st.columns(2)
    
    with col_first:
        if st.button("⏮️ First"):
            st.session_state.current_index = 0
            st.rerun()
    
    with col_last:
        if st.button("⏭️ Last"):
            st.session_state.current_index = len(data) - 1
            st.rerun()