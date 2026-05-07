import streamlit as st
from groq import Groq
import os
import uuid
import json

# PDF + RAG
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp { background-color: #343541; color: white; }
.chat-container { max-width: 800px; margin: auto; }
.user-msg {
    background-color: #3b3c47;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: right;
}
.bot-msg {
    background-color: #444654;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- API ----------------

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
    
# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------- CHAT STORAGE ----------------
def save_chat():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chats, f)

def load_chat():
    try:
        with open("chat_history.json") as f:
            return json.load(f)
    except:
        return {}

# ---------------- INIT ----------------
if "chats" not in st.session_state:
    st.session_state.chats = load_chat()

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🤖 AI Agent")

    # New Chat
    if st.button("➕ New Chat"):
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat = chat_id
        st.session_state.chats[chat_id] = []
        st.rerun()

    st.markdown("---")

    # Chat list
    for cid in st.session_state.chats:
        if st.button(f"Chat {list(st.session_state.chats).index(cid)+1}"):
            st.session_state.current_chat = cid
            st.rerun()

    st.markdown("---")

    # PDF Upload
    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

# ---------------- PDF PROCESSING ----------------
def process_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = embedder.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return chunks, index

# Store PDF
if uploaded_file:
    chunks, index = process_pdf(uploaded_file)
    st.session_state.pdf_chunks = chunks
    st.session_state.pdf_index = index
    st.sidebar.success("PDF Loaded ✅")

# Search PDF
def search_pdf(query):
    query_vec = embedder.encode([query])
    D, I = st.session_state.pdf_index.search(np.array(query_vec), k=3)
    return " ".join([st.session_state.pdf_chunks[i] for i in I[0]])

# ---------------- MAIN UI ----------------
st.markdown("<h2 style='text-align:center;'>🤖 AI Agent Chatbot</h2>", unsafe_allow_html=True)

messages = st.session_state.chats[st.session_state.current_chat]

# Show messages
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for i, msg in enumerate(messages):
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

        if st.button("📋 Copy", key=f"copy_{i}"):
            st.toast("Copied!")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.chat_input("Type your message...")

if user_input:
    messages.append({"role": "user", "content": user_input})

    context = ""

    # Use PDF if available
    if "pdf_index" in st.session_state:
        context = search_pdf(user_input)

    # AI Response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer clearly. Use context if provided."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
        ]
    )

    bot_reply = response.choices[0].message.content

    # Fallback (agent-like behavior)
    if len(bot_reply) < 20:
        bot_reply = "I couldn't find exact info, but here's a general answer:\n\n" + bot_reply

    # Typing animation
    placeholder = st.empty()
    full_text = ""

    for word in bot_reply.split():
        full_text += word + " "
        placeholder.markdown(f"<div class='bot-msg'>🤖 {full_text}▌</div>", unsafe_allow_html=True)

    placeholder.markdown(f"<div class='bot-msg'>🤖 {full_text}</div>", unsafe_allow_html=True)

    messages.append({"role": "assistant", "content": full_text})

    save_chat()

# ---------------- REGENERATE ----------------
if messages and messages[-1]["role"] == "assistant":
    if st.button("🔁 Regenerate Response"):
        messages.pop()

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )

        new_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": new_reply})

        save_chat()
        st.rerun()
