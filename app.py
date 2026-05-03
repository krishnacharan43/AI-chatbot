import streamlit as st
from groq import Groq

# Page config
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Custom CSS (ChatGPT style)
st.markdown("""
<style>
body {
    background-color: #343541;
}
.stApp {
    background-color: #343541;
    color: white;
}
.chat-user {
    background-color: #3e3f4b;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.chat-bot {
    background-color: #444654;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🤖 AI Assistant")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Groq client
client = Groq(api_key="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")

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat display
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>🧑‍💻 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

# Input
user_input = st.chat_input("Send a message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=st.session_state.messages
    )

    bot_reply = response.choices[0].message.content

    # Typing effect
    placeholder = st.empty()
    full_text = ""

    for word in bot_reply.split():
        full_text += word + " "
        placeholder.markdown(f"<div class='chat-bot'>🤖 {full_text}▌</div>", unsafe_allow_html=True)

    placeholder.markdown(f"<div class='chat-bot'>🤖 {full_text}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": full_text})