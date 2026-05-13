import streamlit as st
from groq import Client

st.set_page_config(page_title="Free AI Chatbot", page_icon="💬")
st.title("💬 Free AI Chatbot (Groq)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    api_key = st.sidebar.text_input("Groq API Key (get free at groq.com)", type="password", help="Get free API key at https://console.groq.com/keys")
    if api_key:
        st.session_state.client = Client(api_key=api_key)
    else:
        st.warning("Enter your Groq API key in the sidebar to start chatting.")
        st.info("Get a free key at: https://console.groq.com/keys")
        st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
                temperature=0.7,
                max_tokens=1024
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

st.sidebar.button("Clear Chat", on_click=lambda: st.session_state.clear())