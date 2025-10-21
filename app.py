import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Page config
st.set_page_config(
    page_title="WeldersKit AI",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Styling
st.markdown("""
<style>
    .main { max-width: 600px; margin: auto; }
    .stChatMessage { font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("🔧 WeldersKit AI")
st.markdown("Ask me anything about welding, materials, techniques, or prices in Nigeria")

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/api/ask"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if user_input := st.chat_input("Ask about welding..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get response from backend
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    BACKEND_URL,
                    json={"question": user_input},
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No response found.")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"❌ Error {response.status_code}: {response.text}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        except requests.exceptions.ConnectionError:
            error_msg = "❌ Can't connect to backend. Is it running on http://127.0.0.1:8000?"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        except requests.exceptions.Timeout:
            error_msg = "❌ Request timeout. Backend took too long to respond."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})