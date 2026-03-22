import streamlit as st
import requests
import os

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "https://consultation-welderskit.onrender.com")

st.title("WeldersKit Consultation")

# User input
question = st.text_area("Ask your welding question:", height=100)

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Getting answer from AI..."):
            try:
                # Call the correct endpoint: /api/ask
                response = requests.post(
                    f"{BACKEND_URL}/api/ask",
                    json={"question": question},
                    timeout=60
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        st.success("✅ Answer:")
                        st.write(result["answer"])
                    except ValueError:
                        st.error("❌ Sorry, something went wrong with the answer generation. Please try again.")
                else:
                    st.error("❌ Service temporarily busy. Please try again in a few minutes.")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Connection timed out. Please try again.")
            except Exception:
                st.error("❌ Connection failed. Please try again later.")
    else:
        st.warning("Please enter a question")
