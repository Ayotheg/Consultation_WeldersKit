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
                        st.caption(f"Model: {result.get('model_used', 'unknown')} | Status: {response.status_code}")
                    except ValueError:
                        st.error("❌ Backend returned 200 but response was not valid JSON.")
                        st.code(response.text)
                else:
                    st.error(f"❌ Error {response.status_code}")
                    st.warning("Render backend might be failing or returning an error page.")
                    with st.expander("Show detailed error"):
                        st.write(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The backend might be taking too long to process (free tier can be slow).")
            except Exception as e:
                st.error(f"❌ Connection Failed: {str(e)}")
                st.info(f"Connecting to Backend: {BACKEND_URL}")
    else:
        st.warning("Please enter a question")
