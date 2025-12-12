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
                    result = response.json()
                    st.success("✅ Answer:")
                    st.write(result["answer"])
                    # Source caption removed - no longer displayed
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The backend might be waking up (Render free tier). Please try again.")
            except Exception as e:
                st.error(f"❌ Failed to connect: {str(e)}")
    else:
        st.warning("Please enter a question")
