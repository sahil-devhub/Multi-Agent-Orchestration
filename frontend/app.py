import streamlit as st
import requests
import base64
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Team AI",
    page_icon="🤖",
    layout="wide"
)

# --- App Title & Description ---
st.title("🤖 Team AI: A Multi-Agent Collaboration Platform")
st.write("Ask a question, and the AI team will automatically select the best Groq model and agent for your request.") # Updated description

# --- API URL ---
API_URL = "http://127.0.0.1:8000/agent"

# --- Sidebar ---
with st.sidebar:
    # --- THIS IS THE FIX ---
    # Removed st.header("Configuration") and all model selection UI.
    
    # Image uploader is all that remains
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])


# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask the AI team... (add an image in the sidebar)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("The AI team is thinking..."):
            try:
                image_data = None
                if uploaded_file is not None:
                    bytes_data = uploaded_file.getvalue()
                    image_data = base64.b64encode(bytes_data).decode()

                conversation_history = [m["content"] for m in st.session_state.messages]

                payload = {
                    "system_prompt": "",
                    "messages": conversation_history,
                    "allow_search": True,
                    "image_data": image_data
                }

                resp = requests.post(API_URL, json=payload, timeout=90) 

                if resp.status_code == 200:
                    full_response = resp.json().get("response", "I'm sorry, I couldn't process that request.")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_text = f"API Error {resp.status_code}: {resp.text}"
                    message_placeholder.error(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})

            except requests.exceptions.RequestException as e:
                error_text = f"Failed to connect to the API. Please ensure the backend server is running. Error: {e}"
                message_placeholder.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})