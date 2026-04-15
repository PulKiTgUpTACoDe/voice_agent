import streamlit as st
import requests
import time
import os

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Voice AI Agent", page_icon="🎤", layout="centered")

st.title("🎤 Local Voice-Controlled AI Agent")
st.write("Record audio, upload a file, or type text to interact with the local AI agent.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio", "⌨️ Type Text"])

# State variables
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

def process_audio_file(audio_bytes, filename="audio.wav"):
    with st.spinner("Processing audio and running agent pipeline..."):
        try:
            files = {"file": (filename, audio_bytes, "audio/wav")}
            res = requests.post(f"{API_URL}/agent-run", files=files)
            if res.status_code == 200:
                st.session_state.agent_result = res.json()
            else:
                st.error(f"Error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

def process_text(text):
    with st.spinner("Running agent pipeline..."):
        try:
            res = requests.post(f"{API_URL}/agent-run", params={"text": text})
            if res.status_code == 200:
                st.session_state.agent_result = res.json()
            else:
                st.error(f"Error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

with tab1:
    audio_val = st.audio_input("Record a voice command")
    if audio_val:
        st.audio(audio_val)
        if st.button("Process Recording"):
            process_audio_file(audio_val.getvalue(), "record.wav")

with tab2:
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Process Uploaded File"):
            process_audio_file(uploaded_file.getvalue(), uploaded_file.name)

with tab3:
    user_text = st.text_input("Enter your command manually:")
    if st.button("Send Text"):
        if user_text:
            process_text(user_text)
        else:
            st.warning("Please enter some text.")

st.markdown("---")

# Display Results
if st.session_state.agent_result:
    res = st.session_state.agent_result
    st.subheader("Results")
    
    # 1. Transcription (if any)
    if res.get("stt_text"):
        st.info(f"**Transcription:** {res['stt_text']}")
    
    # 2. Intent Details
    intent_data = res.get("intent", {})
    intent_name = intent_data.get("intent", "UNKNOWN")
    with st.expander(f"🧠 Detected Intent: {intent_name}", expanded=False):
        st.json(intent_data)
        
    # 3. Final Output
    st.success(f"**Agent Output:**\n{res.get('output')}")
    
    if st.button("Clear Output"):
        st.session_state.agent_result = None
        st.rerun()

