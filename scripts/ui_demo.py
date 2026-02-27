import streamlit as st
import httpx
import time
import os
import json
from pathlib import Path

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL = 0.5 # Seconds

st.set_page_config(
    page_title="Audio Intent Pipeline Demo",
    page_icon="🎙️",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🎙️ Audio Intent Pipeline")
st.markdown("""
Upload an audio file to classify the user's intent using our **STT → LLM** pipeline. 
This demo uses **Deepgram** for speech-to-text and **Gemini 2.5 Flash** for intent reasoning.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Audio Language", ["en", "it"], index=0)
    st.info("Ensure the FastAPI server is running at http://localhost:8000")

# --- Main Interface ---
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')
    
    if st.button("🚀 Process Audio", type="primary"):
        with st.status("Processing...", expanded=True) as status:
            try:
                # 1. Post analysis job
                status.write("Uploading audio to pipeline...")
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"language": language}
                
                response = httpx.post(f"{API_BASE_URL}/v1/analyze-audio", files=files, data=data, timeout=30.0)
                response.raise_for_status()
                job_data = response.json()
                job_id = job_data["job_id"]
                
                status.write(f"Job created: `{job_id}`. Polling for results...")
                
                # 2. Poll for results
                start_time = time.time()
                while True:
                    res_response = httpx.get(f"{API_BASE_URL}/v1/result/{job_id}")
                    res_response.raise_for_status()
                    result_data = res_response.json()

                    print("result_data", result_data)
                    
                    if result_data["status"] == "done":
                        status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                        break
                    elif result_data["status"] == "failed":
                        status.update(label="❌ Processing Failed", state="error")
                        st.error(f"Reason: {result_data.get('reason', 'Unknown error')}")
                        st.stop()
                    
                    if time.time() - start_time > 60: # 60s timeout
                        status.update(label="⏰ Polling Timeout", state="error")
                        st.error("The request timed out while polling for results.")
                        st.stop()
                    
                    time.sleep(POLL_INTERVAL)

                # --- Results Display ---
                st.divider()
                
                # Top Row: Intent & Transcript
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    intent = result_data["result"]["intent"].upper().replace("_", " ")
                    confidence = result_data["result"]["confidence"]
                    
                    st.subheader("Detected Intent")
                    st.markdown(f"### {intent}")
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                
                with col2:
                    transcript = result_data["metadata"]["stt"]["transcript"]
                    st.subheader("Transcript")
                    st.info(f"\"{transcript}\"")

                # Middle Row: Action
                st.subheader("Recommended Action")
                st.success(result_data["result"]["action"])
                
                if result_data["result"].get("reasoning"):
                    with st.expander("Reasoning"):
                        st.write(result_data["result"]["reasoning"])

                # Bottom Row: Metrics
                st.divider()
                st.subheader("Pipeline Metrics")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                pipeline = result_data["metadata"]["pipeline"]
                m_col1.metric("Total Latency", f"{pipeline['total_latency_ms']:.0f}ms")
                m_col2.metric("STT Latency", f"{pipeline['stt_latency_ms']:.0f}ms")
                m_col3.metric("LLM Latency", f"{pipeline['llm_latency_ms']:.0f}ms")
                m_col4.metric("Est. Cost", f"${pipeline['estimated_cost_usd']:.5f}")

            except Exception as e:
                status.update(label="💥 Connection Error", state="error")
                st.error(f"Failed to communicate with API: {str(e)}")
                st.info("Please ensure the FastAPI backend is running with: `uv run uvicorn app.main:app` or similar")

else:
    st.info("Please upload an audio file to begin.")
