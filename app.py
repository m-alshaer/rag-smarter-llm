from pdf import extract_chunks_from_pdf
from video import extract_chunks_from_youtube, get_youtube_title
from model import get_model
import streamlit as st
import tempfile
import sys
sys.modules['torch.classes'] = None

# Title
st.set_page_config(page_title="PDF & Video Chatbot", layout="wide")
# Main Tabs
tab_pdf, tab_youtube = st.tabs(["PDF Document", "YouTube Video"])

# Initialize session states
if 'video_chat' not in st.session_state:
    st.session_state.video_chat = []
if 'pdf_chat' not in st.session_state:
    st.session_state.pdf_chat = []

# PDF Tab
with tab_pdf:
    st.header("PDF Document Processor")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_upload")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            pdf_chunks = extract_chunks_from_pdf(tmp_path)
            st.subheader("PDF Chunks Sample")
            st.write(pdf_chunks[:5])
            model = get_model(pdf_chunks)

            st.divider()
            st.subheader("Chat about Document")
            user_input = st.text_input("Ask about the document:", key="doc_q")

            if st.button("Ask about Document", key="doc_ask"):
                with st.spinner("Analyzing..."):
                    response = model.invoke({"question": user_input})
                st.session_state.pdf_chat.append((user_input, response))

            for question, answer in st.session_state.pdf_chat:
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**A:** {answer}")
                st.divider()

# YouTube Tab
with tab_youtube:
    st.header("YouTube Video Processor")
    youtube_url = st.text_input("Enter YouTube URL:", key="youtube_url")

    if youtube_url:
        with st.spinner("Processing Video..."):
            video_chunks = extract_chunks_from_youtube(youtube_url)
            st.subheader("Video Chunks Sample")
            st.write(video_chunks[:5])
            model = get_model(video_chunks)
            if len(video_chunks) > 0:
                st.divider()
                st.subheader(f"Chat about \"{get_youtube_title(youtube_url)}\"")

                user_input = st.text_input("Ask about the video:", key="video_q")

                if st.button("Ask about Video", key="video_ask"):
                    with st.spinner("Thinking..."):
                        response = model.invoke({"question": user_input})
                    st.session_state.video_chat.append((user_input, response))

                for question, answer in st.session_state.video_chat:
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
                    st.divider()


# Sidebar
with st.sidebar:
    st.header("Settings & Info")
    st.markdown("""
    **Instructions:**
    1. For PDF: Upload any text-based PDF
    2. For YouTube: Enter a valid URL
    3. Ask questions about the content
    """)
    st.divider()
    st.caption("Note: YouTube processing requires additional setup for actual transcript extraction")