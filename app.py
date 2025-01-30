# app.py
import os
import shutil
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from rag import RagDocumentationGenerator
import zipfile

st.set_page_config(page_title="RAG with Local DeepSeek R1")

def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            try:
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
            except ValueError as e:
                agent_text = str(e)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_files():
    """Handle directory (zip) upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for uploaded_file in st.session_state["file_uploader"]:
        if uploaded_file.name.endswith(".zip"):
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, uploaded_file.name)

            with open(zip_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Process each extracted file
            for root, _, files in os.walk(temp_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    st.session_state["assistant"].ingest(file_path, debug=True)

            st.session_state["messages"].append(
                (f"Extracted and ingested {uploaded_file.name}", False)
            )

        else:  # If it's a regular file
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)

            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {uploaded_file.name}..."):
                t0 = time.time()
                st.session_state["assistant"].ingest(file_path, debug=True)
                t1 = time.time()

            st.session_state["messages"].append(
                (f"Ingested {uploaded_file.name} in {t1 - t0:.2f} seconds", False)
            )

def page():
    """Main app page layout."""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = RagDocumentationGenerator()

    st.header("RAGGEN met Sanjay")

    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload file(s) or a .zip directory",
        key="file_uploader",
        on_change=read_and_save_files,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    # Retrieval settings
    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    # Display messages and text input
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()


if __name__ == "__main__":
    page()
