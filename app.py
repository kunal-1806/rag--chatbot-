import os
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon=":material/search:")

from main import ask, build_from_directory, build_from_uploads

PERSIST_DIR = "faiss_store"
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss.index")
META_PATH = os.path.join(PERSIST_DIR, "metadata.pkl")
INDEX_EXISTS = os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)

# --- Sidebar ---
with st.sidebar:
    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv", "xlsx"],
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        build_uploaded = st.button(
            "Build from uploads", use_container_width=True, disabled=not uploaded_files
        )
    with col2:
        build_data = st.button("Build from data/", use_container_width=True)

    st.divider()

    if INDEX_EXISTS:
        st.success(":material/check_circle: Index ready")
    else:
        st.info(":material/hourglass_empty: No index yet. Upload files or use data/ to get started.")

# --- Main area ---
st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle build buttons
if build_uploaded and uploaded_files:
    with st.spinner("Building index from uploaded files..."):
        build_from_uploads(uploaded_files, PERSIST_DIR)
    st.rerun()

if build_data:
    with st.spinner("Building index from data/ directory..."):
        build_from_directory(data_dir="data", persist_dir=PERSIST_DIR)
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if INDEX_EXISTS:
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask(prompt)
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
