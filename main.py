import streamlit as st
from dotenv import load_dotenv
import os
import hashlib

from pdf_utils import load_pdf
from vectorstore_utils import create_vectorstore, save_vectorstore, load_vectorstore
from qa_chain_utils import build_qa_chain
from langchain.text_splitter import CharacterTextSplitter


# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("📄 AI Knowledge Assistant")
st.caption("Upload documents and ask intelligent questions.")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    # Combine all PDFs
    all_text = ""
    file_hash_string = ""

    for file in uploaded_files:
        text = load_pdf(file)
        all_text += text + "\n"
        file_hash_string += file.name

    # Create unique folder for embeddings
    file_hash = hashlib.md5(file_hash_string.encode()).hexdigest()
    vectorstore_folder = f"vectorstore_{file_hash}"

    # Split text
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    text_chunks = splitter.split_text(all_text)

    # Load or Create embeddings
    if os.path.exists(vectorstore_folder):
        vectorstore = load_vectorstore(
            vectorstore_folder,
            OPENAI_API_KEY
        )
        st.success("Loaded cached embeddings ✅")
    else:
        vectorstore = create_vectorstore(
            text_chunks,
            OPENAI_API_KEY
        )
        save_vectorstore(vectorstore, vectorstore_folder)
        st.success("Created new embeddings ✅")

    # Build QA chain
    qa_chain = build_qa_chain(
        vectorstore,
        OPENAI_API_KEY
    )

    # Conversation memory
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    def submit_question():
        user_question = st.session_state.user_question

        if user_question:
            result = qa_chain.invoke({"query": user_question})
            answer = result["result"]
            sources = result["source_documents"]

            st.session_state.conversation.append(
                {"role": "user", "text": user_question}
            )

            st.session_state.conversation.append(
                {"role": "bot", "text": answer}
            )

            st.session_state.user_question = ""

    st.text_input(
        "Ask a question:",
        key="user_question",
        on_change=submit_question
    )

    # Display chat
    for chat in st.session_state.conversation:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['text']}")
        else:
            st.markdown(f"**Bot:** {chat['text']}")