from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os


def create_vectorstore(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    return vectorstore


def save_vectorstore(vectorstore, folder_path):
    vectorstore.save_local(folder_path)


def load_vectorstore(folder_path, openai_api_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

    return FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True
    )