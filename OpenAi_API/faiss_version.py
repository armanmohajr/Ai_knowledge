from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import faiss
import os

load_dotenv()
client = OpenAI()

documents = [
    "FAISS is a library for fast similarity search.",
    "Embeddings convert text into numerical vectors.",
    "RAG stands for Retrieval Augmented Generation.",
    "OpenAI provides embedding and chat models."
]

embeddings = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

query = input("Ask a question: ")

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

query_embedding = np.array([query_embedding]).astype("float32")

k = 2  # number of similar docs
distances, indices = index.search(query_embedding, k)

retrieved_docs = [documents[i] for i in indices[0]]
context = "\n".join(retrieved_docs)

final_prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": final_prompt}
    ],
    temperature=0.3
)

print("\nAnswer:\n", response.choices[0].message.content)