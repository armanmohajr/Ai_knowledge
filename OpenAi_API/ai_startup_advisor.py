from openai import OpenAI
from dotenv import load_dotenv
import json
import numpy as np
import os

load_dotenv()
client = OpenAI()

# -----------------------------
# Persistent Memory
# -----------------------------
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"name": "Unknown", "interest": "AI"}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

user_memory = load_memory()

# -----------------------------
# Knowledge Base (RAG Source)
# -----------------------------
documents = [
    "SaaS startups have high margins but require strong marketing.",
    "AI startups require strong data and clear niche focus.",
    "B2B startups usually have longer sales cycles but higher contract value.",
    "Subscription models create predictable revenue."
]

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Precompute document embeddings
doc_embeddings = [get_embedding(doc) for doc in documents]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_context(query):
    query_embedding = get_embedding(query)
    scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    best_doc = documents[np.argmax(scores)]
    return best_doc

# -----------------------------
# LLM Call (Structured JSON)
# -----------------------------
def analyze_startup(user_input, session_history):
    context = retrieve_context(user_input)

    messages = [
        {
            "role": "system",
            "content": f"""
You are an AI startup advisor.
User name: {user_memory['name']}
User interest: {user_memory['interest']}

Use the provided context to analyze the startup.
Respond ONLY in valid JSON with:
- idea_summary
- market_type
- risks
- revenue_model
"""
        },
        {
            "role": "user",
            "content": f"""
Context: {context}

Startup Idea: {user_input}
"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.5,
        max_tokens=400
    )

    return json.loads(response.choices[0].message.content)

# -----------------------------
# Session Chat Loop
# -----------------------------
session_history = []

print("🚀 AI Startup Advisor")
print("Type 'exit' to quit\n")

while True:
    user_input = input("Your startup idea: ")

    if user_input.lower() == "exit":
        break

    if user_memory["name"] == "Unknown":
        user_memory["name"] = input("Before we start, what is your name? ")
        save_memory(user_memory)

    result = analyze_startup(user_input, session_history)

    print("\n📊 Analysis:")
    print(json.dumps(result, indent=2))
    print("\n")