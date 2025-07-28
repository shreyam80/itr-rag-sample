# ---------- Step 1: Install and Import Dependencies ----------
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb
from datetime import datetime
from dotenv import load_dotenv
import os
import time

# ---------- Step 2: Load Environment & Clients ----------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Step 3: Load Dataset ----------
dataset = load_dataset("ZorraZabb/full_coding_sampling_xml_fitered", split="train[0:2]")

# ---------- Step 4: Extract, Chunk, and Summarize Code ----------
code_chunks = []
chunk_metadatas = []
summarized_docs = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

print("\nChunking and summarizing...\n")

for i, row in enumerate(dataset):
    code = row["text"]
    lang = row["lang"]
    repo = row["repo_name"]
    stars = row["star"]
    created = str(row["created_date"])
    updated = str(row["updated_date"])

    # Hybrid metadata for embedding
    hybrid_prefix = f"# Language: {lang}\n# Repo: {repo}\n# Stars: {stars}\n"

    # Chunk the code
    chunks = splitter.split_text(code)

    for j, chunk in enumerate(chunks):
        chunk_id = f"{i}_{j}"
        full_chunk = f"{hybrid_prefix}{chunk}"

        # LLM summary
        try:
            summary_prompt = f"Summarize the purpose of this code snippet:\n\n{chunk}"
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful programming assistant."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=100
            )
            summary = response.choices[0].message.content.strip()
            print(f"Summarized chunk {chunk_id}")
        except Exception as e:
            summary = "Summary unavailable due to LLM error."
            print(f"LLM summary failed for chunk {chunk_id}: {e}")

        # Store combined document
        doc_text = f"[SUMMARY]\n{summary}\n\n[CODE]\n{chunk}"
        summarized_docs.append(doc_text)

        # Metadata
        chunk_metadatas.append({
            "chunk_id": chunk_id,
            "original_index": i,
            "lang": lang,
            "repo": repo,
            "stars": stars,
            "created": created,
            "updated": updated
        })

        # Hybrid text for embedding
        code_chunks.append(f"passage: {hybrid_prefix}{summary}")

        time.sleep(0.5)  # Respect OpenAI rate limit

# ---------- Step 5: Embed Hybrid Chunks ----------
embedder = SentenceTransformer("intfloat/e5-base-v2")
chunk_vectors = embedder.encode(code_chunks, convert_to_numpy=True)

# ---------- Step 6: Store Embeddings in ChromaDB ----------
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="code_search_llm")

collection.add(
    documents=summarized_docs,
    embeddings=chunk_vectors.tolist(),
    ids=[f"chunk_{i}" for i in range(len(summarized_docs))],
    metadatas=chunk_metadatas
)

print("\nStored summary + code in ChromaDB.\n")

# ---------- Step 7: Query ChromaDB ----------
query = "query: Write a function in Objective-C to add a user to a chat room"
query_vector = embedder.encode([query])[0]

results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=5
)

print("\nTop Matching Summaries + Code:\n")
retrieved_docs = []
for i, doc in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]
    print(f"{i+1}. Chunk {metadata['chunk_id']} — From Sample {metadata['original_index']}, Repo: {metadata['repo']}, Stars: {metadata['stars']}")
    print(doc[:600], "...\n")
    retrieved_docs.append(doc)

# ---------- Step 8: Use LLM to Generate Final Answer ----------
print("Generating final answer using GPT-4...\n")

# Join all retrieved summaries+code into a context
context_text = "\n\n---\n\n".join(retrieved_docs)

# Build the augmentation prompt
final_prompt = f"""
You are a software engineer helping another developer.

Here are 5 retrieved code snippets and their summaries:

{context_text}

Using the above, answer this question:
{query.replace('query: ', '')}
"""

# Call GPT-4 to generate the final answer
try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a highly skilled software engineer."},
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=500
    )
    answer = response.choices[0].message.content.strip()
    print("Final Answer:\n")
    print(answer)

except Exception as e:
    print(f"Failed to generate answer: {e}")
