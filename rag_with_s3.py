# ---------- Step 1: Install and Import Dependencies ----------
from datasets import load_dataset
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
import boto3
import os
import time
from datetime import datetime
import uuid

# ---------- Step 2: Load Environment & Clients ----------
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
S3_BUCKET = os.getenv("S3_BUCKET_NAME") # Can we put the code in a column in a table? 
                                        # Create a blob row in the table similar to S3 blob storage

# ---------- Step 3: Load Dataset ----------
dataset = load_dataset("ZorraZabb/full_coding_sampling_xml_fitered", split="train[0:15]")

# ---------- Step 4: Summarize Full Documents + Upload to S3 ----------
summarized_docs = []
doc_embeddings = []
metadatas = []

embedder = SentenceTransformer("intfloat/e5-base-v2")

print("\nSummarizing and uploading to S3...\n")

for i, row in enumerate(dataset):
    code = row["text"]
    lang = row["lang"]
    repo = row["repo_name"]
    stars = row["star"]
    created = str(row["created_date"])
    updated = str(row["updated_date"])

    # Summarize entire document
    try:
        summary_prompt = f"Summarize the overall purpose of this codebase:\n\n{code[:3000]}"
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful programming assistant."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=200
        )
        summary = response.choices[0].message.content.strip()
        print(f"Summarized doc {i}")
    except Exception as e:
        summary = "Summary unavailable due to LLM error."
        print(f"LLM summary failed for doc {i}: {e}")

    # Upload full document to S3
    s3_key = f"code_docs/{uuid.uuid4()}.txt"
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=code.encode("utf-8")
        )
        s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
    except Exception as e:
        s3_url = "Upload failed"
        print(f"S3 upload failed for doc {i}: {e}")

    # Embed summary only
    vector = embedder.encode([f"passage: {summary}"])[0]
    doc_embeddings.append(vector.tolist())
    summarized_docs.append(summary)

    # Metadata for retrieval
    metadatas.append({
        "original_index": i,
        "lang": lang,
        "repo": repo,
        "stars": stars,
        "created": created,
        "updated": updated,
        "s3_url": s3_url
    })

    time.sleep(0.5)

# ---------- Step 5: Store in ChromaDB ----------
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="code_summary_s3")

collection.add(
    documents=summarized_docs,
    embeddings=doc_embeddings,
    ids=[f"doc_{i}" for i in range(len(summarized_docs))],
    metadatas=metadatas
)

print("\nStored summaries and S3 links in ChromaDB.\n")

# ---------- Step 6: Query ----------
query = "query: Write a function in Objective-C to add a user to a chat room"
query_vector = embedder.encode([query])[0]

results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=5
)

print("\nTop Matching Summaries:\n")
for i, doc in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]
    print(f"{i+1}. Doc {metadata['original_index']} — Repo: {metadata['repo']} ({metadata['stars']} stars)")
    print("Summary:", doc[:400], "...")
    print("S3 URL:", metadata["s3_url"], "\n")

# ---------- Step 7: Generate Final Answer ----------
context = "\n\n---\n\n".join([
    f"Summary: {doc}\nS3 Link: {metadata['s3_url']}"
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0])
])

final_prompt = f"""
You are a skilled developer.

You have access to the following code summaries and storage locations (S3 URLs):

{context}

Using this information, answer this question:
{query.replace('query: ', '')}
"""

try:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a highly skilled software engineer."},
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=500
    )
    answer = response.choices[0].message.content.strip()
    print("\nFinal Answer:\n")
    print(answer)
except Exception as e:
    print(f"LLM final answer failed: {e}")
