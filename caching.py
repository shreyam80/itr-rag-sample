# ---------- Import Dependencies ----------
from datasets import load_dataset
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from datetime import datetime
import psycopg2
import psycopg2.extras
import os
import time
import uuid
import hashlib


# ---------- Format Embedding Vector for SQL ----------
def format_vector_for_sql(vec):
    return f"'[{','.join(str(x) for x in vec)}]'"

# ---------- Load Environment Variables ----------
load_dotenv()

# ---------- Connect to PostgreSQL ----------
conn = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    dbname=os.getenv("PG_DB")
)
cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# ---------- Load Dataset ----------
dataset = load_dataset("ZorraZabb/full_coding_sampling_xml_fitered", split="train[0:4]")

# ---------- Load Embedding Model ----------
embedder = SentenceTransformer("intfloat/e5-base-v2")

# ---------- Load Open Source LLM (TinyLlama or Mistral-style) ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Helper: Summarize with LLM ----------
def generate_summary(code_snippet):
    prompt = f"""
You are a helpful AI assistant. Please summarize the purpose of the following codebase in 2–3 concise sentences.

```python
{code_snippet[:3000]}
"""
    try:
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM summary failed: {e}")
        return "Summary unavailable due to LLM error."
    
# ------- hashing code -----
def hash_code(code_snippet):
    return hashlib.sha256(code_snippet.encode("utf-8")).hexdigest()

# ---------- Summarize, Embed, Upload ----------
for i, row in enumerate(dataset):
    code = row["text"]
    lang = row["lang"]
    repo = row["repo_name"]
    stars = row["star"]
    created = str(row["created_date"])
    updated = str(row["updated_date"])

    # ---- Early check: Skip if full document was processed already ----
    doc_code_hash = hash_code(code)
    cursor.execute("SELECT 1 FROM code_docs WHERE original_index = %s", (i,))
    if cursor.fetchone():
        print(f"[→] Skipping doc {i} (already chunked and stored)")
        continue

    # ---- Chunking ----
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(code)

    for j, chunk in enumerate(chunks):
        chunk_id = f"{i}_{j}"
        full_code = chunk
        # ------ check if chunk already in db ------
        chunk_hash = hash_code(full_code)

        cursor.execute("SELECT summary FROM code_docs WHERE code_hash = %s", (chunk_hash,))
        cached_row = cursor.fetchone()
        if cached_row:
            print(f"Chunk {chunk_id} already exists.")
            continue

        # ---- if chunk not in db, generate summary ------
        try:
            summary = generate_summary(full_code)

            # -------- embed and insert summary--------
            vector = embedder.encode([f"passage: {summary}"])[0].tolist()
        except Exception as e:
            print(f"[✗] Failed chunk {chunk_id}: {e}")
            continue

        doc_id = str(uuid.uuid4())
        
        try:
            cursor.execute("""
                INSERT INTO code_docs (
                    id, lang, repo, stars, created_at, updated_at,
                    summary, code, embedding, original_index, chunk_id, code_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                doc_id, lang, repo, stars, created, updated,
                summary, full_code, vector, i, chunk_id, chunk_hash
            ))
            conn.commit()
            print(f"[✓] Inserted chunk {chunk_id} — Repo: {repo}")
        except Exception as e:
            print(f"[✗] PostgreSQL insert failed for chunk {chunk_id}: {e}")
            conn.rollback()

        time.sleep(0.5)  # Optional throttle

# ---------- Query ----------
query = "query: Write a function in Objective-C to add a user to a chat room"
query_vector = embedder.encode([query])[0].tolist()


# ---------- Search PostgreSQL using Cosine Similarity ----------
try:
    search_query = f"""
        SELECT original_index, id, summary, repo, stars
        FROM code_docs
        ORDER BY embedding <-> {format_vector_for_sql(query_vector)}::vector
        LIMIT 3
    """
    cursor.execute(search_query)
    results = cursor.fetchall()
except Exception as e:
    print(f"PostgreSQL query failed: {e}")
    results = []

# ---------- Print Matches ----------
print("\nTop Matching Summaries:\n")
context_blocks = []
for i, row in enumerate(results):
    print(f"{i+1}. Original index {row['original_index']} - Repo: {row['repo']} ({row['stars']} stars)")
    print("Summary:", row["summary"][:400], "...")
    print("DB ID:", row["id"], "\n")
    context_blocks.append(f"Summary: {row['summary']}\nDB ID: {row['id']}")

# ---------- Final Answer Prompt ----------
context = "\n\n---\n\n".join(context_blocks)

final_prompt = f"""
You are a skilled developer.

You have access to the following code summaries and internal DB IDs:

{context}

Using this information, answer this question:
{query.replace('query: ', '')}
"""

# ---------- Check for Cached Answer ----------
cursor.execute("SELECT final_answer FROM query_logs WHERE query = %s", (query,))
cached_result = cursor.fetchone()

if cached_result:
    print("[✓] Cached final answer found:\n")
    print(cached_result["final_answer"])
else:
    # ---------- Generate Final Answer ----------
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt.strip()}],
            max_tokens=500,
        )
        final_answer = response.choices[0].message.content.strip()
        print(final_answer)
        print("\nFinal Answer:\n")

        # Save to query_logs table
        try:
            doc_ids = [row["id"] for row in results]
            query_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO query_logs (id, query, final_answer, retrieved_doc_ids)
                VALUES (%s, %s, %s, %s)
            """, (query_id, query, final_answer, doc_ids))
            conn.commit()
            print(f"[✓] Logged query + answer to query_logs")
        except Exception as e:
            print(f"[✗] Failed to log query + answer: {e}")

    except Exception as e:
        print(f"LLM final answer generation failed: {e}")