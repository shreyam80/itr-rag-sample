# ---------- Import Dependencies ----------
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from datetime import datetime
import psycopg2
import psycopg2.extras
import traceback
import os
import time
import uuid

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
dataset = load_dataset("ZorraZabb/full_coding_sampling_xml_fitered", split="train[0:15]")

# ---------- Load Embedding Model ----------
embedder = SentenceTransformer("intfloat/e5-base-v2")

# ---------- Load Model Clients ----------
# Summary model: text-generation
summary_client = InferenceClient("bigcode/starcoder")

# Answer model: chat interface
answer_client = InferenceClient("meta-llama/Llama-2-7b-chat-hf")


# ---------- LLM Helpers ----------
def generate_summary(code):
    prompt = f"Summarize the purpose of this codebase in 2-3 sentences:\n\n{code[:2048]}"
    try:
        response = summary_client.text_generation(prompt=prompt, max_new_tokens=150)
        return response.strip()
    except Exception as e:
        print("LLM summary failed:")
        traceback.print_exc()
        return "Summary unavailable due to LLM error."


def generate_final_answer(prompt):
    try:
        response = answer_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt.strip()}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("LLM final answer generation failed:")
        traceback.print_exc()
        return "Final answer unavailable due to LLM error."



# ---------- Summarize, Embed, Upload ----------
for i, row in enumerate(dataset):
    code = row["text"]
    lang = row["lang"]
    repo = row["repo_name"]
    stars = row["star"]
    created = str(row["created_date"])
    updated = str(row["updated_date"])

    # ---------- Summarize ----------
    try:
        summary = generate_summary(code)
    except Exception as e:
        summary = "Summary unavailable due to LLM error."
        print(f"LLM summary failed for doc {i}: {e}")

    # ---------- Embed ----------
    try:
        vector = embedder.encode([f"passage: {summary}"])[0].tolist()
    except Exception as e:
        print(f"Embedding failed for doc {i}: {e}")
        continue

    # ---------- Insert into PostgreSQL ----------
    doc_id = str(uuid.uuid4())
    try:
        insert_query = """
            INSERT INTO code_docs (
                id, lang, repo, stars, created_at, updated_at,
                summary, code, embedding, original_index
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            doc_id, lang, repo, stars, created, updated,
            summary, code, vector, i
        ))
        conn.commit()
        print(f"[✓] Inserted doc {i} — Repo: {repo}")
    except Exception as e:
        print(f"[✗] PostgreSQL insert failed for doc {i}: {e}")
        continue

    time.sleep(0.5)  # Optional throttle

# ---------- Query ----------
query = "query: Write a function in Objective-C to add a user to a chat room"
query_vector = embedder.encode([query])[0].tolist()

# ---------- Search PostgreSQL using Cosine Similarity ----------
try:
    search_query = f"""
        SELECT original_index, id, summary, code, repo, stars
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
    context_blocks.append(f"Summary: {row['summary']}\nCode:\n```{row['code']}```\nDB ID: {row['id']}")

# ---------- Final Answer Prompt ----------
context = "\n\n---\n\n".join(context_blocks)

final_prompt = f"""
You are a skilled developer.

You have access to the following code summaries, code documents, and internal DB IDs:

{context}

Using this information, answer this question:
{query.replace('query: ', '')}
"""

# ---------- Generate Final Answer ----------
final_answer = generate_final_answer(final_prompt)
print("\nFinal Answer:\n")
print(final_answer)