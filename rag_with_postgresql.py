# ---------- Import Dependencies ----------
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime
import psycopg2
import psycopg2.extras
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

# ---------- Load Open Source LLM (TinyLlama or Mistral-style) ----------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ---------- Helper: Summarize with LLM ----------
def generate_summary(code_snippet):
    prompt = f"### Instruction:\nSummarize the purpose of this codebase.\n\n### Code:\n{code_snippet[:3000]}\n\n### Summary:"
    response = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].split("### Summary:")[-1].strip()

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

# ---------- Generate Final Answer ----------
try:
    response = generator(final_prompt, max_new_tokens=500, do_sample=True, temperature=0.7)
    print("\nFinal Answer:\n")
    print(response[0]["generated_text"].replace(final_prompt, "").strip())
except Exception as e:
    print(f"LLM final answer generation failed: {e}")
