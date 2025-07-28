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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Mistral model
mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
reranker = CrossEncoder(
    "microsoft/codebert-base",  # this goes as a positional arg
    num_labels=1,
    max_length=512,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)
model = AutoModelForCausalLM.from_pretrained(mistral_model_id, torch_dtype=torch.float16, device_map="auto")


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
    prompt = f"""<s>[INST] You are a helpful assistant. Summarize the purpose of the following Python code in 2–3 sentences:

```python
{code_snippet[:3000]}
``` [/INST]"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt from the output
        return response.split("```")[-1].strip()
    except Exception as e:
        print(f"[✗] Mistral summary failed: {e}")
        return "Summary unavailable due to Mistral error."
    
# ------- hashing code -----
def hash_code(code_snippet):
    return hashlib.sha256(code_snippet.encode("utf-8")).hexdigest()

def generate_final_answer(prompt):
    formatted_prompt = f"<s>[INST] {prompt.strip()} [/INST]"
    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()
    except Exception as e:
        print(f"Final answer generation failed: {e}")
        return "Final answer unavailable due to LLM error."
    
def generate_followup_query_multi(user_query, top_docs, max_docs=3):
    context_blocks = []
    for i, doc in enumerate(top_docs[:max_docs]):
        context_blocks.append(f"""
Document {i+1}:
Summary:
{doc['summary']}

Code:
{doc['code'][:800]}  # Truncate to avoid token overflow
""")

    combined_context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant evaluating retrieved documents.

The user originally asked:
"{user_query}"

You were given the following top-ranked documents:

{combined_context}

Does this set of documents fully answer the user's question?

If not, suggest a more specific follow-up query that could help retrieve a better answer.

Respond with:
- "No follow-up needed" if the documents fully answer the question
- A short follow-up query if more information is needed
"""

    return generate_final_answer(prompt).strip()


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

# ---------- Print Matches + Rerank ----------
print("\nTop Matching Summaries (Pre-Rerank):\n")

# Show initial matches
for i, row in enumerate(results):
    print(f"{i+1}. Original index {row['original_index']} - Repo: {row['repo']} ({row['stars']} stars)")
    print("Summary:", row["summary"][:400], "...")
    print("DB ID:", row["id"], "\n")

# ----- Combine query and summary+code for reranking -----
rerank_inputs = [(query, f"{row['summary']}\n{row['code']}") for row in results]
rerank_scores = reranker.predict(rerank_inputs)

# ----- Sort results by rerank score -----
reranked = sorted(zip(results, rerank_scores), key=lambda x: x[1], reverse=True)
top_reranked_results = [r[0] for r in reranked[:3]]  # Keep top 3 after rerank

# ---------- Generate Follow-Up Query ----------
followup_query = generate_followup_query_multi(
    user_query=query.replace("query: ", ""),
    top_docs=top_reranked_results
)

print("\n[🤖] Follow-up query suggestion:")
print(followup_query)

# ---------- If follow-up is needed, do second round of retrieval ----------
if followup_query.lower() != "no follow-up needed":
    followup_vector = embedder.encode([f"query: {followup_query}"])[0].tolist()

    try:
        cursor.execute(f"""
            SELECT original_index, id, summary, code, repo, stars
            FROM code_docs
            ORDER BY embedding <-> {format_vector_for_sql(followup_vector)}::vector
            LIMIT 3
        """)
        followup_results = cursor.fetchall()
    except Exception as e:
        print(f"[✗] Follow-up query failed: {e}")
        followup_results = []

    # ---------- Combine original + follow-up results and rerank again ----------
    combined = top_reranked_results + followup_results
    rerank_inputs = [(query, f"{row['summary']}\n{row['code']}") for row in combined]
    rerank_scores = reranker.predict(rerank_inputs)
    reranked = sorted(zip(combined, rerank_scores), key=lambda x: x[1], reverse=True)
    top_reranked_results = [r[0] for r in reranked[:3]]

# ---------- Prepare context blocks (now with code) ----------
context_blocks = []
for i, row in enumerate(top_reranked_results):
    context_blocks.append(f"Code:\n{row['code']}\n\nSummary: {row['summary']}\nDB ID: {row['id']}")


# ---------- Final Answer Prompt ----------
context = "\n\n---\n\n".join(context_blocks)

final_prompt = f"""
You are a skilled developer.

You have access to the following code snippets and summaries retrieved from an internal database.
The snippets are listed in order of estimated relevance to the user’s question (most relevant first).

Each block contains the code and a brief summary.

{context}

Using this information, answer the following question:
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
        final_answer = generate_final_answer(final_prompt)
        print(final_answer)

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