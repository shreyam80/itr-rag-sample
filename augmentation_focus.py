from datasets import load_dataset

# Load only a subset to keep things fast
dataset = load_dataset("ZorraZabb/full_coding_sampling_xml_fitered", split="train[:5000]")

cleaned_docs = []

for row in dataset:
    if row["lang"] and row["len_tokens"] < 1000:  # avoid massive files
        cleaned_docs.append({
            "text": row["text"],
            "metadata": {
                "language": row["lang"],
                "created_date": row["created_date"],
                "updated_date": row["updated_date"],
                "repo": row["repo_name"],
                "repo_full_name": row["repo_full_name"],
                "stars": row["star"],
                "length_tokens": row["len_tokens"],
                "dir": row["dir"],
                "source": "HuggingFace_ZorraZabb"
            }
        })

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Set up embedding model
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Chunk text if needed (for long code), or keep as-is
texts = [doc["text"] for doc in cleaned_docs]
metadatas = [doc["metadata"] for doc in cleaned_docs]

# Create ChromaDB
db = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embedding_function,
    persist_directory="./zorra_chroma"
)

db.persist()

retriever = db.as_retriever(search_kwargs={"k": 3})
query = "How do I mutate strings and test them against a regular expression in JavaScript?"

results = retriever.get_relevant_documents(query)

for doc in results:
    print(f"--- Source: {doc.metadata['repo']} | Lang: {doc.metadata['language']}")
    print(doc.page_content[:300])
    print()