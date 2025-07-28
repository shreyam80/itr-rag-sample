# Step 1: Load code dataset
from datasets import load_dataset

# Load a small sample of the Hugging Face code dataset
dataset = load_dataset("ZorraZabb/full_coding_sampling_xml_fitered",split="train[0:15]")

# Print the fields available in the dataset
print("Available fields:", dataset.column_names)

# Print the first sample
#print("\nFirst sample:")
#print(dataset[0])

# --------- Encoding ---------

from sentence_transformers import SentenceTransformer

# Step 2: Load CodeBERT model
code_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Extract the raw code
code_samples = [sample["text"] for sample in dataset]

#print("\n📄 Sample code snippets embedded:\n")
##for i, sample in enumerate(code_samples):
    #print(f"--- Code {i+1} ---")
    #print(sample[:300])  # print first 300 characters to keep it short
    #print()

    
# Step 4: Convert code into embeddings (vectors)
code_vectors = code_embedder.encode(code_samples)

print("\n🧪 Scanning for code with addition logic...")

'''
for i, code in enumerate(code_samples):
        print(f"\n--- Code Sample {i} ---")
        print(code)  # Show first 500 characters
'''

# Step 5: Print the first vector and its shape
#print("First vector:", code_vectors[0][:10])  # Just show first 10 numbers
#print("Vector shape:", len(code_vectors[0]))

# --------------- Chroma upload -------------------

# --------- Step 6: Import ChromaDB and setup the DB ---------
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB in-memory client (you can also persist to disk)
chroma_client = chromadb.Client()

# Create a collection to hold your code vectors
collection = chroma_client.create_collection(name="code_snippets")

# --------- Step 7: Add embeddings to the collection ---------

# Add entries with IDs, documents (optional), and embeddings
collection.add(
    documents=code_samples,                      # raw code strings
    embeddings=code_vectors.tolist(),            # convert numpy to list of lists
    ids=[f"code_{i}" for i in range(len(code_samples))],
    metadatas=[{"source_index": i} for i in range(len(code_samples))]  # unique string IDs
)

print("✅ Stored code snippets in ChromaDB.")

# --------- Step 8: Search the DB with a new query ---------

# Sample query
query = "How do I mutate strings and test them against a regular expression in JavaScript?"

# Embed the query using CodeBERT
query_vector = code_embedder.encode([query])[0]

# Query ChromaDB using the vector
results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=3  # Top 3 most similar results
)

# --------- Step 9: View Results ---------
print("\n🔍 Top matching code snippets for query:")
for i, doc in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]  # Access the metadata for this result
    print(f"{i+1}. Code Sample {metadata['source_index']}")
    print(doc)
    print()