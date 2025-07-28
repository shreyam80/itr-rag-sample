# -----------------------------------------------
# Step 1–2: Load Code Snippets and Embed with CodeBERT
# -----------------------------------------------

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load dataset
print("📦 Loading dataset...")
raw_dataset = load_dataset("ZorraZabb/full_coding_sampling_xml_fitered")

# Check for split
if isinstance(raw_dataset, dict) and "train" in raw_dataset:
    dataset = raw_dataset["train"]
else:
    dataset = raw_dataset

# Select first 10 samples
try:
    dataset_10 = dataset.select(range(10))
except AttributeError:
    # If select is not available, fallback to slicing
    dataset_10 = dataset[:10]

# Inspect field names
print("✅ Dataset fields:", list(dataset_10[0].keys()))

# Step 2: Extract just the raw code
code_samples = [item["text"] for item in dataset_10]

print(f"\n🧪 Loaded {len(code_samples)} code snippets.")

# Step 3: Load CodeBERT for code embeddings
print("\n🤖 Loading CodeBERT model...")
code_embedder = SentenceTransformer("microsoft/codebert-base")

# Step 4: Generate embeddings for each code snippet
print("📐 Generating embeddings...")
code_vectors = code_embedder.encode(code_samples)

# Step 5: Show embedding shape
print("\n🔍 First embedding vector (first 10 values):")
print(code_vectors[0][:10])
print("➡️ Vector dimension:", len(code_vectors[0]))

# Step 6: Save embeddings (optional for future use)
np.save("codebert_embeddings.npy", code_vectors)

print("\n✅ Success! You now have embedded code vectors.")