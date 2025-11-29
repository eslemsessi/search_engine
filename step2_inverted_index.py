import pandas as pd
import json
import re
from collections import defaultdict

# 1. Load the data created in the previous step
print("Loading data...")
df = pd.read_csv("arxiv_sample.csv")

# 2. Define a simple tokenizer
def tokenize(text):
    if not isinstance(text, str): return []
    text = text.lower()
    # Keep only words with a-z and length > 2
    tokens = re.findall(r'\b[a-z]{3,}\b', text)
    return tokens

# 3. Build the Index
# Structure: { "algorithm": { "doc_1": 5, "doc_2": 1 } }
inverted_index = defaultdict(lambda: defaultdict(int))

print("Building Inverted Index... (This takes a moment)")

for idx, row in df.iterrows():
    doc_id = str(row['doc_id']) # Ensure ID is a string for JSON
    text = str(row['title']) + " " + str(row['abstract'])
    
    words = tokenize(text)
    
    for word in words:
        inverted_index[word][doc_id] += 1

# 4. Save to JSON (Deliverable 1)
output_file = "inverted_index.json"
print(f"Saving to {output_file}...")

# Convert to standard dict for saving
final_index = {k: dict(v) for k, v in inverted_index.items()}

with open(output_file, "w") as f:
    json.dump(final_index, f)

print(f" 🗸 Task 1 Complete! Index saved. Vocabulary size: {len(final_index)} words.")