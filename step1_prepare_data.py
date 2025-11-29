import pandas as pd
import json
import re
import matplotlib
matplotlib.use('Agg') # Force saving to file
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# 1. Load Data
print("Loading data...")
df = pd.read_csv("arxiv_sample.csv")

# 2. Tokenizer
def tokenize(text):
    if not isinstance(text, str): return []
    text = text.lower()
    # Keep words > 2 chars
    tokens = re.findall(r'\b[a-z]{3,}\b', text)
    return tokens

# 3. Build Index
inverted_index = defaultdict(lambda: defaultdict(int))
# Counter for visualization
global_word_count = Counter()

print("Building Inverted Index...")
for idx, row in df.iterrows():
    doc_id = str(row['doc_id'])
    text = str(row['title']) + " " + str(row['abstract'])
    words = tokenize(text)
    
    # Update index and global count
    global_word_count.update(words)
    for word in words:
        inverted_index[word][doc_id] += 1

# 4. Save JSON
print("Saving inverted_index.json...")
final_index = {k: dict(v) for k, v in inverted_index.items()}
with open("inverted_index.json", "w") as f:
    json.dump(final_index, f)

# --- TERMINAL OUTPUT FIRST ---
print("\n" + "="*30)
print("TOP 20 FREQUENT TERMS")
print("="*30)
top_words = global_word_count.most_common(20)
terms, freqs = zip(*top_words)

for t, f in top_words:
    print(f"{t:<15} : {f}")
print("="*30 + "\n")

# --- GENERATE PNG LAST ---
print("Generating 'step1_inverted_index.png'...")

plt.figure(figsize=(10, 6))
plt.bar(terms, freqs, color='teal')
plt.title('Step 1: Top 20 Terms in Inverted Index')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('step1_inverted_index.png')
plt.close()

print(" 🗸 Step 1 Complete: Index saved, View 'step1_inverted_index.png'.")