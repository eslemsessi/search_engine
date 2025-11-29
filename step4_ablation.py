import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import nltk

# Download nltk data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("Loading data for Ablation Study...")
df = pd.read_csv("arxiv_sample.csv")
raw_docs = (df['title'].fillna('') + ' ' + df['abstract'].fillna('')).tolist()

# Define Queries and Targets (Same as step 3)
queries_config = {
    "quantum field theory": ["hep-th", "quant-ph"],
    "graph neural network": ["cs.LG", "cs.AI"],
    "black hole thermodynamics": ["gr-qc", "astro-ph"],
    "natural language processing": ["cs.CL"]
}

# Helper to run the full eval pipeline for a specific vectorizer setup
def run_pipeline(name, vectorizer_obj, documents):
    X = vectorizer_obj.fit_transform(documents)
    
    ap_scores = []
    for query_text, targets in queries_config.items():
        q_vec = vectorizer_obj.transform([query_text])
        sims = cosine_similarity(q_vec, X).flatten()
        top_indices = np.argsort(sims)[::-1][:10]
        results = df.iloc[top_indices]
        
        # Calculate AP
        relevant_count = 0
        precisions = []
        for i, row in enumerate(results.itertuples()):
            if any(t in row.categories for t in targets):
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        if precisions:
            ap_scores.append(sum(precisions)/len(precisions))
        else:
            ap_scores.append(0)
            
    return np.mean(ap_scores)

print("\nStarting Comparison (This takes time)...")

# --- CONFIG 1: No Stopwords, No Stemming ---
print("1. Testing: Raw...")
map_1 = run_pipeline("Raw", TfidfVectorizer(stop_words=None, min_df=5), raw_docs)

# --- CONFIG 2: Stopwords, No Stemming ---
print("2. Testing: Stopwords...")
map_2 = run_pipeline("Stopwords", TfidfVectorizer(stop_words='english', min_df=5), raw_docs)

# --- CONFIG 3 & 4: Stemming Setup ---
print("Pre-processing Stemming (PorterStemmer)...")
stemmer = PorterStemmer()
stemmed_docs = []
for doc in raw_docs:
    # Simple split and stem
    words = doc.lower().split()
    stemmed = [stemmer.stem(w) for w in words]
    stemmed_docs.append(" ".join(stemmed))

# --- CONFIG 3: Stemming, No Stopwords ---
print("3. Testing: Stemming...")
map_3 = run_pipeline("Stemming", TfidfVectorizer(stop_words=None, min_df=5), stemmed_docs)

# --- CONFIG 4: Stemming + Stopwords ---
print("4. Testing: Stemming + Stopwords...")
map_4 = run_pipeline("Both", TfidfVectorizer(stop_words='english', min_df=5), stemmed_docs)

# --- FINAL TABLE ---
print("\n" + "="*40)
print(f"{'Configuration':<25} | {'MAP Score':<10}")
print("="*40)
print(f"{'1. Raw (No prep)':<25} | {map_1:.4f}")
print(f"{'2. Stopwords Only':<25} | {map_2:.4f}")
print(f"{'3. Stemming Only':<25} | {map_3:.4f}")
print(f"{'4. Stopwords + Stemming':<25} | {map_4:.4f}")
print("="*40)