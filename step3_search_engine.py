import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevents "no display" errors by saving to file directly
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print(f"Current Directory: {os.getcwd()}")
print("Loading data...")

if not os.path.exists("arxiv_sample.csv"):
    print("❌ ERROR: arxiv_sample.csv not found! Please run step 1 first.")
    exit()

df = pd.read_csv("arxiv_sample.csv")
# Create corpus (Title + Abstract)
docs = (df['title'].fillna('') + ' ' + df['abstract'].fillna('')).tolist()

# --- CONFIGURATION (Ground Truth) ---
queries_config = {
    "quantum field theory": ["hep-th", "quant-ph", "physics"],
    "graph neural network": ["cs.LG", "cs.AI", "cs.CV"],
    "black hole thermodynamics": ["gr-qc", "astro-ph"],
    "reinforcement learning": ["cs.LG", "cs.AI"],
    "dark matter galaxy": ["astro-ph"],
    "superconducting qubits": ["quant-ph", "cond-mat"],
    "natural language processing": ["cs.CL"],
    "gravitational waves": ["gr-qc", "astro-ph"]
}

# ==========================================
# 2. VECTORIZATION (TF-IDF)
# ==========================================
print("Vectorizing documents (TF-IDF)...")
vectorizer = TfidfVectorizer(stop_words='english', min_df=5)
X = vectorizer.fit_transform(docs)
print(f"Matrix Shape: {X.shape}")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def search(query_text, k=10, q_vec_input=None):
    """ Runs cosine similarity search. """
    if q_vec_input is None:
        q_vec = vectorizer.transform([query_text])
    else:
        q_vec = q_vec_input

    sims = cosine_similarity(q_vec, X).flatten()
    top_indices = np.argsort(sims)[::-1][:k]
    return df.iloc[top_indices], top_indices, q_vec

def count_total_relevant(target_cats):
    """ Counts total relevant docs in DB for Recall calculation. """
    count = 0
    for cats in df['categories']:
        if any(t in cats for t in target_cats):
            count += 1
    return count

def calculate_metrics(retrieved_df, target_cats, total_relevant_in_db):
    """ Calculates P@5, P@10, Recall@10, and AP. """
    categories_list = retrieved_df['categories'].tolist()
    
    # Binary vector: 1 if relevant, 0 if not
    is_relevant_list = []
    for cat_str in categories_list:
        relevant = any(t in cat_str for t in target_cats)
        is_relevant_list.append(1 if relevant else 0)
    
    # P@5 and P@10
    p5 = sum(is_relevant_list[:5]) / 5.0
    p10 = sum(is_relevant_list[:10]) / 10.0
    
    # Recall@10
    relevant_retrieved = sum(is_relevant_list[:10])
    r10 = relevant_retrieved / total_relevant_in_db if total_relevant_in_db > 0 else 0.0
        
    # AP (Average Precision)
    precisions = []
    rel_count = 0
    for i, val in enumerate(is_relevant_list):
        if val == 1:
            rel_count += 1
            precisions.append(rel_count / (i + 1))
            
    ap = sum(precisions) / len(precisions) if precisions else 0.0
        
    return p5, p10, r10, ap

# ==========================================
# 4. MAIN EXECUTION LOOP (EVALUATION)
# ==========================================
print("\n" + "="*85)
print(f"{'Query':<30} | {'P@5':<6} | {'P@10':<6} | {'Rec@10':<8} | {'AP':<6}")
print("="*85)

query_names = []
p5_scores = []
p10_scores = []
r10_scores = []
ap_scores = []

for query_text, targets in queries_config.items():
    # 1. Get total relevant for Recall
    total_rel = count_total_relevant(targets)
    
    # 2. Search
    results, _, _ = search(query_text, k=10)
    
    # 3. Calculate Metrics
    p5, p10, r10, ap = calculate_metrics(results, targets, total_rel)
    
    # 4. Store for Plotting
    query_names.append(query_text)
    p5_scores.append(p5)
    p10_scores.append(p10)
    r10_scores.append(r10)
    ap_scores.append(ap)
    
    print(f"{query_text[:28]:<30} | {p5:.3f}  | {p10:.3f}  | {r10:.4f}   | {ap:.3f}")

print("-" * 85)
print(f"Global MAP: {np.mean(ap_scores):.4f}")
print("=" * 85)


# ==========================================
# 5. ROCCHIO FEEDBACK ALGORITHM
# ==========================================
print("\n" + "="*60)
print(">>> RUNNING ROCCHIO FEEDBACK (Query Optimization)")
print("="*60)

# 1. Select a Test Query
q_test = "graph neural network"
targets_test = queries_config[q_test]
total_rel_test = count_total_relevant(targets_test)

print(f"Target Query: '{q_test}'")

# 2. Initial Search
res_old, idxs, vec_old = search(q_test)
_, _, _, ap_old = calculate_metrics(res_old, targets_test, total_rel_test)

# 3. Identify Relevant (D+) and Non-Relevant (D-) indices
rel_idxs = [i for i in idxs if any(t in targets_test for t in df.iloc[i]['categories'])]
nrel_idxs = [i for i in idxs if i not in rel_idxs]

print(f"Feedback found: {len(rel_idxs)} Relevant docs, {len(nrel_idxs)} Non-Relevant docs.")

# 4. Calculate Centroids
if rel_idxs:
    mean_pos = np.asarray(np.mean(X[rel_idxs], axis=0))
else:
    mean_pos = np.zeros((1, X.shape[1]))

if nrel_idxs:
    mean_neg = np.asarray(np.mean(X[nrel_idxs], axis=0))
else:
    mean_neg = np.zeros((1, X.shape[1]))

# 5. Update Vector (Alpha=1, Beta=0.75, Gamma=0.15)
vec_new = (1.0 * vec_old.toarray()) + (0.75 * mean_pos) - (0.15 * mean_neg)

# 6. Re-evaluate
res_new, _, _ = search(q_test, q_vec_input=vec_new)
_, _, _, ap_new = calculate_metrics(res_new, targets_test, total_rel_test)

print("-" * 40)
print(f"Original AP : {ap_old:.4f}")
print(f"Rocchio AP  : {ap_new:.4f}")
print(f"Improvement : {ap_new - ap_old:+.4f}")
print("="*60)


# ==========================================
# 6. GENERATE CHART (PNG) -- DONE LAST
# ==========================================
print("\nGenerating evaluation_metrics.png...")
try:
    x = np.arange(len(query_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - 1.5*width, p5_scores, width, label='P@5', color='#3498db')
    rects2 = ax.bar(x - 0.5*width, p10_scores, width, label='P@10', color='#2ecc71')
    rects3 = ax.bar(x + 0.5*width, r10_scores, width, label='Recall@10', color='#e74c3c')
    rects4 = ax.bar(x + 1.5*width, ap_scores, width, label='AP', color='#f1c40f')

    ax.set_ylabel('Score')
    ax.set_title('Search Engine Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace(" ", "\n") for q in query_names], rotation=0, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("step3_evaluation_metrics.png")
    plt.close()
    print(" 🗸 Chart saved successfully.")
except Exception as e:
    print(f"❗ Error saving chart: {e}")