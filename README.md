Here is a professional and structured README.md file for your repository. You can copy and paste this directly into your GitHub project.

🔍 Scientific Article Search Engine (arXiv)

![alt text](https://img.shields.io/badge/Python-3.8%2B-blue)


![alt text](https://img.shields.io/badge/Library-Scikit--Learn-orange)


![alt text](https://img.shields.io/badge/NLP-NLTK-green)

This project implements a comprehensive Information Retrieval (IR) System in Python. It applies the Vector Space Model (VSM) to a corpus of scientific articles extracted from the arXiv dataset, focusing on Physics, Computer Science, and Mathematics.

The system is built from scratch to demonstrate core IR concepts including indexing, vectorization, similarity scoring, and relevance feedback.

📋 Overview

The goal of this project is to retrieve relevant scientific abstracts based on user queries. It moves beyond simple keyword matching by using statistical weighting (TF-IDF) and optimizing results through user feedback simulations.

Key Features

📂 Inverted Index: Manual implementation of a dictionary-based inverted index structure.

🧮 TF-IDF & Cosine Similarity: Vectorization of documents and ranking based on cosine distance.

📊 Formal Evaluation: Calculation of Precision@k, Recall, and MAP (Mean Average Precision) using arXiv categories as a ground-truth proxy.

🔄 Rocchio Algorithm: Implementation of Relevance Feedback to optimize query vectors and improve results (Query Expansion).

🧪 Ablation Study: Analysis of the impact of text preprocessing steps (Stopwords, Stemming) on search performance.

⚙️ Installation
Prerequisites

Python 3.x

PIP (Python Package Manager)

1. Clone the Repository
code
Bash
download
content_copy
expand_less
git clone https://github.com/yourusername/arxiv-search-engine.git
cd arxiv-search-engine
2. Install Dependencies
code
Bash
download
content_copy
expand_less
pip install pandas numpy scikit-learn nltk matplotlib
3. Download Data

Due to size constraints, the raw dataset is not included in this repo.

Download the arXiv Dataset (arxiv-metadata-oai-snapshot.json) from Kaggle.

Place the JSON file in the root directory of the project.

🚀 Usage

To run the complete pipeline (Data loading -> Indexing -> Searching -> Evaluation -> Rocchio -> Ablation):

code
Bash
download
content_copy
expand_less
python main.py
What happens when you run it?

Data Sampling: It reads the large JSON file and creates a smaller arxiv_sample.csv (10k articles).

Indexing: It builds inverted_index.json.

Search: It executes 8 predefined queries (e.g., "Quantum Field Theory", "Graph Neural Network").

Feedback: It applies the Rocchio algorithm to improve the initial results.

Visualization: It generates a project_summary.png chart comparing performances.

📂 Project Structure
code
Code
download
content_copy
expand_less
├── arxiv-metadata-oai-snapshot.json  # (You must download this)
├── arxiv_sample.csv                  # Generated subset of data
├── main.py                           # Main execution script
├── inverted_index.json               # Generated Inverted Index
├── project_summary.png               # Output visualization of results
└── README.md                         # Project documentation
📊 Evaluation Metrics

The system performance is measured using the following metrics:

Metric	Description
Precision@10	The percentage of relevant documents in the top 10 results.
MAP	Mean Average Precision across all queries.
Recall	The ability of the system to find all relevant documents in the corpus.

Detailed results and comparisons between the Base Model and the Rocchio Optimized Model are printed to the console during execution.

👤 Author

Student: Eslem sessi

Course: Indexing and Referencing Techniques (University of Manouba - ISAMM)
