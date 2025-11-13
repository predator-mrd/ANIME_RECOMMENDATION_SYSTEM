# --- Install dependencies ---
%pip -q install sentence-transformers pandas numpy scikit-learn

# --- Imports ---
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
CSV_PATH = "/content/anime_with_synopsis.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10

# --- Load data (dataset-specific columns) ---
df = pd.read_csv(CSV_PATH)

required_cols = ["Name", "sypnopsis"]  # note: 'sypnopsis' is misspelled in the dataset
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Expected column '{c}' not found; got: {list(df.columns)}")

# --- Build combined text per row ---
def make_text(row):
    parts = []
    name = str(row["Name"]) if pd.notna(row["Name"]) else ""
    syn = str(row["sypnopsis"]) if pd.notna(row["sypnopsis"]) else ""
    genres = str(row["Genres"]) if "Genres" in df.columns and pd.notna(row["Genres"]) else ""
    if name: parts.append(f"Title: {name}")
    if genres: parts.append(f"Genres: {genres}")
    if syn: parts.append(f"Synopsis: {syn}")
    return " ".join(parts) if parts else "No content"

df["combined_text"] = df.apply(make_text, axis=1)

# --- Embed all items once ---
print("Loading model and encoding items...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)
embeddings = embeddings.astype(np.float32)

# --- Helpers ---
def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def best_title_index(query):
    q = clean(query).lower()
    if not q:
        return None
    # contains match
    mask = df["Name"].astype(str).str.lower().str.contains(re.escape(q), na=False)
    if mask.any():
        return int(df[mask].index[0])
    # exact match fallback
    mask2 = df["Name"].astype(str).str.lower().str.strip().eq(q)
    if mask2.any():
        return int(df[mask2].index[0])
    return None

def top_k(similarities, k=TOP_K, skip_index=None):
    order = np.argsort(-similarities)  # descending
    out = []
    for i in order:
        if skip_index is not None and i == skip_index:
            continue
        out.append((i, float(similarities[i])))
        if len(out) >= k:
            break
    return out

def recommend_from_title(title_text, k=TOP_K):
    i0 = best_title_index(title_text)
    if i0 is None:
        print(f"No title matched '{title_text}'. Falling back to semantic query.\n")
        return recommend_from_query(title_text, k)
    sims = cosine_similarity(embeddings[i0].reshape(1, -1), embeddings)[0]
    results = top_k(sims, k=k, skip_index=i0)
    print(f"\nTop {k} similar to '{df.iloc[i0]['Name']}':\n")
    print_results(results)

def recommend_from_query(description, k=TOP_K):
    q = clean(description)
    if not q:
        print("Please type a non-empty query or title.")
        return
    q_emb = model.encode([q]).astype(np.float32)
    sims = cosine_similarity(q_emb, embeddings)[0]
    results = top_k(sims, k=k)
    print(f"\nTop {k} matches for '{description}':\n")
    print_results(results)

def print_results(results):
    if not results:
        print("No results found.")
        return
    for rank, (i, sim) in enumerate(results, 1):
        name = str(df.iloc[i]["Name"])
        score = df.iloc[i]["Score"] if "Score" in df.columns else "N/A"
        genres = df.iloc[i]["Genres"] if "Genres" in df.columns else ""
        print(f"{rank}. {name} | Score: {score} | Similarity: {sim:.3f}")
        if genres and str(genres).lower() != "nan":
            print(f"   Genres: {genres}")
        print()

# --- Simple CLI (no prefixes needed) ---
print("=" * 60)
print("ANIME RECOMMENDER")
print("Type either a title (e.g., naruto) or a description (e.g., dark fantasy psychological).")
print("Add a leading '>' if you like; it's ignored. Type 'exit' to quit.")
print("=" * 60)

while True:
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not raw:
        continue
    if raw.lower() == "exit":
        print("Goodbye!")
        break
    # ignore a leading '>' like shell prompts
    if raw.startswith(">"):
        raw = raw.lstrip(">").strip()
    # auto-detect: try title match, else treat as free-text query
    idx = best_title_index(raw)
    if idx is not None:
        recommend_from_title(raw, TOP_K)
    else:
        recommend_from_query(raw, TOP_K)
