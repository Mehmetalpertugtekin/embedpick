import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QUERIES = ["para iadesi", "iade", "iadesi", "para", "ürün iadesi"]
TOP_N = 8

corpus = pd.read_csv("data/corpus.csv", encoding="utf-8")
model = SentenceTransformer(MODEL)

corpus_emb = model.encode(corpus["text"].tolist(), normalize_embeddings=True)

for query in QUERIES:
    query_emb = model.encode([query], normalize_embeddings=True)
    scores = corpus_emb @ query_emb[0]
    order = np.argsort(-scores)[:TOP_N]

    print(f'\n=== "{query}" ===')
    for rank, i in enumerate(order, start=1):
        mark = " <<<" if corpus["id"][i] in (5, 16) else ""
        print(f"{rank:2d}. {scores[i]:.4f}  [{corpus['id'][i]}] {corpus['text'][i]}{mark}")