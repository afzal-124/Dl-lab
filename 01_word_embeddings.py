# Experiment 1: Generate word embeddings using Word2Vec
# Install: pip install gensim

from gensim.models import Word2Vec

corpus = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["deep", "learning"],
]

model = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=2,
    min_count=1,
    workers=1,
    seed=42,
    epochs=100,
)

print("Embedding for 'learning':")
print(model.wv["learning"])

print("\nWords similar to 'learning':")
for word, score in model.wv.most_similar("learning"):
    print(f"{word}: {score:.4f}")
