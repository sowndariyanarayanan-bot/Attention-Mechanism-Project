import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer

# ==========================================================
# 1. Load Sample Sentences from CSV
# ==========================================================

df = pd.read_csv("dataset/sample_sentences.csv")

sentences = df["Sentence"].tolist()

print("Sample Sentences:")

for sentence in sentences:
    print(sentence)


# ==========================================================
# 2. Generate Sentence Embeddings
# ==========================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    sentences,
    convert_to_numpy=True
)

# Save embeddings in NumPy format
np.save(
    "dataset/embeddings.npy",
    embeddings
)

print("\nEmbeddings saved successfully!")

print("\nEmbedding Shape:")
print(embeddings.shape)


# ==========================================================
# 3. Create Query (Q), Key (K), Value (V)
# ==========================================================

np.random.seed(42)

embedding_dim = embeddings.shape[1]

W_Q = np.random.randn(embedding_dim, embedding_dim)
W_K = np.random.randn(embedding_dim, embedding_dim)
W_V = np.random.randn(embedding_dim, embedding_dim)

Q = embeddings @ W_Q
K = embeddings @ W_K
V = embeddings @ W_V

print("\nQuery (Q) Matrix:")
print(Q)

print("\nKey (K) Matrix:")
print(K)

print("\nValue (V) Matrix:")
print(V)


# ==========================================================
# 4. Calculate Attention Scores
# ==========================================================

attention_scores = Q @ K.T

print("\nAttention Scores:")
print(attention_scores)


# ==========================================================
# 5. Scaled Attention Scores
# ==========================================================

scaled_scores = attention_scores / np.sqrt(embedding_dim)

print("\nScaled Attention Scores:")
print(scaled_scores)


# ==========================================================
# 6. Softmax Function
# ==========================================================

def softmax(x):

    x = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


# ==========================================================
# 7. Attention Weights
# ==========================================================

attention_weights = softmax(scaled_scores)

print("\nAttention Weights:")
print(attention_weights)


# ==========================================================
# 8. Row Sum Check
# ==========================================================

row_sums = np.sum(
    attention_weights,
    axis=1
)

print("\nRow Sum of Attention Weights:")
print(row_sums)


# ==========================================================
# 9. Final Attention Output
# ==========================================================

attention_output = attention_weights @ V

print("\nFinal Attention Output:")
print(attention_output)


# ==========================================================
# 10. Save Attention Weights CSV
# ==========================================================

attention_df = pd.DataFrame(
    attention_weights,
    columns=[
        f"Sentence_{i+1}"
        for i in range(len(sentences))
    ]
)

attention_df.insert(
    0,
    "Sentence",
    sentences
)

attention_df.to_csv(
    "dataset/attention_weights.csv",
    index=False
)

print(
    "\nAttention weights saved to "
    "dataset/attention_weights.csv"
)


# ==========================================================
# 11. Save Attention Output CSV
# ==========================================================

output_df = pd.DataFrame(attention_output)

output_df.insert(
    0,
    "Sentence",
    sentences
)

output_df.to_csv(
    "dataset/attention_output.csv",
    index=False
)

print(
    "Attention output saved to "
    "dataset/attention_output.csv"
)


# ==========================================================
# 12. Attention Heatmap using Matplotlib
# ==========================================================

plt.figure(figsize=(10, 7))

plt.imshow(
    attention_weights,
    cmap="viridis"
)

plt.xticks(
    range(len(sentences)),
    [f"S{i+1}" for i in range(len(sentences))]
)

plt.yticks(
    range(len(sentences)),
    [f"S{i+1}" for i in range(len(sentences))]
)

plt.xlabel("Key Sentences")
plt.ylabel("Query Sentences")

plt.title(
    "Scaled Dot-Product Attention Heatmap"
)

plt.colorbar(
    label="Attention Weight"
)

plt.tight_layout()

plt.savefig(
    "dataset/attention_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nAttention heatmap saved to "
    "dataset/attention_heatmap.png"
)


# ==========================================================
# 13. PyTorch Implementation
# ==========================================================

Q_torch = torch.tensor(
    Q,
    dtype=torch.float32
)

K_torch = torch.tensor(
    K,
    dtype=torch.float32
)

V_torch = torch.tensor(
    V,
    dtype=torch.float32
)

d_k = K_torch.shape[-1]

torch_scores = torch.matmul(
    Q_torch,
    K_torch.T
) / np.sqrt(d_k)

torch_weights = torch.softmax(
    torch_scores,
    dim=-1
)

torch_output = torch.matmul(
    torch_weights,
    V_torch
)

print("\nPyTorch Attention Weights:")
print(torch_weights)

print("\nPyTorch Attention Output:")
print(torch_output)


# ==========================================================
# 14. Difference between NumPy and PyTorch
# ==========================================================

difference = np.max(
    np.abs(
        attention_weights -
        torch_weights.detach().numpy()
    )
)

print("\nDifference between NumPy and PyTorch:")
print(difference)


# ==========================================================
# 15. Embedding Dimension
# ==========================================================

print("\nEmbedding Dimension Experiment:")

for dimension in [4, 8, 16, 32]:

    X = embeddings[:, :dimension]

    WQ = np.random.randn(
        dimension,
        dimension
    )

    WK = np.random.randn(
        dimension,
        dimension
    )

    WV = np.random.randn(
        dimension,
        dimension
    )

    q = X @ WQ
    k = X @ WK
    v = X @ WV

    scores = (
        q @ k.T
    ) / np.sqrt(dimension)

    weights = softmax(scores)

    print(
        f"\nEmbedding Dimension: {dimension}"
    )

    print(
        "Attention Matrix Shape:",
        weights.shape
    )

    print(
        "Maximum Attention:",
        np.max(weights)
    )


# ==========================================================
# 16. Completion Message
# ==========================================================

print("\n==========================================")
print("Attention Mechanism Project Completed!")
print("==========================================")