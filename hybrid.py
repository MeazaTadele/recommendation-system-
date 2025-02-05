import numpy as np
import pandas as pd

from collaborative import item_similarity_matrix
from content_based import content_similarity_matrix

# Load dataset
df = pd.read_csv('articles_dataset.csv')
articles = df['Content'].tolist()

# Combine content-based and collaborative filtering similarity matrices
alpha = 0.5
hybrid_similarity_matrix = (alpha * content_similarity_matrix) + ((1 - alpha) * item_similarity_matrix)

def recommend_hybrid_articles(article_index, top_k=5):
    """Recommend articles using a hybrid model (content-based + collaborative filtering)."""
    similarities = hybrid_similarity_matrix[article_index]
    top_indices = similarities.argsort()[-top_k-1:-1][::-1]
    return top_indices

# Example usage
if __name__ == "__main__":
    article_index = 7
    recommended_hybrid_indices = recommend_hybrid_articles(article_index)

    print("Recommended Articles (Hybrid Model):")
    for idx in recommended_hybrid_indices:
        print(f"Article {idx}: {articles[idx]}")
