import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('articles_dataset.csv')
articles = df['Content'].tolist()
num_articles = len(articles)

# Simulate user-item interactions (ratings)
num_users = 100
np.random.seed(42)
user_item_interactions = np.random.randint(1, 6, size=(num_users, num_articles))

# Create DataFrame
user_item_df = pd.DataFrame(user_item_interactions, columns=[f"Article_{i}" for i in range(num_articles)])
user_item_df.index.name = "User_ID"

# Normalize user-item interactions
user_mean_ratings = user_item_df.mean(axis=1)
user_item_normalized = user_item_df.sub(user_mean_ratings, axis=0)

# Compute item-item similarity matrix
item_similarity_matrix = cosine_similarity(user_item_normalized.T)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_item_df.columns, columns=user_item_df.columns)

def recommend_similar_items(item_id, top_k=5):
    """Recommend similar articles based on collaborative filtering."""
    similarities = item_similarity_df[item_id]
    top_indices = similarities.sort_values(ascending=False).index[1:top_k+1]
    return top_indices

# Example usage
if __name__ == "__main__":
    item_id = "Article_7"
    recommended_items = recommend_similar_items(item_id)

    print(f"Recommended Articles for {item_id} (Collaborative Filtering):")
    print(recommended_items.tolist())
