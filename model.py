
# content- based recommendation, it Recommens articles similar to the given article

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

df = pd.read_csv('articles_dataset.csv')


print(df.head())

articles = df['Content'].tolist()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embeddings(content):
    inputs = tokenizer(content, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

article_embeddings = [get_bert_embeddings(article) for article in articles]
article_embeddings = torch.stack(article_embeddings)
print(article_embeddings.shape)

content_similarity_matrix = cosine_similarity(article_embeddings)
print(content_similarity_matrix.shape)


def recommend_similar_articles(article_index, top_k=3):
    similarities = content_similarity_matrix[article_index]
    top_indices = similarities.argsort()[-top_k-1:-1][::-1]
    return top_indices

# Example
article_index = 7
recommended_indices = recommend_similar_articles(article_index)
print("Recommended Articles (Content-Based):")
for idx in recommended_indices:
    print(f"Article {idx}: {articles[idx]}")





# generative recommendation
# recommends articles similar to the given article based on user interactions

import numpy as np

# Created a user-item interaction dataset
num_users = 100
num_articles = len(articles)
np.random.seed(42)

# Generate random interactions (ratings between 1 and 5)
user_item_interactions = np.random.randint(1, 6, size=(num_users, num_articles))


user_item_df = pd.DataFrame(user_item_interactions, columns=[f"Article_{i}" for i in range(num_articles)])
user_item_df.index.name = "User_ID"

# Normalization
user_mean_ratings = user_item_df.mean(axis=1)
user_item_normalized = user_item_df.sub(user_mean_ratings, axis=0)


#item-item similarity matrix
item_similarity_matrix = cosine_similarity(user_item_normalized.T)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_item_df.columns, columns=user_item_df.columns)


def recommend_similar_items(item_id, top_k=5):
    similarities = item_similarity_df[item_id]
    top_indices = similarities.sort_values(ascending=False).index[1:top_k+1]  # Exclude the item itself
    return top_indices

# Example
item_id = "Article_7"
recommended_items = recommend_similar_items(item_id)
print(f"Recommended items for {item_id} (Collaborative Filtering): {recommended_items.tolist()}")


# Hybrid Recommendation System


alpha = 0.5
hybrid_similarity_matrix = (alpha * content_similarity_matrix) + ((1 - alpha) * item_similarity_matrix)


def recommend_hybrid_articles(article_index, top_k=5):
    similarities = hybrid_similarity_matrix[article_index]
    top_indices = similarities.argsort()[-top_k-1:-1][::-1]
    return top_indices


article_index = 7
recommended_hybrid_indices = recommend_hybrid_articles(article_index)

print("Recommended Articles (Hybrid Model):")
for idx in recommended_hybrid_indices:
    print(f"Article {idx}: {articles[idx]}")



