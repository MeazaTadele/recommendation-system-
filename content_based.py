import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

# Load dataset
df = pd.read_csv('articles_dataset.csv')
articles = df['Content'].tolist()

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embeddings(content):
    """Generate BERT embeddings for a given content."""
    inputs = tokenizer(content, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()


# Compute embeddings for all articles
article_embeddings = torch.stack([get_bert_embeddings(article) for article in articles])

# Compute content similarity matrix
content_similarity_matrix = cosine_similarity(article_embeddings)

def recommend_similar_articles(article_index, top_k=3):
    """Recommend articles similar to a given article index."""
    similarities = content_similarity_matrix[article_index]
    top_indices = similarities.argsort()[-top_k-1:-1][::-1]
    return top_indices

# Example usage
if __name__ == "__main__":
    article_index = 7
    recommended_indices = recommend_similar_articles(article_index)

    print("Recommended Articles (Content-Based):")
    for idx in recommended_indices:
        print(f"Article {idx}: {articles[idx]}")
