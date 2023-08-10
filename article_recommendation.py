import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity           
import warnings
import pickle
warnings.filterwarnings("ignore")

article = pd.read_csv("C:/Users/HP/Desktop/article.csv")

article_list = article.values.tolist()
flattened_list = [article[0] for article in article_list]

   
def recommend_article(a):
        # Preprocessed user question or keyword
        user_query = a

        # List of preprocessed paragraphs
        paragraphs = flattened_list

        # Combine user query and paragraphs for vectorization
        documents = [user_query] + paragraphs

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Calculate TF-IDF vectors for user query and paragraphs
        tfidf_vectors = vectorizer.fit_transform(documents)

        # Calculate cosine similarity between user query and paragraphs
        cosine_similarities = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1:])[0]

        # Sort and get the indices of paragraphs based on similarity scores
        similar_paragraph_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[0:10]

        # Retrieve the most similar paragraphs
        similar_paragraphs = [paragraphs[i] for i in similar_paragraph_indices]
        
        result = []
        for paragraph in similar_paragraphs:
            result.append(paragraph)

        return(result)


