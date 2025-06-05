import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from utils import load_books_data
from thefuzz import process

class ContentBasedRecommender:
    def __init__(self,config_path):
        with open(config_path,'r') as f:
            self.config = yaml.safe_load(f)

        self.books = load_books_data(self.config['data']['books_csv'])
        self.top_n = self.config['recommendation']['top_n']

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.books['content'])

        #self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        # use this when high computation RAM available
    def find_closest_title(self,input_title,score_cutoff=70):
        titles = self.books['Book-Title'].tolist()
        results = process.extractBests(input_title,titles,score_cutoff=score_cutoff,limit=1)
        if results:
            return results[0][0]
        return None

    def recommend_books(self,book_title):
        matched_title = self.find_closest_title(book_title)
        if not matched_title:
            print(f"No Close match found for {book_title} . Try again")
            return 
        try:
            idx = self.books[self.books['Book-Title'].str.lower()==matched_title.lower()].index[0]
        except IndexError:
            print(f"Book '{matched_title}' not found in dataset")
            return []
        #sim_scores = list(enumerate(self.similarity_matrix[idx]))
        #sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
        #sim_scores = sim_scores[1: self.top_n+1]

        #recommended_indices = [i[0] for i in sim_scores]
        book_vector = self.tfidf_matrix[idx]
        sim_scores= cosine_similarity(book_vector,self.tfidf_matrix).flatten()
        sim_scores[idx]=-1
        top_indices = sim_scores.argsort()[::-1][:self.top_n]
        recommended_books =  self.books.iloc[top_indices][['Book-Title','Book-Author']]
       
        return recommended_books.reset_index(drop=True)

    def plot_similarity_distribution(self, book_title):
        matched_title = self.find_closest_title(book_title)
        if not matched_title:
            print(f"No Close match found for {matched_title} .Skipping plot")
            return 
        try:
            idx = self.books[self.books['Book-Title'].str.lower()==matched_title.lower()].index[0]
        except IndexError:
            print(f"Book '{matched_title}' not found in dataset")
            return 
        book_vector = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(book_vector,self.tfidf_matrix).flatten()
        #sim_scores = self.similarity_matrix[idx]   %run when high RAM available
        plt.hist(sim_scores,bins=30,color='skyblue',edgecolor='black')
        plt.title(f"Cosine similarity distribution for '{book_title}'")
        plt.xlabel('Similarity score')
        plt.ylabel('Frequency')
        plot_path = f"output/similarity_distribution_{book_title.replace(' ','_')}.png"
        plt.savefig(plot_path)
        plt.close()
        plt.show()

if __name__ == "__main__":
    import os

    config_path = os.path.join(os.path.dirname(__file__),'config.yaml')
    recommender = ContentBasedRecommender(config_path)

    while True:
        book_name = input("\nEnter a book title for Recommendation ( or exit to quit):").strip()
        if book_name.lower()=='exit':
            print('Annyeongggg')
            break

        recommendations = recommender.recommend_books(book_name)
        
        if recommendations.empty:
            print("Try another book title.")
        else:
            print(f"\nTop {recommender.top_n} recommendations similar to '{book_name}':\n")
            print(recommendations.to_string(index=False))
            recommender.plot_similarity_distribution(book_name)
            output_path = f"output/recommendations_for_{book_name.replace(' ','_')}.csv"
            recommendations.to_csv(output_path,index=False)
