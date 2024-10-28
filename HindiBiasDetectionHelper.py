import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

hindi_gender_direction = [["महिला","आदमी"],
          ["बच्ची","बच्चा"],
          ["बेटी","बेटा"],
          ["माँ","पिता"],
          ["लड़की","लड़का"],
          ["स्त्री","मर्द"],
          ["उसकी","उसका"],
          ["स्त्रीलिंग","पुल्लिंग"],
          ["औरत","आदमी"],
          ["विद्या", "राम"]
      ]

hindi_gendered_words = ["पुरुष"]

class HindiEmbedder:
    def __init__(self):
      
      self.model = SentenceTransformer('l3cube-pune/hindi-sentence-bert-nli')
      self.hindi_gender_direction = hindi_gender_direction
          
      self.gendered_words = hindi_gendered_words
      self.hindi_gender_subspace = self.create_gender_direction_embedding()

    def cosine_similarity(self,a, b):
      return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def hindi_word_embedding(self, hindi_word):
      tokens = word_tokenize(hindi_word)
      embedding = np.array(self.model.encode(tokens))
      return embedding[0]
    
    def create_gender_direction_embedding(self):
      gender_direction_vector = [[self.hindi_word_embedding(sent) for sent in pair] for pair in self.hindi_gender_direction]
      gender_direction_vector = [vect[0] - vect[1] for vect in gender_direction_vector]
      gender_direction_vector = np.array(gender_direction_vector)
      pca = PCA(n_components=1)
      pca.fit(gender_direction_vector)
      return pca.components_[0]

    def hindi_sentence_embedding(self, hindi_sentence):
      tokens = word_tokenize(hindi_sentence)
      embeddings = self.model.encode(tokens)
      sentence_embedding = self.model.encode(hindi_sentence)
      # Step 3: Calculate Cosine Similarity Between Each Word and Sentence Embedding
      similarities = cosine_similarity([sentence_embedding], embeddings)[0]

      # Step 4: Normalize Scores
      total_similarity = sum(similarities)
      normalized_scores = [sim / total_similarity for sim in similarities]
      return tokens,embeddings, normalized_scores
    
    def get_gender_bias_score_of_sentence(self, sentence):
      tokens, embeddings, word_importance = self.hindi_sentence_embedding(sentence)
      female_bias_score = 0
      male_bias_score = 0
      bias_tokens = {}
      for i in range(len(tokens)):
        token = tokens[i]
        if token.lower() not in self.gendered_words:
          word_vector = np.array(embeddings[i])
          similarity = self.cosine_similarity(word_vector, self.hindi_gender_subspace)
          bias_tokens[token.lower()] = {"cosine_similarity": similarity, "word_importance": word_importance}
          if similarity > 0:
            female_bias_score += similarity*word_importance[i]
          else:
            male_bias_score += similarity*word_importance[i]
      print(f"Female bias score: {female_bias_score}/1")
      print(f"Male bias score: {abs(male_bias_score)}/1")
      return {
          "female_bias_score" : female_bias_score,
          "male_bias_score" : male_bias_score,
          "bias_tokens": bias_tokens
      }