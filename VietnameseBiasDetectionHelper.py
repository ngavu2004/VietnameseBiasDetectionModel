from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import regex as re
import torch
from spacy.lang.vi import Vietnamese
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from transformers import AutoConfig, AutoModel, AutoTokenizer
from pyvi import ViTokenizer
import json

bert_pretrained_model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
gendered_words = ["đàn_bà","cô_ấy","con_gái","mẹ","bố",
                  "phụ_nữ", "vợ", "con_trai", "mẹ", "cha","bố" "con_gái", 
                   "chàng_trai", "các_cô_gái", "cô_gái", "cậu_bé", "anh_trai", "em_trai", 
                  "nữ", "chị_gái", "nam", "anh", "bố", "nữ_diễn_viên", "bạn_gái", "quý_bà", "bạn_trai", 
]

vietnamese_gender_direction = [
    ["đàn_bà","đàn_ông"],
    ["cô_ấy","anh_ấy"],
    ["con_gái","con_trai"],
    ["mẹ","bố"],
    ["cô_gái", "chàng_trai"],
    ["nữ", "nam"],
    ["nữ_tính","nam_tính"],
    ["Thúy", "Hùng"]
]

class VietnameseEmbedder:
  def __init__(self):
    self.model_config = AutoConfig.from_pretrained(bert_pretrained_model_name, output_attentions = True)
    self.model = AutoModel.from_pretrained(bert_pretrained_model_name, config=self.model_config)
    self.tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_model_name)
    self.gendered_words = gendered_words
    self.gender_direction = vietnamese_gender_direction
    self.gender_subspace = self.create_gender_direction_embedding()
    
  def cosine_similarity(self,a, b):
      return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

  def get_word_embedding(self, word):
    # Tokenize the word and convert to tensor format for the model
    inputs = self.tokenizer(word, return_tensors="pt")
    outputs = self.model (**inputs)

    # Get the embedding for the specific word token
    # BERT's output gives us a hidden state for each layer. We'll take the last layer's output.
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

    # The first token in the sequence is often `[CLS]`, so we take the embedding for our target token
    word_embedding = last_hidden_state[0][1]  # Get embedding for "example" (2nd token)

    # Convert to numpy for easier handling
    word_embedding = word_embedding.detach().numpy()
    return word_embedding

  def create_gender_direction_embedding(self):
    gender_direction_vector = [[self.get_word_embedding(sent) for sent in pair] for pair in self.gender_direction]
    gender_direction_vector = [vect[0] - vect[1] for vect in gender_direction_vector]
    gender_direction_vector = np.array(gender_direction_vector)
    pca = PCA(n_components=1)
    pca.fit(gender_direction_vector)
    return pca.components_[0]

  def get_sentence_embedding(self, sentence):
    tokens = self.tokenizer.tokenize(sentence)
    embeddings = []
    # apply separate word embedding on each token in the sentence
    for token in tokens:
      embedding = self.get_word_embedding(token)
      embeddings.append(embedding)

    # apply word embedding and extract the gradient for each token
    inputs = self.tokenizer(sentence, return_tensors="pt")
    outputs = self.model(**inputs)
    # Get attention weights from all layers
    # Shape of attentions: (num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions

    # Average across heads
    mean_attention_across_heads = torch.mean(torch.stack(attentions), dim=2)  # Average across heads

    # Average across layers
    mean_attention_across_layers = torch.mean(mean_attention_across_heads, dim=0)  # Average across layers

    # Get the attention weights for the [CLS] token with respect to each word
    # The [CLS] token often holds global sentence information in BERT
    cls_attention = mean_attention_across_layers[0, 0, :].detach().numpy()

    # Convert tokens back to readable words
    tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze())[1:-1]
    token_scores = [(token, cls_attention[i]) for i, token in enumerate(tokens)]

    # Normalize scores and print
    total_score = sum(score for _, score in token_scores)
    token_scores_normalized = [(score / total_score) for token, score in token_scores]

    print(f"Tokens length: {len(tokens)}")
    print(f"Embeddings length: {len(embeddings)}")
    print(f"Importance scores length: {len(token_scores_normalized)}")
    return tokens, embeddings, token_scores_normalized

  def get_gender_bias_score_of_sentence(self, sentence):
    tokens, embeddings, importance_scores = self.get_sentence_embedding(sentence)
    # word_importance = 1/len(tokens)
    female_bias_score = 0
    male_bias_score = 0
    bias_tokens = {}
    for i in range(len(tokens)):
      token = tokens[i]
      if token.lower() not in self.gendered_words:
        print(f"Token: {token}")
        print(f"Importance scores: {importance_scores[i]}")
        word_vector = np.array(embeddings[i])
        similarity = self.cosine_similarity(word_vector, self.gender_subspace)
        bias_tokens[token.lower()] = {"cosine_similarity": similarity, "word_importance": importance_scores[i]}
        if similarity > 0:
          female_bias_score += similarity*importance_scores[i]
        else:
          male_bias_score += similarity*importance_scores[i]
    print(f"Female bias score: {female_bias_score*100}%")
    print(f"Male bias score: {abs(male_bias_score)*100}%")
    return {
        "female_bias_score" : female_bias_score,
        "male_bias_score" : male_bias_score,
        "bias_tokens": bias_tokens,
    }