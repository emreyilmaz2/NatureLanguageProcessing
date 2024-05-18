import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# BERT modelini ve tokenizer'ı yükle
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# BERT vektör temsillerinin bulunduğu dizin
bert_vectors_directory = 'bert_vectors'
processed_texts_directory = 'processed_texts'  # İşlenmiş TXT dosyalarının bulunduğu dizin

# Girdi cümlesi
input_sentence = "netflix"

# Girdi cümlesi için BERT vektör temsili oluşturma
inputs = tokenizer(input_sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
# Tüm token'ların ortalamasını alarak vektör temsili oluşturma
input_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

# Tüm vektör temsillerini yükleme ve benzerlik hesaplama
similarities = []
for filename in os.listdir(bert_vectors_directory):
    if filename.endswith('.npy'):
        filepath = os.path.join(bert_vectors_directory, filename)
        vectors = np.load(filepath)
        for vector in vectors:
            similarity = cosine_similarity([input_vector], [vector.flatten()])[0][0]
            similarities.append((similarity, filename))

# En uyumlu 5 sonucu bulma
top_5_similar = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]

print("BERT ile en uyumlu 5 sonuç:")
for similarity, filename in top_5_similar:
    # Dosyanın orijinal içeriğini yazdırma
    original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
    with open(original_filepath, 'r') as file:
        content = file.read()
    print(f"Dosya: {filename}, Benzerlik: {similarity}")
    print(f"İçerik: {content}\n")

# En uyumsuz 5 sonucu bulma
bottom_5_similar = sorted(similarities, key=lambda x: x[0])[:5]

print("BERT ile en uyumsuz 5 sonuç:")
for similarity, filename in bottom_5_similar:
    # Dosyanın orijinal içeriğini yazdırma
    original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
    with open(original_filepath, 'r') as file:
        content = file.read()
    print(f"Dosya: {filename}, Benzerlik: {similarity}")
    print(f"İçerik: {content}\n")
