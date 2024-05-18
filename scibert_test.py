import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# SciBERT modelini ve tokenizer'ı yükle
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# SciBERT vektör temsillerinin bulunduğu dizin
scibert_vectors_directory = 'scibert_vectors'
processed_texts_directory = 'processed_texts'  # İşlenmiş TXT dosyalarının bulunduğu dizin

# Girdi cümlesi
input_sentence = "netflix"

# Girdi cümlesi için SciBERT vektör temsili oluşturma
inputs = tokenizer(input_sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
# Tüm token'ların ortalamasını alarak vektör temsili oluşturma
input_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

# Tüm vektör temsillerini yükleme ve benzerlik hesaplama
similarities = []
for filename in os.listdir(scibert_vectors_directory):
    if filename.endswith('.npy'):
        filepath = os.path.join(scibert_vectors_directory, filename)
        vectors = np.load(filepath)
        for vector in vectors:
            similarity = cosine_similarity([input_vector], [vector.flatten()])[0][0]
            similarities.append((similarity, filename))

# En uyumsuz 5 sonucu bulma
bottom_5_similar = sorted(similarities, key=lambda x: x[0])[:5]

print("SciBERT ile en uyumsuz 5 sonuç:")
for similarity, filename in bottom_5_similar:
    # Dosyanın orijinal içeriğini yazdırma
    original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
    with open(original_filepath, 'r') as file:
        content = file.read()
    print(f"Dosya: {filename}, Benzerlik: {similarity}")
    print(f"İçerik: {content}\n")
