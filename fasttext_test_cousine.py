import os
import numpy as np
import fasttext
from sklearn.metrics.pairwise import cosine_similarity

# FastText modelini yükleme
fasttext_model = fasttext.load_model('cc.en.300.bin')

# FastText vektör temsillerinin bulunduğu dizin
fasttext_vectors_directory = 'fasttext_vectors'
processed_texts_directory = 'processed_texts'  # İşlenmiş TXT dosyalarının bulunduğu dizin

# Girdi cümlesi
input_sentence = "netflix"

# Girdi cümlesi için FastText vektör temsili oluşturma
def get_fasttext_sentence_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.get_word_vector(word) for word in words if word in model]
    if word_vectors:
        sentence_vector = np.mean(word_vectors, axis=0)
    else:
        sentence_vector = np.zeros(model.get_dimension())
    return sentence_vector

input_vector = get_fasttext_sentence_vector(input_sentence, fasttext_model)

# Tüm vektör temsillerini yükleme ve benzerlik hesaplama
similarities = []
for filename in os.listdir(fasttext_vectors_directory):
    if filename.endswith('.npy'):
        filepath = os.path.join(fasttext_vectors_directory, filename)
        vectors = np.load(filepath)
        for vector in vectors:
            similarity = cosine_similarity([input_vector], [vector])[0][0]
            similarities.append((similarity, filename))

# En uyumlu 5 sonucu bulma
top_5_similar = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]

print("FastText ile en uyumlu 5 sonuç:")
for similarity, filename in top_5_similar:
    # Dosyanın orijinal içeriğini yazdırma
    original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
    with open(original_filepath, 'r') as file:
        content = file.read()
    print(f"Dosya: {filename}, Benzerlik: {similarity}")
    print(f"İçerik: {content}\n")

# En uyumsuz 5 sonucu bulma
bottom_5_similar = sorted(similarities, key=lambda x: x[0])[:5]

print("FastText ile en uyumsuz 5 sonuç:")
for similarity, filename in bottom_5_similar:
    # Dosyanın orijinal içeriğini yazdırma
    original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
    with open(original_filepath, 'r') as file:
        content = file.read()
    print(f"Dosya: {filename}, Benzerlik: {similarity}")
    print(f"İçerik: {content}\n")
