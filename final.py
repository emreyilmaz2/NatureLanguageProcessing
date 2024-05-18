import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

# Girdi cümlesi
input_sentence = "netflix"

# FastText modelini yükleme
import fasttext
fasttext_model = fasttext.load_model('cc.en.300.bin')

# BERT ve SciBERT modellerini ve tokenizer'larını yükle
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# Vektör temsillerinin bulunduğu dizinler
fasttext_vectors_directory = 'fasttext_vectors'
scibert_vectors_directory = 'scibert_vectors'
bert_vectors_directory = 'bert_vectors'
processed_texts_directory = 'processed_texts'  # İşlenmiş TXT dosyalarının bulunduğu dizin

# FastText için cümle vektörü oluşturma
def get_fasttext_sentence_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.get_word_vector(word) for word in words if word in model]
    if word_vectors:
        sentence_vector = np.mean(word_vectors, axis=0)
    else:
        sentence_vector = np.zeros(model.get_dimension())
    return sentence_vector

# BERT ve SciBERT için cümle vektörü oluşturma
def get_bert_sentence_vector(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    sentence_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return sentence_vector

# Girdi cümlesi için vektör temsilleri oluşturma
input_vector_fasttext = get_fasttext_sentence_vector(input_sentence, fasttext_model)
input_vector_bert = get_bert_sentence_vector(input_sentence, bert_tokenizer, bert_model)
input_vector_scibert = get_bert_sentence_vector(input_sentence, scibert_tokenizer, scibert_model)

# Vektör temsillerini yükleme ve benzerlik hesaplama
def calculate_similarities(input_vector, vectors_directory):
    similarities = []
    for filename in os.listdir(vectors_directory):
        if filename.endswith('.npy'):
            filepath = os.path.join(vectors_directory, filename)
            vectors = np.load(filepath)
            for vector in vectors:
                similarity = cosine_similarity([input_vector], [vector.flatten()])[0][0]
                similarities.append((similarity, filename))
    return similarities
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

# Girdi cümlesi
input_sentence = "netflix"

# FastText modelini yükleme
import fasttext
fasttext_model = fasttext.load_model('cc.en.300.bin')

# BERT ve SciBERT modellerini ve tokenizer'larını yükle
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# Vektör temsillerinin bulunduğu dizinler
fasttext_vectors_directory = 'fasttext_vectors'
scibert_vectors_directory = 'scibert_vectors'
bert_vectors_directory = 'bert_vectors'
processed_texts_directory = 'processed_texts'  # İşlenmiş TXT dosyalarının bulunduğu dizin

# FastText için cümle vektörü oluşturma
def get_fasttext_sentence_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.get_word_vector(word) for word in words if word in model]
    if word_vectors:
        sentence_vector = np.mean(word_vectors, axis=0)
    else:
        sentence_vector = np.zeros(model.get_dimension())
    return sentence_vector

# BERT ve SciBERT için cümle vektörü oluşturma
def get_bert_sentence_vector(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    sentence_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return sentence_vector

# Girdi cümlesi için vektör temsilleri oluşturma
input_vector_fasttext = get_fasttext_sentence_vector(input_sentence, fasttext_model)
input_vector_bert = get_bert_sentence_vector(input_sentence, bert_tokenizer, bert_model)
input_vector_scibert = get_bert_sentence_vector(input_sentence, scibert_tokenizer, scibert_model)

# Vektör temsillerini yükleme ve benzerlik hesaplama
def calculate_similarities(input_vector, vectors_directory):
    similarities = []
    for filename in os.listdir(vectors_directory):
        if filename.endswith('.npy'):
            filepath = os.path.join(vectors_directory, filename)
            vectors = np.load(filepath)
            for vector in vectors:
                similarity = cosine_similarity([input_vector], [vector.flatten()])[0][0]
                similarities.append((similarity, filename))
    return similarities

# En uyumlu ve en uyumsuz sonuçları bulma
def find_top_and_bottom_similarities(similarities, top_n=5):
    top_similar = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_n]
    bottom_similar = sorted(similarities, key=lambda x: x[0])[:top_n]
    return top_similar, bottom_similar

# Benzerlik hesaplamaları
similarities_fasttext = calculate_similarities(input_vector_fasttext, fasttext_vectors_directory)
similarities_bert = calculate_similarities(input_vector_bert, bert_vectors_directory)
similarities_scibert = calculate_similarities(input_vector_scibert, scibert_vectors_directory)

# En uyumlu ve en uyumsuz 5 sonucu bulma
top_5_similar_fasttext, bottom_5_similar_fasttext = find_top_and_bottom_similarities(similarities_fasttext)
top_5_similar_bert, bottom_5_similar_bert = find_top_and_bottom_similarities(similarities_bert)
top_5_similar_scibert, bottom_5_similar_scibert = find_top_and_bottom_similarities(similarities_scibert)

# Sonuçları yazdırma
def print_similarities(top_similar, bottom_similar, model_name):
    print(f"{model_name} ile en uyumlu 5 sonuç:")
    for similarity, filename in top_similar:
        # Dosyanın orijinal içeriğini yazdırma
        original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
        with open(original_filepath, 'r') as file:
            content = file.read()
        print(f"Dosya: {filename}, Benzerlik: {similarity}")
        print(f"İçerik: {content}\n")

    print(f"{model_name} ile en uyumsuz 5 sonuç:")
    for similarity, filename in bottom_similar:
        # Dosyanın orijinal içeriğini yazdırma
        original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
        with open(original_filepath, 'r') as file:
            content = file.read()
        print(f"Dosya: {filename}, Benzerlik: {similarity}")
        print(f"İçerik: {content}\n")

# FastText sonuçları
print_similarities(top_5_similar_fasttext, bottom_5_similar_fasttext, "FastText")

# BERT sonuçları
print_similarities(top_5_similar_bert, bottom_5_similar_bert, "BERT")

# SciBERT sonuçları
print_similarities(top_5_similar_scibert, bottom_5_similar_scibert, "SciBERT")

# En uyumlu ve en uyumsuz sonuçları bulma
def find_top_and_bottom_similarities(similarities, top_n=5):
    top_similar = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_n]
    bottom_similar = sorted(similarities, key=lambda x: x[0])[:top_n]
    return top_similar, bottom_similar

# Benzerlik hesaplamaları
similarities_fasttext = calculate_similarities(input_vector_fasttext, fasttext_vectors_directory)
similarities_bert = calculate_similarities(input_vector_bert, bert_vectors_directory)
similarities_scibert = calculate_similarities(input_vector_scibert, scibert_vectors_directory)

# En uyumlu ve en uyumsuz 5 sonucu bulma
top_5_similar_fasttext, bottom_5_similar_fasttext = find_top_and_bottom_similarities(similarities_fasttext)
top_5_similar_bert, bottom_5_similar_bert = find_top_and_bottom_similarities(similarities_bert)
top_5_similar_scibert, bottom_5_similar_scibert = find_top_and_bottom_similarities(similarities_scibert)

# Sonuçları yazdırma
def print_similarities(top_similar, bottom_similar, model_name):
    print(f"{model_name} ile en uyumlu 5 sonuç:")
    for similarity, filename in top_similar:
        # Dosyanın orijinal içeriğini yazdırma
        original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
        with open(original_filepath, 'r') as file:
            content = file.read()
        print(f"Dosya: {filename}, Benzerlik: {similarity}")
        print(f"İçerik: {content}\n")

    print(f"{model_name} ile en uyumsuz 5 sonuç:")
    for similarity, filename in bottom_similar:
        # Dosyanın orijinal içeriğini yazdırma
        original_filepath = os.path.join(processed_texts_directory, os.path.splitext(filename)[0] + '.txt')
        with open(original_filepath, 'r') as file:
            content = file.read()
        print(f"Dosya: {filename}, Benzerlik: {similarity}")
        print(f"İçerik: {content}\n")

# FastText sonuçları
print_similarities(top_5_similar_fasttext, bottom_5_similar_fasttext, "FastText")

# BERT sonuçları
print_similarities(top_5_similar_bert, bottom_5_similar_bert, "BERT")

# SciBERT sonuçları
print_similarities(top_5_similar_scibert, bottom_5_similar_scibert, "SciBERT")
