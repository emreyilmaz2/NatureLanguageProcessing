import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# İşlenmiş TXT dosyalarının bulunduğu dizin
input_directory = 'processed_texts'  # Bu dizini kendi TXT dosyalarınızın bulunduğu dizinle değiştirin

# SciBERT vektör temsillerinin kaydedileceği dizin
scibert_output_directory = 'scibert_vectors'
os.makedirs(scibert_output_directory, exist_ok=True)

# SciBERT modelini ve tokenizer'ı yükle
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# Her bir TXT dosyasını işleyip SciBERT vektör temsillerini oluşturma
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(input_directory, filename)
        
        # TXT dosyasını oku
        with open(filepath, 'r') as file:
            text = file.read()
        
        # Metni cümlelere böl
        sentences = text.split('\n')
        
        # Her cümlenin vektör temsillerini oluştur
        sentence_vectors = []
        for sentence in sentences:
            if sentence:
                inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
                outputs = model(**inputs)
                # [CLS] tokeninin çıktısını kullan
                cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
                sentence_vectors.append(cls_embedding)
        
        # Vektörleri numpy array olarak kaydet
        output_filepath = os.path.join(scibert_output_directory, os.path.splitext(filename)[0] + '.npy')
        np.save(output_filepath, np.array(sentence_vectors))

print(f"SciBERT vektör temsilleri '{scibert_output_directory}' dizinine kaydedildi.")
