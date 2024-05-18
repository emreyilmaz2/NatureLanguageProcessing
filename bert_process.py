import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# İşlenmiş TXT dosyalarının bulunduğu dizin
input_directory = 'processed_texts'  # Bu dizini kendi TXT dosyalarınızın bulunduğu dizinle değiştirin

# BERT vektör temsillerinin kaydedileceği dizin
bert_output_directory = 'bert_vectors'
os.makedirs(bert_output_directory, exist_ok=True)

# BERT modelini ve tokenizer'ı yükle
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Her bir TXT dosyasını işleyip BERT vektör temsillerini oluşturma
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
                # Tüm token'ların ortalamasını alarak vektör temsili oluşturma
                sentence_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                sentence_vectors.append(sentence_vector)
        
        # Vektörleri numpy array olarak kaydet
        output_filepath = os.path.join(bert_output_directory, os.path.splitext(filename)[0] + '.npy')
        np.save(output_filepath, np.array(sentence_vectors))

print(f"BERT vektör temsilleri '{bert_output_directory}' dizinine kaydedildi.")
