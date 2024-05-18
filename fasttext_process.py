import os
import fasttext
import numpy as np

# İşlenmiş TXT dosyalarının bulunduğu dizin
input_directory = 'processed_texts'  # Bu dizini kendi TXT dosyalarınızın bulunduğu dizinle değiştirin

# FastText vektör temsillerinin kaydedileceği dizin
fasttext_output_directory = 'fasttext_vectors'
os.makedirs(fasttext_output_directory, exist_ok=True)

# FastText önceden eğitilmiş modeli yükleme
model_path = 'cc.en.300.bin'  # FastText model dosyasının yolu
fasttext_model = fasttext.load_model(model_path)

# Her bir TXT dosyasını işleyip FastText vektör temsillerini oluşturma
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(input_directory, filename)
        
        # TXT dosyasını oku
        with open(filepath, 'r') as file:
            text = file.read()
        
        # Metni cümlelere böl
        sentences = text.split('\n')
        
        # Her cümlenin vektör temsillerini oluştur
        sentence_vectors = [fasttext_model.get_sentence_vector(sentence) for sentence in sentences if sentence]
        
        # Vektörleri numpy array olarak kaydet
        output_filepath = os.path.join(fasttext_output_directory, os.path.splitext(filename)[0] + '.npy')
        np.save(output_filepath, np.array(sentence_vectors))

print(f"FastText vektör temsilleri '{fasttext_output_directory}' dizinine kaydedildi.")