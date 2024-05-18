import os
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string



# NLTK bileşenleri
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Ek karakterler
additional_punctuations = {"'", "-", "''", "--"}

# Ön işlem fonksiyonu
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation and word not in additional_punctuations]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# XML dosyalarının bulunduğu dizin
input_directory = 'Dataset/500N-KPCrowd/train'  # Bu dizini kendi XML dosyalarınızın bulunduğu dizinle değiştirin

# TXT dosyalarının kaydedileceği dizin
output_directory = 'processed_texts'  # Bu dizini kendi hedef dizininizle değiştirin
os.makedirs(output_directory, exist_ok=True)

# Dizindeki tüm XML dosyalarını işle
for filename in os.listdir(input_directory):
    if filename.endswith('.xml'):
        filepath = os.path.join(input_directory, filename)
        
        # XML dosyasını oku
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Sonuçları kaydetmek için
        preprocessed_sentences = []
        
        # XML yapısını gez ve her cümledeki kelimeleri ön işlemden geçir
        for sentence in root.iter('sentence'):
            sentence_text = ' '.join(token.find('word').text for token in sentence.find('tokens'))
            processed_sentence = preprocess(sentence_text)
            preprocessed_sentences.append(processed_sentence)
        
        # Her bir XML dosyası için ayrı bir TXT dosyası kaydet
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_filepath = os.path.join(output_directory, output_filename)
        with open(output_filepath, 'w') as f:
            for sentence in preprocessed_sentences:
                f.write(sentence + '\n')

print(f"Ön işleme tamamlandı ve sonuçlar '{output_directory}' dizinine kaydedildi.")
