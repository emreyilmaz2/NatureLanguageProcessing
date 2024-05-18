import os
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Metin Ön İşleme Fonksiyonu
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# XML Dosyalarını Okuma Fonksiyonu
def extract_text_from_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        sentences = root.findall('.//sentence')
        
        text = []
        for sentence in sentences:
            tokens = sentence.findall('.//token')
            for token in tokens:
                word = token.find('word').text
                text.append(word)
        
        return ' '.join(text)
    except ET.ParseError:
        print(f"Error parsing {xml_file}")
        return ""

def load_dataset(xml_folder):
    articles = []
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_file = os.path.join(xml_folder, filename)
            text = extract_text_from_xml(xml_file)
            articles.append(text)
    return articles

# Cümle Yerleştirmeleri için SBERT Modeli
model = SentenceTransformer('all-mpnet-base-v2')

def get_sentence_embeddings(sentences):
    return model.encode(sentences, convert_to_tensor=True)

# Cosine Similarity Hesaplama ve Öneri Fonksiyonu
def get_recommendations(user_vector, article_vectors, top_n):
    similarities = util.pytorch_cos_sim(user_vector, article_vectors).cpu().numpy()
    top_indices = similarities.argsort()  # Benzerlik değerlerine göre doğru sıralama
    top_indices = [int(idx) for idx in top_indices.flatten()]  # Her indeksin integer olduğundan emin ol
    top_indices = top_indices[-top_n:]  # Dizideki son 3 değeri al
    return top_indices


def main(xml_folder, user_interest_text, top_n=3):
    # XML Dosyalarını Yükleme
    articles = load_dataset(xml_folder)
    preprocessed_articles = [preprocess(article) for article in articles]

    # Makale ve Kullanıcı Vektörlerini Oluşturma
    article_vectors = get_sentence_embeddings(preprocessed_articles)
    user_vector = get_sentence_embeddings([user_interest_text])[0]

    # Önerileri Alma
    recommendations = get_recommendations(user_vector, article_vectors, top_n)

    # Önerilen makalelerin metinlerini yazdırma
    print("Recommended Articles:")
    for idx in recommendations:  # Sadece en yakın 3 makaleyi yazdırma
        print(f"Article {idx}:")
        print(articles[idx])
        print()

    return recommendations



# Örnek Kullanım
if __name__ == "__main__":
    # Ortamı ayarla

    # Örnek kullanım
    xml_folder = '/Users/selmanorhan/Documents/GitHub/NatureLanguageProcessing/Dataset/ake-datasets-master/datasets/500N-KPCrowd/train'
    user_interest_text = "I love fashion."
    top_n = 3

    recommendations = main(xml_folder, user_interest_text, top_n)
    print(f"Recommended article indices: {recommendations}")
