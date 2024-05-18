import os, re
import xml.etree.ElementTree as ET
from django.core.management.base import BaseCommand
from rentacar.models import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from summa import keywords as summa_keywords, summarizer
from datetime import datetime

# XML Dosyalarını Okuma Fonksiyonuk
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

def extract_title(content):
    # İlk özetleme
    first_summary = summarizer.summarize(content, ratio=0.3)  # Orijinal metnin %10'u kadarını özetle
    title = first_summary.strip().split('\n')[0]  # İlk cümleyi başlık olarak al
    if title:
        return title
    return "Untitled"

def load_dataset(xml_folder):
    articles = []
    count = 0
    for filename in os.listdir(xml_folder):
        print("buldum ve ekliyorum : ", count)
        count=count+1
        if filename.endswith('.xml'):
            xml_file = os.path.join(xml_folder, filename)
            text = extract_text_from_xml(xml_file)
            articles.append({
                "content": text,
                "keywords": extract_keywords(text),
                "title": extract_title(text),
                "date": extract_publication_date(text)
            })
    return articles

def extract_keywords(content):
    keyword_list = summa_keywords.keywords(content).split('\n')
    return ', '.join(keyword_list[:10])  # İlk 10 anahtar kelimeyi alın
    # vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    # X = vectorizer.fit_transform([content])
    # keywords = vectorizer.get_feature_names_out()
    # return ', '.join(keywords)

def extract_publication_date(content):
    match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
    if match:
        date_str = match.group(1)
        try:
            # Yıl, ay ve gün değerlerini kontrol etmek
            year, month, day = map(int, date_str.split('-'))
            datetime(year, month, day)  # Bu, tarihin geçerli olup olmadığını kontrol eder
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            pass
    return datetime.now().date()

class Command(BaseCommand):
    help = 'Load articles from XML dataset'

    def handle(self, *args, **kwargs):
        xml_folder = 'Dataset/500N-KPCrowd/train'  # Bu yolu kendi XML dosyalarınızın bulunduğu dizin ile değiştirin
        articles_data = load_dataset(xml_folder)
        
        for article_data in articles_data:
            article = Article(
                text=article_data["content"],
                keywords=article_data["keywords"],
                heading=article_data["title"],
                published_date=article_data["date"]
            )
            article.save()
        
        self.stdout.write(self.style.SUCCESS('Successfully loaded articles'))