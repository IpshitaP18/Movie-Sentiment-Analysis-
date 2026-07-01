import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Lower case
    text = text.lower()
    # HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Numbers
    text = re.sub(r"\d+", "", text)
    # Tokenization
    words = text.split()

    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)