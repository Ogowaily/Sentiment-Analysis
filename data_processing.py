
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Preprocess tweet: clean + remove stopwords + stemming
def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove urls
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'#', '', text)  # remove hashtag symbol
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # keep letters only
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces

    words = text.split()
    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)
