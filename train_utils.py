"""Text cleaning shared by training and prediction."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))
STEM = PorterStemmer()


def clean(text):
    """Lowercase, keep words only, drop stopwords, stem."""
    text = re.sub(r"\d{5,}", " longnumber ", text.lower())  # phone numbers / shortcodes are spam signals
    words = re.findall(r"[a-z]+", text)
    return " ".join(STEM.stem(w) for w in words if w not in STOP)
