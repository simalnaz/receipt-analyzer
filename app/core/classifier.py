import logging
import os
import pickle
import numpy as np

logger = logging.getLogger(__name__)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_PATH = 'data/category_model.pkl'

DEFAULT_TFIDF_CONFIG = {
    'max_features': 500,
    'ngram_range': (1, 2),
    'stop_words': 'english',
}

DEFAULT_CLASSIFIER_CONFIG = {
    'C': 1.0,
    'max_iter': 500,
}

CATEGORY_KEYWORDS = {
    'grocery': ['supermarket', 'market', 'grocery', 'mart', 'fresh', 'vegetables', 'fruits', 'dairy'],
    'restaurant': ['restaurant', 'cafe', 'coffee', 'food', 'kitchen', 'dining', 'bistro', 'eatery', 'pizza', 'burger'],
    'fuel': ['petrol', 'fuel', 'gas', 'petroleum', 'shell', 'petronas', 'station', 'diesel'],
    'pharmacy': ['pharmacy', 'chemist', 'clinic', 'medical', 'health', 'drug', 'guardian', 'watson'],
    'electronics': ['electronics', 'computer', 'phone', 'digital', 'gadget', 'tech', 'mobile'],
    'clothing': ['fashion', 'apparel', 'clothing', 'boutique', 'wear', 'textile'],
    'utilities': ['electricity', 'water', 'telco', 'internet', 'broadband', 'utility', 'bill'],
    'entertainment': ['cinema', 'movie', 'entertainment', 'games', 'sport', 'gym'],
}


def build_keyword_feature_vector(text):
    lower_text = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        if not keywords:
            scores[category] = 0.0
            continue
        hit_count = 0
        for kw in keywords:
            if kw in lower_text:
                hit_count += 1
        scores[category] = hit_count / len(keywords)
    return scores


def classify_by_keywords(text):
    scores = build_keyword_feature_vector(text)
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    if best_score == 0.0:
        return 'other', 0.0
    return best_category, round(best_score, 4)


def train_classifier(texts, labels, model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=DEFAULT_TFIDF_CONFIG['max_features'],
            ngram_range=DEFAULT_TFIDF_CONFIG['ngram_range'],
            stop_words=DEFAULT_TFIDF_CONFIG['stop_words'],
        )),
        ('classifier', LogisticRegression(
            C=DEFAULT_CLASSIFIER_CONFIG['C'],
            max_iter=DEFAULT_CLASSIFIER_CONFIG['max_iter'],
        )),
    ])
    pipeline.fit(texts, labels)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline


def load_classifier(model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning('failed to load classifier from %s: %s', model_path, exc)
        return None


def classify_spending_category(text, model_path=None):
    model = load_classifier(model_path)
    if model is not None:
        predicted = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = float(np.max(probabilities))
        return predicted, round(confidence, 4)
    return classify_by_keywords(text)
