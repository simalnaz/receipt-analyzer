import pytest
from app.core.classifier import classify_by_keywords, build_keyword_feature_vector


def test_grocery_classification():
    text = 'AEON SUPERMARKET fresh vegetables dairy products'
    category, confidence = classify_by_keywords(text)
    assert category == 'grocery'
    assert confidence > 0


def test_restaurant_classification():
    text = 'KFC RESTAURANT chicken burger combo meal dining'
    category, confidence = classify_by_keywords(text)
    assert category == 'restaurant'
    assert confidence > 0


def test_fuel_classification():
    text = 'PETRONAS petrol station diesel fuel pump'
    category, confidence = classify_by_keywords(text)
    assert category == 'fuel'
    assert confidence > 0


def test_unknown_text_returns_other():
    category, confidence = classify_by_keywords('xyzzy bloop random gibberish')
    assert category == 'other'
    assert confidence == 0.0


def test_keyword_feature_vector_has_all_categories():
    vector = build_keyword_feature_vector('supermarket')
    assert 'grocery' in vector
    assert 'restaurant' in vector
    assert isinstance(vector['grocery'], float)
