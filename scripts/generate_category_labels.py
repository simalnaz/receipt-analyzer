"""
Generate weak category labels for SROIE training receipts using keyword heuristic,
then train and evaluate the TF-IDF/LR classifier.

Usage:
    python scripts/generate_category_labels.py
    python scripts/generate_category_labels.py --min_confidence 0.1
    python scripts/generate_category_labels.py --limit 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.ocr import extract_text_and_confidence
from app.core.classifier import classify_by_keywords, train_classifier

SROIE_TRAIN_IMG = 'data/sroie/SROIE2019/train/img'
SROIE_TRAIN_ENTITIES = 'data/sroie/SROIE2019/train/entities'
LABELS_FILE = 'data/processed/category_labels.json'
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def load_company_names(entities_dir):
    companies = {}
    for filename in sorted(os.listdir(entities_dir)):
        if not filename.endswith('.txt'):
            continue
        stem = os.path.splitext(filename)[0]
        filepath = os.path.join(entities_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                companies[stem] = data.get('company', '')
            except json.JSONDecodeError:
                companies[stem] = ''
    return companies


def collect_image_paths(img_dir):
    paths = {}
    for filename in os.listdir(img_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            stem = os.path.splitext(filename)[0]
            paths[stem] = os.path.join(img_dir, filename)
    return paths


def run_label_generation_and_training(min_confidence, limit=None):
    print('Loading company names from ground truth ...')
    companies = load_company_names(SROIE_TRAIN_ENTITIES)

    print('Collecting image paths ...')
    image_paths = collect_image_paths(SROIE_TRAIN_IMG)

    stems = sorted(set(companies.keys()) & set(image_paths.keys()))
    if limit is not None:
        stems = stems[:limit]
    print('Found {} receipts with both image and ground truth.'.format(len(stems)))

    texts = []
    labels = []
    skipped_low_confidence = 0
    skipped_ocr_error = 0
    label_counts = {}

    for i, stem in enumerate(stems, 1):
        image_path = image_paths[stem]
        company = companies[stem]

        print('[{}/{}] {} ({})'.format(i, len(stems), stem, company or 'no company'))

        try:
            raw_text, _ = extract_text_and_confidence(image_path)
        except Exception as exc:
            print('  OCR ERROR: {}'.format(exc))
            skipped_ocr_error += 1
            continue

        combined_text = company + ' ' + raw_text if company else raw_text
        category, confidence = classify_by_keywords(combined_text)

        if confidence < min_confidence and category != 'other':
            skipped_low_confidence += 1
            continue

        texts.append(combined_text)
        labels.append(category)
        label_counts[category] = label_counts.get(category, 0) + 1

    print('\nLabel distribution:')
    for cat, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print('  {:<15} {}'.format(cat, count))

    print('\nSkipped: {} low confidence, {} OCR errors'.format(
        skipped_low_confidence, skipped_ocr_error
    ))

    labeled_data = [
        {'text': text, 'label': label}
        for text, label in zip(texts, labels)
    ]
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    with open(LABELS_FILE, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    print('\nLabels saved to {}'.format(LABELS_FILE))

    if len(set(labels)) < 2:
        print('ERROR: need at least 2 distinct categories to train. Got: {}'.format(set(labels)))
        sys.exit(1)

    model_path = 'data/category_model.pkl'
    print('\nTraining TF-IDF + LogisticRegression classifier ...')
    train_classifier(texts, labels, model_path=model_path)

    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from app.core.classifier import load_classifier
    import numpy as np

    model = load_classifier(model_path)
    scores = cross_val_score(model, texts, labels, cv=5, scoring='accuracy')
    report = {
        'model_path': model_path,
        'cv_accuracy_mean': round(float(np.mean(scores)), 4),
        'cv_accuracy_std': round(float(np.std(scores)), 4),
        'n_samples': len(texts),
        'n_classes': len(set(labels)),
    }

    print('\nTraining complete.')
    print('  Samples:          {}'.format(report['n_samples']))
    print('  Classes:          {}'.format(report['n_classes']))
    print('  CV accuracy mean: {:.1%}'.format(report['cv_accuracy_mean']))
    print('  CV accuracy std:  {:.1%}'.format(report['cv_accuracy_std']))
    print('  Model saved to:   {}'.format(report['model_path']))

    report_path = 'data/processed/classifier_training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print('  Report saved to:  {}'.format(report_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate category labels and train classifier.')
    parser.add_argument(
        '--min_confidence',
        type=float,
        default=0.05,
        help='Minimum keyword confidence to include a sample (default: 0.05).',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max receipts to process (default: all 626).',
    )
    args = parser.parse_args()
    run_label_generation_and_training(args.min_confidence, args.limit)
