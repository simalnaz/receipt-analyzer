"""
Evaluate regex extraction accuracy against SROIE ground truth.

Usage:
    python scripts/evaluate_extraction.py
    python scripts/evaluate_extraction.py --split train
    python scripts/evaluate_extraction.py --limit 50
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.ocr import extract_text_and_confidence
from app.core.extractor import extract_receipt_fields
from scripts.scoring import FIELDS, load_ground_truth, score_field

SROIE_ROOT = 'data/sroie/SROIE2019'
OUTPUT_FILE = 'data/processed/extraction_accuracy.json'


def print_results_table(exact_acc, f1_acc, n_receipts):
    print('\nExtraction Results on {} receipts'.format(n_receipts))
    print('-' * 50)
    print('{:<12} {:>10} {:>12} {:>10}'.format('Field', 'Exact', 'Token F1', 'Correct'))
    print('-' * 50)
    for field in FIELDS:
        exact = exact_acc[field]['correct'] / exact_acc[field]['total'] if exact_acc[field]['total'] else 0
        f1 = f1_acc[field]['total_f1'] / f1_acc[field]['count'] if f1_acc[field]['count'] else 0
        correct = exact_acc[field]['correct']
        total = exact_acc[field]['total']
        print('{:<12} {:>9.1%} {:>11.1%} {:>7}/{}'.format(field, exact, f1, correct, total))
    print()


def run_evaluation(split, limit):
    img_dir = os.path.join(SROIE_ROOT, split, 'img')
    entities_dir = os.path.join(SROIE_ROOT, split, 'entities')

    if not os.path.exists(img_dir):
        print('ERROR: {} not found'.format(img_dir))
        sys.exit(1)

    print('Loading ground truth from {} ...'.format(entities_dir))
    ground_truth = load_ground_truth(entities_dir)

    stems = sorted(ground_truth.keys())
    if limit is not None:
        stems = stems[:limit]

    exact_acc = {field: {'correct': 0, 'total': 0} for field in FIELDS}
    f1_acc = {field: {'total_f1': 0.0, 'count': 0} for field in FIELDS}
    per_receipt = []
    errors = []

    for i, stem in enumerate(stems, 1):
        image_path = os.path.join(img_dir, stem + '.jpg')
        if not os.path.exists(image_path):
            errors.append({'stem': stem, 'error': 'image not found'})
            continue

        print('[{}/{}] {}'.format(i, len(stems), stem), end='  ')

        try:
            raw_text, _ = extract_text_and_confidence(image_path)
            fields_obj = extract_receipt_fields(raw_text)
            extracted = {
                'company': fields_obj.company,
                'address': fields_obj.address,
                'date': fields_obj.date,
                'total': fields_obj.total,
            }
        except Exception as exc:
            errors.append({'stem': stem, 'error': str(exc)})
            print('ERROR: {}'.format(exc))
            continue

        gt = ground_truth[stem]
        field_results = {}

        for field in FIELDS:
            predicted = extracted.get(field)
            expected = gt.get(field)
            exact, token_f1 = score_field(field, predicted, expected)

            exact_acc[field]['correct'] += int(exact)
            exact_acc[field]['total'] += 1
            f1_acc[field]['total_f1'] += token_f1
            f1_acc[field]['count'] += 1

            field_results[field] = {
                'predicted': predicted,
                'expected': expected,
                'exact_match': exact,
                'token_f1': round(token_f1, 4),
            }

        matches = sum(1 for f in FIELDS if field_results[f]['exact_match'])
        avg_f1 = sum(field_results[f]['token_f1'] for f in FIELDS) / len(FIELDS)
        print('{}/{} exact  avg_f1={:.0%}'.format(matches, len(FIELDS), avg_f1))
        per_receipt.append({'stem': stem, 'fields': field_results})

    print_results_table(exact_acc, f1_acc, len(per_receipt))

    summary = {}
    for field in FIELDS:
        total = exact_acc[field]['total']
        correct = exact_acc[field]['correct']
        avg_f1 = f1_acc[field]['total_f1'] / f1_acc[field]['count'] if f1_acc[field]['count'] else 0
        summary[field] = {
            'correct': correct,
            'total': total,
            'exact_accuracy': round(correct / total, 4) if total > 0 else 0.0,
            'avg_token_f1': round(avg_f1, 4),
        }

    output = {
        'split': split,
        'n_evaluated': len(per_receipt),
        'n_errors': len(errors),
        'summary': summary,
        'per_receipt': per_receipt,
        'errors': errors,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print('Full results saved to {}'.format(OUTPUT_FILE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate extraction accuracy on SROIE.')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--limit', type=int, default=None, help='Max receipts to evaluate.')
    args = parser.parse_args()
    run_evaluation(args.split, args.limit)
