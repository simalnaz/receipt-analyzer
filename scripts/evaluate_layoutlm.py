"""
Evaluate fine-tuned LayoutLM accuracy against SROIE ground truth.

Loads the preprocessed box-file dataset (prepare_dataset.py output) and runs
LayoutLM inference, then computes exact match and token F1 per field.

Usage:
    python scripts/evaluate_layoutlm.py
    python scripts/evaluate_layoutlm.py --split train
    python scripts/evaluate_layoutlm.py --limit 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.layoutlm_extractor import extract_fields_from_words
from scripts.scoring import FIELDS, load_ground_truth, score_field

DATASET_DIR = 'data/processed/layoutlm_dataset'
SROIE_ROOT = 'data/sroie/SROIE2019'
OUTPUT_FILE = 'data/processed/layoutlm_results.json'


def load_dataset(split):
    path = os.path.join(DATASET_DIR, '{}.json'.format(split))
    if not os.path.exists(path):
        print('ERROR: {} not found. Run prepare_dataset.py first.'.format(path))
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_results_table(exact_acc, f1_acc, n_receipts):
    print('\nLayoutLM Results on {} receipts'.format(n_receipts))
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
    entities_dir = os.path.join(SROIE_ROOT, split, 'entities')
    ground_truth_all = load_ground_truth(entities_dir)

    examples = load_dataset(split)
    if limit is not None:
        examples = examples[:limit]

    exact_acc = {field: {'correct': 0, 'total': 0} for field in FIELDS}
    f1_acc = {field: {'total_f1': 0.0, 'count': 0} for field in FIELDS}
    per_receipt = []
    errors = []

    for i, example in enumerate(examples, 1):
        stem = example['receipt_id']
        words = example['words']
        boxes = example['boxes']

        print('[{}/{}] {}'.format(i, len(examples), stem), end='  ')

        try:
            fields_obj = extract_fields_from_words(words, boxes)
            extracted = {
                'company': fields_obj.company,
                'address': fields_obj.address,
                'date': fields_obj.date,
                'total': str(fields_obj.total) if fields_obj.total is not None else None,
            }
        except Exception as exc:
            errors.append({'stem': stem, 'error': str(exc)})
            print('ERROR: {}'.format(exc))
            continue

        gt = ground_truth_all.get(stem, {})
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
    parser = argparse.ArgumentParser(description='Evaluate LayoutLM accuracy on SROIE.')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--limit', type=int, default=None, help='Max receipts to evaluate.')
    args = parser.parse_args()
    run_evaluation(args.split, args.limit)
