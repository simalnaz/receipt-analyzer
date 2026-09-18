"""
Box-File Baseline Comparison

Measures how much accuracy is lost due to OCR noise by comparing two pipelines:
  Pipeline A: Tesseract OCR -> regex extraction (results from extraction_accuracy.json)
  Pipeline B: SROIE box file text -> same regex extraction (clean annotated text)

The delta reveals: "How much of our accuracy gap comes from OCR noise vs. regex logic?"

Usage:
    python scripts/compare_box_vs_tesseract.py
    python scripts/compare_box_vs_tesseract.py --split train
    python scripts/compare_box_vs_tesseract.py --n_limit 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.extractor import extract_receipt_fields
from scripts.scoring import FIELDS, load_ground_truth, score_field

SROIE_ROOT = 'data/sroie/SROIE2019'
TESSERACT_RESULTS_FILE = 'data/processed/extraction_accuracy.json'
OUTPUT_FILE = 'data/processed/box_vs_tesseract_comparison.json'

DEFAULT_CONFIG = {
    'sroie_dir': SROIE_ROOT,
    'tesseract_results_file': TESSERACT_RESULTS_FILE,
    'output_file': OUTPUT_FILE,
    'line_group_threshold': 15,
}


def load_tesseract_results(results_file):
    if not os.path.exists(results_file):
        return {}
    with open(results_file, 'r') as f:
        data = json.load(f)
    results = {}
    for item in data.get('per_receipt', []):
        stem = item.get('stem')
        if stem is not None:
            results[stem] = item.get('fields', {})
    return results


def parse_box_file(box_path):
    boxes = []
    with open(box_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 8)
            if len(parts) < 9:
                continue
            try:
                x1 = int(parts[0])
                y1 = int(parts[1])
            except ValueError:
                continue
            text = parts[8].strip()
            if text:
                boxes.append({'x1': x1, 'y1': y1, 'text': text})
    return boxes


def reconstruct_text_from_boxes(boxes, line_threshold):
    if not boxes:
        return ''

    sorted_boxes = sorted(boxes, key=lambda b: (b['y1'], b['x1']))

    lines = []
    current_line = [sorted_boxes[0]]

    for i in range(1, len(sorted_boxes)):
        box = sorted_boxes[i]
        current_line_y = current_line[0]['y1']
        if abs(box['y1'] - current_line_y) <= line_threshold:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    lines.append(current_line)

    text_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda b: b['x1'])
        text_lines.append(' '.join(b['text'] for b in line_sorted))

    return '\n'.join(text_lines)


def run_box_pipeline(box_path, line_threshold):
    boxes = parse_box_file(box_path)
    text = reconstruct_text_from_boxes(boxes, line_threshold)
    fields_obj = extract_receipt_fields(text)
    return {
        'company': fields_obj.company,
        'address': fields_obj.address,
        'date': fields_obj.date,
        'total': fields_obj.total,
    }


def print_comparison_table(tess_exact_acc, tess_f1_acc, box_exact_acc, box_f1_acc, n):
    print('\nComparison: Tesseract Pipeline vs Box-Text Baseline ({} receipts)'.format(n))
    print('-' * 82)
    print('{:<10} {:>11} {:>11} {:>8}  {:>11} {:>11} {:>8}'.format(
        'Field', 'Tess Exact', 'Box Exact', 'Delta', 'Tess F1', 'Box F1', 'Delta'))
    print('-' * 82)
    for field in FIELDS:
        tess_total = tess_exact_acc[field]['total']
        box_total = box_exact_acc[field]['total']
        tess_e = tess_exact_acc[field]['correct'] / tess_total if tess_total > 0 else 0.0
        box_e = box_exact_acc[field]['correct'] / box_total if box_total > 0 else 0.0
        tess_f = tess_f1_acc[field]['total_f1'] / tess_f1_acc[field]['count'] if tess_f1_acc[field]['count'] > 0 else 0.0
        box_f = box_f1_acc[field]['total_f1'] / box_f1_acc[field]['count'] if box_f1_acc[field]['count'] > 0 else 0.0
        delta_e = box_e - tess_e
        delta_f = box_f - tess_f
        sign_e = '+' if delta_e >= 0 else ''
        sign_f = '+' if delta_f >= 0 else ''
        print('{:<10} {:>10.1%} {:>10.1%} {:>8}  {:>10.1%} {:>10.1%} {:>8}'.format(
            field, tess_e, box_e,
            '{}{:.1%}'.format(sign_e, delta_e),
            tess_f, box_f,
            '{}{:.1%}'.format(sign_f, delta_f)))
    print()


def run_comparison(split, n_limit):
    box_dir = os.path.join(DEFAULT_CONFIG['sroie_dir'], split, 'box')
    entities_dir = os.path.join(DEFAULT_CONFIG['sroie_dir'], split, 'entities')

    if not os.path.exists(box_dir):
        print('ERROR: box directory not found: {}'.format(box_dir))
        sys.exit(1)

    print('Loading ground truth ...')
    ground_truth = load_ground_truth(entities_dir)

    print('Loading existing Tesseract results from {} ...'.format(
        DEFAULT_CONFIG['tesseract_results_file']))
    tesseract_data = load_tesseract_results(DEFAULT_CONFIG['tesseract_results_file'])
    if not tesseract_data:
        print('WARNING: Tesseract results not found. Run evaluate_extraction.py first.')
        print('         Tesseract columns will show 0.0.')

    stems = sorted(ground_truth.keys())
    if n_limit is not None:
        stems = stems[:n_limit]

    print('Evaluating {} receipts (split={}) ...\n'.format(len(stems), split))

    box_exact_acc = {field: {'correct': 0, 'total': 0} for field in FIELDS}
    box_f1_acc = {field: {'total_f1': 0.0, 'count': 0} for field in FIELDS}
    tess_exact_acc = {field: {'correct': 0, 'total': 0} for field in FIELDS}
    tess_f1_acc = {field: {'total_f1': 0.0, 'count': 0} for field in FIELDS}

    per_receipt = []
    errors = []

    for i, stem in enumerate(stems, 1):
        box_path = os.path.join(box_dir, stem + '.txt')
        if not os.path.exists(box_path):
            errors.append({'stem': stem, 'error': 'box file not found'})
            continue

        print('[{}/{}] {}'.format(i, len(stems), stem), end='  ')

        try:
            box_extracted = run_box_pipeline(box_path, DEFAULT_CONFIG['line_group_threshold'])
        except Exception as exc:
            errors.append({'stem': stem, 'error': str(exc)})
            print('BOX ERROR: {}'.format(exc))
            continue

        gt = ground_truth[stem]
        tess_fields = tesseract_data.get(stem, {})
        receipt_result = {'stem': stem, 'fields': {}}

        for field in FIELDS:
            box_pred = box_extracted.get(field)
            box_exact, box_f1 = score_field(field, box_pred, gt.get(field))

            box_exact_acc[field]['correct'] += int(box_exact)
            box_exact_acc[field]['total'] += 1
            box_f1_acc[field]['total_f1'] += box_f1
            box_f1_acc[field]['count'] += 1

            tess_field_data = tess_fields.get(field, {})
            tess_exact = tess_field_data.get('exact_match', False)
            tess_f1 = tess_field_data.get('token_f1', 0.0)
            tess_pred = tess_field_data.get('predicted', None)

            if tess_field_data:
                tess_exact_acc[field]['correct'] += int(tess_exact)
                tess_exact_acc[field]['total'] += 1
                tess_f1_acc[field]['total_f1'] += tess_f1
                tess_f1_acc[field]['count'] += 1

            receipt_result['fields'][field] = {
                'box_predicted': box_pred,
                'tesseract_predicted': tess_pred,
                'expected': gt.get(field),
                'box_exact': box_exact,
                'tesseract_exact': tess_exact,
                'box_f1': round(box_f1, 4),
                'tesseract_f1': round(tess_f1, 4),
            }

        box_matches = sum(1 for f in FIELDS if receipt_result['fields'][f]['box_exact'])
        print('box={}/4'.format(box_matches))
        per_receipt.append(receipt_result)

    print_comparison_table(tess_exact_acc, tess_f1_acc, box_exact_acc, box_f1_acc, len(per_receipt))

    fields_summary = {}
    for field in FIELDS:
        tess_total = tess_exact_acc[field]['total']
        tess_correct = tess_exact_acc[field]['correct']
        tess_avg_e = round(tess_correct / tess_total, 4) if tess_total > 0 else 0.0
        tess_avg_f = round(tess_f1_acc[field]['total_f1'] / tess_f1_acc[field]['count'], 4) if tess_f1_acc[field]['count'] > 0 else 0.0

        box_total = box_exact_acc[field]['total']
        box_correct = box_exact_acc[field]['correct']
        box_avg_e = round(box_correct / box_total, 4) if box_total > 0 else 0.0
        box_avg_f = round(box_f1_acc[field]['total_f1'] / box_f1_acc[field]['count'], 4) if box_f1_acc[field]['count'] > 0 else 0.0

        fields_summary[field] = {
            'tesseract_exact': tess_avg_e,
            'box_exact': box_avg_e,
            'delta_exact': round(box_avg_e - tess_avg_e, 4),
            'tesseract_f1': tess_avg_f,
            'box_f1': box_avg_f,
            'delta_f1': round(box_avg_f - tess_avg_f, 4),
        }

    output = {
        'split': split,
        'n_evaluated': len(per_receipt),
        'n_errors': len(errors),
        'fields': fields_summary,
        'per_receipt': per_receipt,
        'errors': errors,
    }

    os.makedirs(os.path.dirname(DEFAULT_CONFIG['output_file']), exist_ok=True)
    with open(DEFAULT_CONFIG['output_file'], 'w') as f:
        json.dump(output, f, indent=2)
    print('Full results saved to {}'.format(DEFAULT_CONFIG['output_file']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Tesseract pipeline vs box-file baseline on SROIE.')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--n_limit', type=int, default=None,
                        help='Max receipts to evaluate (default: all).')
    args = parser.parse_args()
    run_comparison(args.split, args.n_limit)
