"""
CLI script: analyze all receipt images in a directory and print results.
Usage: python scripts/batch_analyze.py --input_dir data/sroie/SROIE2019/train/img
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.pipeline import run_receipt_analysis_pipeline, ReceiptProcessingError

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}


def collect_image_paths(input_dir):
    paths = []
    for filename in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(input_dir, filename))
    return paths


def serialize_result(result):
    return {
        'receipt_id': result.receipt_id,
        'fields': result.fields.model_dump(),
        'category': result.category,
        'category_confidence': result.category_confidence,
        'anomaly': result.anomaly.model_dump(),
        'processed_at': result.processed_at.isoformat(),
    }


def run_batch_analysis(input_dir, output_file):
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        print('No images found in {}'.format(input_dir))
        return

    results = []
    historical_totals = []

    for i, path in enumerate(image_paths, 1):
        filename = os.path.basename(path)
        print('[{}/{}] Processing {} ...'.format(i, len(image_paths), filename))
        try:
            result = run_receipt_analysis_pipeline(path, historical_totals=historical_totals)
            if result.fields.total is not None:
                historical_totals.append(result.fields.total)
            results.append(serialize_result(result))
        except ReceiptProcessingError as exc:
            print('  SKIP {}: {}'.format(filename, exc))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nDone. {} receipts written to {}'.format(len(results), output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch analyze receipt images.')
    parser.add_argument('--input_dir', required=True, help='Directory containing receipt images.')
    parser.add_argument('--output_file', default='data/processed/results.json')
    args = parser.parse_args()
    run_batch_analysis(args.input_dir, args.output_file)
