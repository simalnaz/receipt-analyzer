"""
Prepare BIO-labeled dataset for LayoutLM fine-tuning.

Reads SROIE box files and ground truth entities, assigns BIO labels to each word,
normalizes bounding boxes to [0, 1000], and saves train.json + test.json.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --limit 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from scripts.scoring import load_ground_truth

SROIE_ROOT = 'data/sroie/SROIE2019'
OUTPUT_DIR = 'data/processed/layoutlm_dataset'

LABEL2ID = {
    'O': 0,
    'B-COMPANY': 1, 'I-COMPANY': 2,
    'B-DATE': 3,    'I-DATE': 4,
    'B-ADDRESS': 5, 'I-ADDRESS': 6,
    'B-TOTAL': 7,   'I-TOTAL': 8,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

FIELD_TO_LABEL_PREFIX = {
    'company': 'COMPANY',
    'date':    'DATE',
    'address': 'ADDRESS',
    'total':   'TOTAL',
}


def parse_box_file(box_path):
    """
    Parse SROIE box file into parallel lists of words and axis-aligned bounding boxes.
    Box file format per line: x1,y1,x2,y2,x3,y3,x4,y4,text  (quadrilateral corners)
    Returns: (words list, boxes list of [x_min, y_min, x_max, y_max])
    """
    words = []
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
                coords = [int(p) for p in parts[:8]]
            except ValueError:
                continue
            text = parts[8].strip()
            if not text:
                continue
            words.append(text)
            xs = [coords[0], coords[2], coords[4], coords[6]]
            ys = [coords[1], coords[3], coords[5], coords[7]]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return words, boxes


def normalize_boxes(boxes, img_width, img_height):
    """Scale [x0, y0, x1, y1] coordinates to [0, 1000] as required by LayoutLM."""
    normalized = []
    for box in boxes:
        x0 = max(0, min(1000, int(box[0] / img_width * 1000)))
        y0 = max(0, min(1000, int(box[1] / img_height * 1000)))
        x1 = max(0, min(1000, int(box[2] / img_width * 1000)))
        y1 = max(0, min(1000, int(box[3] / img_height * 1000)))
        normalized.append([x0, y0, x1, y1])
    return normalized


def find_span_indices(words, gt_value):
    """
    Find the first contiguous span in words (case-insensitive) matching gt_value tokens.
    Returns a dict {word_index: 'B' or 'I'} for the matched span, or {} if no match.
    """
    if not gt_value:
        return {}
    gt_tokens = gt_value.strip().upper().split()
    if not gt_tokens:
        return {}
    words_upper = [w.upper() for w in words]
    n = len(words_upper)
    k = len(gt_tokens)
    for start in range(n - k + 1):
        if words_upper[start:start + k] == gt_tokens:
            result = {start: 'B'}
            for j in range(1, k):
                result[start + j] = 'I'
            return result
    return {}


def assign_bio_labels(words, ground_truth):
    """
    Assign a BIO label ID to each word. Earlier fields win on overlap.
    """
    labels = [LABEL2ID['O']] * len(words)
    for field, label_prefix in FIELD_TO_LABEL_PREFIX.items():
        gt_value = ground_truth.get(field)
        span = find_span_indices(words, gt_value)
        for idx, bio in span.items():
            if labels[idx] == LABEL2ID['O']:
                labels[idx] = LABEL2ID['{}-{}'.format(bio, label_prefix)]
    return labels


def get_image_size(img_path):
    with Image.open(img_path) as img:
        return img.width, img.height


def process_split(split, limit):
    box_dir = os.path.join(SROIE_ROOT, split, 'box')
    entities_dir = os.path.join(SROIE_ROOT, split, 'entities')
    img_dir = os.path.join(SROIE_ROOT, split, 'img')

    ground_truth_all = load_ground_truth(entities_dir)
    stems = sorted(ground_truth_all.keys())
    if limit is not None:
        stems = stems[:limit]

    examples = []
    skipped = 0

    for i, stem in enumerate(stems, 1):
        box_path = os.path.join(box_dir, stem + '.txt')
        img_path = os.path.join(img_dir, stem + '.jpg')

        if not os.path.exists(box_path):
            skipped += 1
            continue

        words, boxes = parse_box_file(box_path)
        if not words:
            skipped += 1
            continue

        if os.path.exists(img_path):
            img_width, img_height = get_image_size(img_path)
        else:
            all_x = [b[2] for b in boxes]
            all_y = [b[3] for b in boxes]
            img_width = max(all_x) + 1 if all_x else 1000
            img_height = max(all_y) + 1 if all_y else 1000

        normalized = normalize_boxes(boxes, img_width, img_height)
        gt = ground_truth_all[stem]
        labels = assign_bio_labels(words, gt)

        examples.append({
            'receipt_id': stem,
            'words': words,
            'boxes': normalized,
            'labels': labels,
        })

        labeled_count = sum(1 for l in labels if l != LABEL2ID['O'])
        print('[{}/{}] {}  words={}  labeled={}'.format(
            i, len(stems), stem, len(words), labeled_count))

    print('\n{} split: {} examples processed, {} skipped'.format(
        split, len(examples), skipped))
    return examples


def main(limit):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ['train', 'test']:
        print('\n=== Processing {} split ==='.format(split))
        examples = process_split(split, limit)
        out_path = os.path.join(OUTPUT_DIR, '{}.json'.format(split))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print('Saved {} examples to {}'.format(len(examples), out_path))

    label_map_path = os.path.join(OUTPUT_DIR, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump({'label2id': LABEL2ID, 'id2label': ID2LABEL}, f, indent=2)
    print('\nLabel map saved to {}'.format(label_map_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare BIO-labeled dataset for LayoutLM.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max receipts per split to process (default: all).')
    args = parser.parse_args()
    main(args.limit)
