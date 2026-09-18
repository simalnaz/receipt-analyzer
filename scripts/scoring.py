import json
import os

FIELDS = ['company', 'date', 'address', 'total']


def load_ground_truth(entities_dir):
    ground_truth = {}
    for filename in sorted(os.listdir(entities_dir)):
        if not filename.endswith('.txt'):
            continue
        stem = os.path.splitext(filename)[0]
        filepath = os.path.join(entities_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                ground_truth[stem] = json.load(f)
            except json.JSONDecodeError:
                continue
    return ground_truth


def normalize_total(value):
    if value is None:
        return None
    cleaned = str(value).upper()
    for prefix in ['RM', 'MYR', 'USD', '$']:
        cleaned = cleaned.replace(prefix, '')
    cleaned = cleaned.strip()
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def normalize_string(value):
    if value is None:
        return ''
    return str(value).strip().upper()


def normalize_field(field_name, value):
    if field_name == 'total':
        normalized = normalize_total(value)
        if normalized is None:
            return ''
        return '{:.2f}'.format(normalized)
    return normalize_string(value)


def compute_token_f1(predicted_str, expected_str):
    pred_tokens = set(predicted_str.split())
    exp_tokens = set(expected_str.split())
    if not pred_tokens and not exp_tokens:
        return 1.0
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_field(field_name, predicted, expected):
    norm_pred = normalize_field(field_name, predicted)
    norm_exp = normalize_field(field_name, expected)
    exact = norm_pred == norm_exp
    if field_name == 'total':
        token_f1 = 1.0 if exact else 0.0
    else:
        token_f1 = compute_token_f1(norm_pred, norm_exp)
    return exact, token_f1
