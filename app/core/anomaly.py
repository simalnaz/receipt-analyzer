import re
import numpy as np
from app.models.schemas import AnomalyReport

DEFAULT_THRESHOLDS = {
    'max_amount': 10000.0,
    'min_amount': 0.01,
    'garbage_ratio': 0.15,
    'zscore_threshold': 3.0,
    'min_history_for_stats': 5,
    'anomaly_confidence_per_type': 0.3,
}

GARBAGE_CHAR_PATTERN = re.compile(r'[^\x20-\x7E\n\r\t]')


def detect_amount_out_of_range(total, thresholds=None):
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if total is None:
        return False
    return total > thresholds['max_amount'] or total < thresholds['min_amount']


def detect_ocr_garbage_characters(text, thresholds=None):
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if not text:
        return False
    garbage_count = len(GARBAGE_CHAR_PATTERN.findall(text))
    return (garbage_count / len(text)) > thresholds['garbage_ratio']


def detect_missing_critical_fields(company, date, total):
    missing_count = 0
    if company is None:
        missing_count += 1
    if date is None:
        missing_count += 1
    if total is None:
        missing_count += 1
    return missing_count >= 2


def detect_statistical_amount_outlier(total, historical_totals, thresholds=None):
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if len(historical_totals) < thresholds['min_history_for_stats']:
        return False
    arr = np.array(historical_totals)
    mean = np.mean(arr)
    std = np.std(arr)
    if np.isclose(std, 0.0):
        return False
    z_score = abs(total - mean) / std
    return bool(z_score > thresholds['zscore_threshold'])


def detect_duplicate_total_values(text, total):
    if total is None:
        return False
    formatted = '{:.2f}'.format(total)
    pattern = r'(?:RM|MYR|USD|\$)?\s*' + re.escape(formatted)
    matches = re.findall(pattern, text)
    return len(matches) > 3


def detect_receipt_anomalies(raw_text, company, date, total, historical_totals=None, thresholds=None):
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    anomaly_types = []

    if detect_ocr_garbage_characters(raw_text, thresholds):
        anomaly_types.append('ocr_garbage_characters')

    if detect_amount_out_of_range(total, thresholds):
        anomaly_types.append('amount_out_of_range')

    if detect_missing_critical_fields(company, date, total):
        anomaly_types.append('missing_critical_fields')

    if total is not None and historical_totals is not None:
        if detect_statistical_amount_outlier(total, historical_totals, thresholds):
            anomaly_types.append('statistical_amount_outlier')

    if detect_duplicate_total_values(raw_text, total):
        anomaly_types.append('duplicate_total_values')

    has_anomaly = len(anomaly_types) > 0
    confidence = min(1.0, len(anomaly_types) * thresholds['anomaly_confidence_per_type'])

    return AnomalyReport(
        has_anomaly=has_anomaly,
        anomaly_types=anomaly_types,
        confidence=confidence,
    )
