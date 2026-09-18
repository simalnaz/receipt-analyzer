import os
from datetime import datetime
from app.core.ocr import extract_text_and_confidence
from app.core.extractor import extract_receipt_fields
from app.core.classifier import classify_spending_category
from app.core.anomaly import detect_receipt_anomalies
from app.models.schemas import ReceiptAnalysisResult


class ReceiptProcessingError(Exception):
    pass


def run_receipt_analysis_pipeline(image_path, historical_totals=None, use_layoutlm=False):
    try:
        raw_text, word_data = extract_text_and_confidence(image_path)
    except Exception as exc:
        raise ReceiptProcessingError('OCR failed for {}: {}'.format(image_path, exc)) from exc

    if raw_text.strip() == '':
        raise ReceiptProcessingError('OCR returned no text for {}'.format(image_path))

    try:
        if use_layoutlm:
            from app.core.layoutlm_extractor import extract_fields_with_layoutlm
            fields = extract_fields_with_layoutlm(image_path)
        else:
            fields = extract_receipt_fields(raw_text)

        category, category_confidence = classify_spending_category(raw_text)
        anomaly = detect_receipt_anomalies(
            raw_text=raw_text,
            company=fields.company,
            date=fields.date,
            total=fields.total,
            historical_totals=historical_totals,
        )
    except Exception as exc:
        raise ReceiptProcessingError('Analysis failed for {}: {}'.format(image_path, exc)) from exc

    receipt_id = int.from_bytes(os.urandom(4), 'big')

    return ReceiptAnalysisResult(
        receipt_id=receipt_id,
        raw_text=raw_text,
        fields=fields,
        category=category,
        category_confidence=category_confidence,
        anomaly=anomaly,
        processed_at=datetime.utcnow(),
    )
