import json
from sqlalchemy.orm import Session
from app.db.models import Receipt
from app.models.schemas import ReceiptAnalysisResult


def save_receipt_analysis(session, result, image_filename):
    record = Receipt(
        image_filename=image_filename,
        raw_text=result.raw_text,
        company=result.fields.company,
        address=result.fields.address,
        date=result.fields.date,
        total=result.fields.total,
        category=result.category,
        category_confidence=result.category_confidence,
        has_anomaly=result.anomaly.has_anomaly,
        anomaly_types=json.dumps(result.anomaly.anomaly_types),
        processed_at=result.processed_at,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return _deserialize_anomaly_types(record)


def _deserialize_anomaly_types(record):
    raw = record.anomaly_types or '[]'
    try:
        record.anomaly_types = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        record.anomaly_types = []
    return record


def fetch_receipt_by_id(session, receipt_id):
    record = session.query(Receipt).filter(Receipt.id == receipt_id).first()
    if record is not None:
        _deserialize_anomaly_types(record)
    return record


def fetch_receipts(session, skip=0, limit=100):
    records = session.query(Receipt).offset(skip).limit(limit).all()
    for record in records:
        _deserialize_anomaly_types(record)
    return records


def fetch_receipts_by_category(session, category):
    records = session.query(Receipt).filter(Receipt.category == category).all()
    for record in records:
        _deserialize_anomaly_types(record)
    return records


def fetch_flagged_receipts(session):
    records = session.query(Receipt).filter(Receipt.has_anomaly.is_(True)).all()
    for record in records:
        _deserialize_anomaly_types(record)
    return records


def delete_receipt_by_id(session, receipt_id):
    record = session.query(Receipt).filter(Receipt.id == receipt_id).first()
    if record is None:
        return False
    session.delete(record)
    session.commit()
    return True
