import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from app.core.pipeline import run_receipt_analysis_pipeline, ReceiptProcessingError
from app.db.database import open_db_session
from app.db.models import Receipt
from app.db.repository import (
    save_receipt_analysis,
    fetch_receipt_by_id,
    fetch_receipts,
    fetch_receipts_by_category,
    fetch_flagged_receipts,
    delete_receipt_by_id,
)
from app.models.schemas import ReceiptRecord

router = APIRouter(prefix='/receipts', tags=['receipts'])

UPLOAD_DIR = 'data/uploads'
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

os.makedirs(UPLOAD_DIR, exist_ok=True)


def _safe_filename(original_filename):
    ext = os.path.splitext(original_filename)[1].lower()
    return uuid.uuid4().hex + ext


def _fetch_historical_totals(session):
    rows = session.query(Receipt.total).filter(Receipt.total != None).all()
    return [row[0] for row in rows]


@router.post('/analyze', response_model=ReceiptRecord)
def analyze_uploaded_receipt(
    file: UploadFile = File(...),
    session: Session = Depends(open_db_session),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail='Unsupported file type. Allowed: jpg, jpeg, png, bmp')
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Uploaded file must be an image.')

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='File exceeds 10 MB limit.')

    safe_name = _safe_filename(file.filename)
    image_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(image_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    historical_totals = _fetch_historical_totals(session)

    try:
        result = run_receipt_analysis_pipeline(image_path, historical_totals=historical_totals)
        return save_receipt_analysis(session, result, file.filename)
    except ReceiptProcessingError as exc:
        os.remove(image_path)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        os.remove(image_path)
        raise


@router.get('/filter/anomalies', response_model=list[ReceiptRecord])
def list_flagged_receipts(session: Session = Depends(open_db_session)):
    return fetch_flagged_receipts(session)


@router.get('/filter/category/{category}', response_model=list[ReceiptRecord])
def list_receipts_by_category(category: str, session: Session = Depends(open_db_session)):
    return fetch_receipts_by_category(session, category)


@router.get('/{receipt_id}', response_model=ReceiptRecord)
def get_receipt_by_id(receipt_id: int, session: Session = Depends(open_db_session)):
    record = fetch_receipt_by_id(session, receipt_id)
    if record is None:
        raise HTTPException(status_code=404, detail='Receipt {} not found.'.format(receipt_id))
    return record


@router.get('/', response_model=list[ReceiptRecord])
def list_receipts(skip: int = 0, limit: int = Query(default=100, ge=1, le=1000), session: Session = Depends(open_db_session)):
    return fetch_receipts(session, skip=skip, limit=limit)


@router.delete('/{receipt_id}')
def remove_receipt_by_id(receipt_id: int, session: Session = Depends(open_db_session)):
    success = delete_receipt_by_id(session, receipt_id)
    if not success:
        raise HTTPException(status_code=404, detail='Receipt {} not found.'.format(receipt_id))
    return {'message': 'Receipt {} deleted.'.format(receipt_id)}
