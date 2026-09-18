from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime


class ReceiptFields(BaseModel):
    company: str | None = None
    address: str | None = None
    date: str | None = None
    total: float | None = None

    @field_validator('total')
    @classmethod
    def total_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('total cannot be negative')
        return v


class AnomalyReport(BaseModel):
    has_anomaly: bool
    anomaly_types: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ReceiptAnalysisResult(BaseModel):
    receipt_id: int
    raw_text: str
    fields: ReceiptFields
    category: str | None = None
    category_confidence: float = 0.0
    anomaly: AnomalyReport
    processed_at: datetime


class ReceiptRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_filename: str
    raw_text: str
    company: str | None
    address: str | None
    date: str | None
    total: float | None
    category: str | None
    category_confidence: float
    has_anomaly: bool
    anomaly_types: list[str]
    processed_at: datetime
