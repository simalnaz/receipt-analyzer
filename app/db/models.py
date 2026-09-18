from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Receipt(Base):
    __tablename__ = "receipts"

    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String(255), nullable=False)
    raw_text = Column(Text, nullable=False)
    company = Column(String(255), nullable=True)
    address = Column(Text, nullable=True)
    date = Column(String(50), nullable=True)
    total = Column(Float, nullable=True)
    category = Column(String(100), nullable=True)
    category_confidence = Column(Float, default=0.0)
    has_anomaly = Column(Boolean, default=False)
    anomaly_types = Column(Text, default="")
    processed_at = Column(DateTime, default=datetime.utcnow)
