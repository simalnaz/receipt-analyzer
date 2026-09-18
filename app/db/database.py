import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./receipts.db')

engine = create_engine(
    DATABASE_URL,
    connect_args={'check_same_thread': False} if 'sqlite' in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def initialize_database():
    Base.metadata.create_all(bind=engine)


def open_db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
