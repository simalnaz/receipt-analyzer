from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.db.database import initialize_database


@asynccontextmanager
async def lifespan(app):
    initialize_database()
    yield


app = FastAPI(
    title='Receipt Analyzer API',
    description='Extract fields, classify spending, and detect anomalies from receipt images.',
    version='0.1.0',
    lifespan=lifespan,
)

app.include_router(router)


@app.get('/health')
def health_check():
    return {'status': 'ok'}
