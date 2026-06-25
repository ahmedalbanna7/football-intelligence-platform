from fastapi import FastAPI
from app.services.minio_client import create_bucket_if_not_exists
from app.modules.players.router import router as players_router
from app.modules.matches.router import router as matches_router
from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import Optional

RABBITMQ_URL: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_bucket_if_not_exists()
    yield

app = FastAPI(
    title="Sports Intelligence Platform",
    lifespan=lifespan
)


app.include_router(
    matches_router,
    prefix="/matches",
    tags=["Matches"]
)





app.include_router(
    players_router,
    prefix="/players",
    tags=["Players"]
)

@app.get("/")
def health():
    return {
        "status": "healthy"
    }

