from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.minio_client import create_bucket_if_not_exists
from app.modules.players.router import router as players_router
from app.modules.matches.router import router as matches_router
from app.modules.primary_team.router import router as primary_team_router
from app.modules.teams.router import router as teams_router
from app.modules.ai.router import router as ai_router
from app.modules.first_analysis.router import router as first_analysis_router
from app.modules.match_analysis_plus.router import router as match_analysis_plus_router
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    matches_router,
    prefix="/matches",
    tags=["Matches"]
)

app.include_router(
    primary_team_router,
    prefix="/primary-team",
    tags=["Primary Team"]
)

app.include_router(
    teams_router,
    prefix="/teams",
    tags=["Teams"]
)

app.include_router(
    ai_router,
    prefix="/ai",
    tags=["AI"]
)

app.include_router(
    first_analysis_router,
    prefix="/first-analysis",
    tags=["First Analysis"]
)

app.include_router(
    match_analysis_plus_router,
    prefix="/match-analysis-plus",
    tags=["Match Analysis Plus"]
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

