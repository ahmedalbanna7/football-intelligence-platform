from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from app.match_analysis_plus.runner import MatchAnalysisPlusRunner


class MatchAnalysisRequest(BaseModel):
    run_id: int
    match_id: int
    bucket: str
    object_name: str
    artifact_prefix: str
    mode: str = "FULL_ANALYSIS"
    max_frames: int = 450


app = FastAPI(title="Match Analysis Plus Worker")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "match-analysis-worker"}


@app.post("/runs")
def run_match_analysis(payload: MatchAnalysisRequest) -> dict:
    try:
        return MatchAnalysisPlusRunner().run(
            run_id=payload.run_id,
            match_id=payload.match_id,
            bucket=payload.bucket,
            object_name=payload.object_name,
            artifact_prefix=payload.artifact_prefix,
            mode=payload.mode,
            max_frames=payload.max_frames,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
