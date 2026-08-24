from pydantic import BaseModel, Field
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
    start_frame: int = 0
    calibration_points: list[dict[str, float]] = Field(default_factory=list)
    reuse_detections_object: str | None = None
    reuse_model_mode: str | None = None
    reuse_ball_detection_mode: str | None = None


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
            start_frame=payload.start_frame,
            calibration_points=payload.calibration_points,
            reuse_detections_object=payload.reuse_detections_object,
            reuse_model_mode=payload.reuse_model_mode,
            reuse_ball_detection_mode=payload.reuse_ball_detection_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
