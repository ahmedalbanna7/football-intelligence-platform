from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class VideoUploadedEvent(BaseModel):
    event_type: str = "video.uploaded"
    match_id: int
    video_id: int
    bucket: str
    object_name: str
    filename: str
    content_type: str | None = None
    match_context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MatchAnalysisRequestedEvent(BaseModel):
    event_type: str = "match_analysis.requested"
    run_id: int
    match_id: int
    video_id: int
    bucket: str
    object_name: str
    artifact_prefix: str
    mode: str = "FULL_ANALYSIS"
    max_frames: int = 450
    start_frame: int = 0
    calibration_points: list[dict[str, float]] = Field(default_factory=list)
    match_context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class PipelineResult(BaseModel):
    status: str
    data: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
