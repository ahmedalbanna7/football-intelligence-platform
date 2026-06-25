from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class VideoUploadedEvent(BaseModel):
    event_type: str = "video.uploaded"
    match_id: int
    video_id: int
    bucket: str
    object_name: str
    filename: str
    content_type: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PipelineResult(BaseModel):
    status: str
    data: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
