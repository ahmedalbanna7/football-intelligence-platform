# app/models/match_video.py

from sqlalchemy import ForeignKey
from sqlalchemy import String

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class MatchVideo(Base):

    __tablename__ = "match_videos"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id")
    )

    object_name: Mapped[str] = mapped_column(
        String(500)
    )

    match = relationship(
        "Match",
        back_populates="videos"
    )