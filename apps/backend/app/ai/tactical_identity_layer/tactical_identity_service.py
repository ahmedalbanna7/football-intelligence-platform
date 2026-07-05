from typing import Any

from app.ai.tactical_identity_layer.identity_resolver import IdentityResolver


class TacticalIdentityService:
    name = "tactical_identity_stub_v1"

    def __init__(self) -> None:
        self.resolver = IdentityResolver()

    def resolve(
        self,
        tracks: list[dict[str, Any]],
        match_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = match_context or {}
        resolutions = self.resolver.resolve(tracks, context)
        assignments = [
            resolution.model_dump()
            for resolution in resolutions
        ]

        return {
            "status": "ok",
            "data": {
                "assignments": assignments,
                "assignments_count": len(assignments),
                "resolved_count": len(
                    [
                        item
                        for item in assignments
                        if item.get("resolved_player_id") is not None
                    ]
                ),
            },
            "meta": {
                "engine": self.name,
                "inputs": [
                    "team_classification",
                    "lineup",
                    "substitutions",
                    "jersey_number_stub_or_ocr",
                    "future_pitch_position",
                ],
                "future_inputs": [
                    "homography",
                    "visual_reid",
                    "face_recognition",
                    "manual_corrections",
                ],
            },
        }
