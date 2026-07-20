from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import io
import json
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.match_analysis_run import MatchAnalysisRun
from app.models.player import Player
from app.models.tracking_quality import (
    TrackReviewCorrection,
    TrackReviewItem,
    TrackingQualityAssessment,
)
from app.services.minio_client import BUCKET_NAME, client
from app.tracking_quality.metrics import evaluate_tracking


QUALITY_THRESHOLDS = {
    "approve_identity_confidence": 0.82,
    "review_identity_confidence": 0.68,
    "high_risk_identity_confidence": 0.52,
    "high_risk_fragments": 2,
    "high_risk_raw_id_transitions": 4,
}


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class TrackingQualityService:
    correction_actions = {
        "approve",
        "reject",
        "merge",
        "split",
        "assign_player",
        "change_team",
    }

    def sync_from_summary(
        self,
        db: Session,
        run: MatchAnalysisRun,
        summary: dict[str, Any],
    ) -> TrackingQualityAssessment:
        quality = summary.get("tracking_quality") or self._quality_from_legacy_summary(summary)
        overview = quality.get("overview", {})
        tracker_runtime = quality.get("tracker_runtime", {})
        reid = tracker_runtime.get("reid", {})
        assessment = (
            db.query(TrackingQualityAssessment)
            .filter(TrackingQualityAssessment.run_id == run.id)
            .first()
        )
        if assessment is None:
            assessment = TrackingQualityAssessment(run_id=run.id)
            db.add(assessment)

        assessment.status = str(overview.get("status", "needs_review"))
        assessment.tracker_engine = str(
            tracker_runtime.get("engine") or quality.get("engine") or "unknown"
        )
        assessment.reid_enabled = bool(reid.get("active", False))
        assessment.reid_model = str(reid.get("model")) if reid.get("model") is not None else None
        assessment.average_identity_confidence = self._optional_float(
            overview.get("average_identity_confidence")
        )
        assessment.suspected_id_switches = int(overview.get("suspected_id_switches", 0))
        assessment.fragmented_tracks = int(overview.get("fragmented_tracks", 0))
        assessment.tracks_needing_review = int(overview.get("tracks_needing_review", 0))
        assessment.predictions_object = quality.get("predictions_object")
        assessment.metrics_json = {
            "health": overview,
            "tracker_runtime": tracker_runtime,
            "benchmark": quality.get("benchmark", {"status": "ground_truth_required"}),
        }
        assessment.thresholds_json = quality.get("thresholds", QUALITY_THRESHOLDS)

        quality_tracks = quality.get("tracks", [])
        existing = {
            item.track_id: item
            for item in db.query(TrackReviewItem)
            .filter(TrackReviewItem.run_id == run.id)
            .all()
        }
        for track in quality_tracks:
            track_id = int(track["track_id"])
            item = existing.get(track_id)
            if item is None:
                item = TrackReviewItem(
                    run_id=run.id,
                    track_id=track_id,
                    canonical_track_id=track_id,
                )
                db.add(item)
            item.team_number = self._optional_int(track.get("team"))
            if item.status not in {"approved", "rejected", "merged", "split"}:
                item.status = "pending"
            item.identity_confidence = float(track.get("identity_confidence", 0.0))
            item.reid_confidence = float(track.get("reid_confidence", 0.0))
            item.motion_consistency = float(track.get("motion_consistency", 0.0))
            item.team_consistency = float(track.get("team_consistency", 0.0))
            item.switch_risk = str(track.get("switch_risk", "medium"))
            item.fragment_count = int(track.get("fragment_count", 0))
            item.raw_id_transitions = int(track.get("raw_id_transitions", 0))
            item.first_frame = self._optional_int(track.get("first_frame"))
            item.last_frame = self._optional_int(track.get("last_frame"))
            item.observation_count = int(track.get("observation_count", 0))
            item.raw_track_ids = [int(value) for value in track.get("raw_track_ids", [])]
            item.issue_codes = [str(value) for value in track.get("issue_codes", [])]
            item.crop_objects = track.get("crop_objects", [])
            item.observations_json = track.get("review_observations", [])

        db.commit()
        db.refresh(assessment)
        return assessment

    def get_quality(
        self,
        db: Session,
        run: MatchAnalysisRun,
    ) -> dict[str, Any]:
        assessment = (
            db.query(TrackingQualityAssessment)
            .filter(TrackingQualityAssessment.run_id == run.id)
            .first()
        )
        if assessment is None:
            assessment = self.sync_from_summary(db, run, run.summary_json or {})
        items = (
            db.query(TrackReviewItem)
            .filter(TrackReviewItem.run_id == run.id)
            .order_by(TrackReviewItem.track_id)
            .all()
        )
        corrections = (
            db.query(TrackReviewCorrection)
            .filter(TrackReviewCorrection.run_id == run.id)
            .order_by(desc(TrackReviewCorrection.created_at), desc(TrackReviewCorrection.id))
            .all()
        )
        match = run.match
        team_ids = {
            value
            for value in (
                getattr(match, "primary_team_id", None),
                getattr(match, "opponent_team_id", None),
                getattr(match, "another_team_id", None),
            )
            if value is not None
        }
        players_query = db.query(Player)
        if team_ids:
            players_query = players_query.filter(Player.team_id.in_(team_ids))
        players = players_query.order_by(Player.team_id, Player.jersey_number, Player.name).all()
        return {
            "run_id": run.id,
            "match_id": run.match_id,
            "assessment": self._serialize_assessment(assessment),
            "tracks": [self._serialize_item(item) for item in items],
            "corrections": [self._serialize_correction(item) for item in corrections],
            "players": [
                {
                    "id": player.id,
                    "name": player.name,
                    "jersey_number": player.jersey_number,
                    "team_id": player.team_id,
                }
                for player in players
            ],
        }

    def apply_correction(
        self,
        db: Session,
        run: MatchAnalysisRun,
        payload: dict[str, Any],
    ) -> TrackReviewCorrection:
        action = str(payload.get("action", "")).lower()
        if action not in self.correction_actions:
            raise ValueError(f"Unsupported correction action: {action}")
        source_track_id = self._optional_int(payload.get("source_track_id"))
        if source_track_id is None:
            raise ValueError("source_track_id is required")
        source = self._get_item(db, run.id, source_track_id)
        before = self._snapshot_item(source)

        target_track_id = self._optional_int(payload.get("target_track_id"))
        split_frame = self._optional_int(payload.get("split_frame"))
        assigned_player_id = self._optional_int(payload.get("assigned_player_id"))
        assigned_team_number = self._optional_int(payload.get("assigned_team_number"))

        if action == "approve":
            source.status = "approved"
        elif action == "reject":
            source.status = "rejected"
        elif action == "merge":
            if target_track_id is None or target_track_id == source_track_id:
                raise ValueError("A different target_track_id is required for merge")
            target = self._get_item(db, run.id, target_track_id)
            source.canonical_track_id = target.canonical_track_id
            source.status = "merged"
        elif action == "split":
            if split_frame is None:
                raise ValueError("split_frame is required")
            if source.first_frame is not None and split_frame <= source.first_frame:
                raise ValueError("split_frame must be after the track start")
            if source.last_frame is not None and split_frame > source.last_frame:
                raise ValueError("split_frame must be inside the track")
            source.status = "split"
        elif action == "assign_player":
            if assigned_player_id is None or db.get(Player, assigned_player_id) is None:
                raise ValueError("A valid assigned_player_id is required")
            source.assigned_player_id = assigned_player_id
            source.status = "approved"
        elif action == "change_team":
            if assigned_team_number not in {1, 2}:
                raise ValueError("assigned_team_number must be 1 or 2")
            source.team_number = assigned_team_number

        correction = TrackReviewCorrection(
            run_id=run.id,
            action=action,
            source_track_id=source_track_id,
            target_track_id=target_track_id,
            split_frame=split_frame,
            assigned_player_id=assigned_player_id,
            assigned_team_number=assigned_team_number,
            before_json=before,
            after_json=self._snapshot_item(source),
            note=payload.get("note"),
        )
        db.add(correction)
        self._refresh_assessment_status(db, run.id)
        db.commit()
        db.refresh(correction)
        return correction

    def undo_correction(
        self,
        db: Session,
        run: MatchAnalysisRun,
        correction_id: int,
    ) -> TrackReviewCorrection:
        correction = (
            db.query(TrackReviewCorrection)
            .filter(TrackReviewCorrection.run_id == run.id)
            .filter(TrackReviewCorrection.id == correction_id)
            .first()
        )
        if correction is None:
            raise ValueError("Correction not found")
        if correction.undone:
            raise ValueError("Correction is already undone")
        newer = (
            db.query(TrackReviewCorrection)
            .filter(TrackReviewCorrection.run_id == run.id)
            .filter(TrackReviewCorrection.source_track_id == correction.source_track_id)
            .filter(TrackReviewCorrection.undone.is_(False))
            .filter(TrackReviewCorrection.id > correction.id)
            .first()
        )
        if newer is not None:
            raise ValueError("Undo newer corrections for this track first")
        if correction.source_track_id is not None and correction.before_json:
            item = self._get_item(db, run.id, correction.source_track_id)
            self._restore_snapshot(item, correction.before_json)
        correction.undone = True
        self._refresh_assessment_status(db, run.id)
        db.commit()
        db.refresh(correction)
        return correction

    def recalculate(self, db: Session, run: MatchAnalysisRun) -> dict[str, Any]:
        summary = deepcopy(run.summary_json or {})
        layer_summary = summary.get("visual_layers") or {}
        original_object = layer_summary.get("original_object_name") or layer_summary.get("object_name")
        if not original_object:
            raise ValueError("This run does not have visual layer data")
        layers = self._get_json(BUCKET_NAME, original_object)
        corrections = (
            db.query(TrackReviewCorrection)
            .filter(TrackReviewCorrection.run_id == run.id)
            .filter(TrackReviewCorrection.undone.is_(False))
            .order_by(TrackReviewCorrection.id)
            .all()
        )
        corrected = self._apply_layer_corrections(layers, corrections, db)
        prefix = original_object.rsplit("/", 1)[0]
        corrected_object = f"{prefix}/visual_layers.corrected.json"
        self._put_json(BUCKET_NAME, corrected_object, corrected)
        layer_summary = {
            **layer_summary,
            "object_name": corrected_object,
            "original_object_name": original_object,
            "corrections_applied": len(corrections),
            "tracks_count": len(corrected.get("tracks", [])),
        }
        summary["visual_layers"] = layer_summary
        tracking_quality = deepcopy(summary.get("tracking_quality") or {})
        tracking_quality["corrected_layers_object"] = corrected_object
        tracking_quality["active_corrections"] = len(corrections)
        summary["tracking_quality"] = tracking_quality
        run.summary_json = summary
        self._refresh_assessment_status(db, run.id)
        if run.summary_object:
            self._put_json(BUCKET_NAME, run.summary_object, summary)
        db.commit()
        return {
            "run_id": run.id,
            "object_name": corrected_object,
            "corrections_applied": len(corrections),
            "tracks_count": len(corrected.get("tracks", [])),
        }

    def benchmark(
        self,
        db: Session,
        run: MatchAnalysisRun,
        ground_truth: dict[str, Any],
        iou_threshold: float,
    ) -> dict[str, Any]:
        assessment = (
            db.query(TrackingQualityAssessment)
            .filter(TrackingQualityAssessment.run_id == run.id)
            .first()
        )
        if assessment is None:
            assessment = self.sync_from_summary(db, run, run.summary_json or {})
        if not assessment.predictions_object:
            raise ValueError("This run predates quality prediction artifacts; run the analysis again")
        predictions = self._get_predictions(BUCKET_NAME, assessment.predictions_object)
        metrics = evaluate_tracking(predictions, ground_truth, iou_threshold)
        prefix = assessment.predictions_object.rsplit("/", 1)[0]
        ground_truth_object = f"{prefix}/ground_truth.json"
        self._put_json(BUCKET_NAME, ground_truth_object, ground_truth)
        assessment.benchmark_status = "measured"
        assessment.id_switches = int(metrics["id_switches"])
        assessment.idf1 = float(metrics["idf1"])
        assessment.hota = float(metrics["hota"])
        assessment.fragmentation = int(metrics["fragmentation"])
        assessment.ground_truth_object = ground_truth_object
        assessment.metrics_json = {
            **(assessment.metrics_json or {}),
            "benchmark": metrics,
        }
        summary = deepcopy(run.summary_json or {})
        tracking_quality = deepcopy(summary.get("tracking_quality") or {})
        tracking_quality["benchmark"] = metrics
        summary["tracking_quality"] = tracking_quality
        run.summary_json = summary
        if run.summary_object:
            self._put_json(BUCKET_NAME, run.summary_object, summary)
        db.commit()
        return metrics

    def _apply_layer_corrections(
        self,
        layers: dict[str, Any],
        corrections: list[TrackReviewCorrection],
        db: Session,
    ) -> dict[str, Any]:
        corrected = deepcopy(layers)
        tracks = {int(track["track_id"]): track for track in corrected.get("tracks", [])}
        next_track_id = max(tracks, default=0) + 1
        for correction in corrections:
            source_id = correction.source_track_id
            source = tracks.get(source_id) if source_id is not None else None
            if source is None:
                continue
            if correction.action == "reject":
                del tracks[source_id]
            elif correction.action == "merge" and correction.target_track_id in tracks:
                target = tracks[correction.target_track_id]
                for path_key in ("video_path", "pitch_path"):
                    target[path_key] = self._merge_paths(
                        target.get(path_key, []),
                        source.get(path_key, []),
                    )
                target["frames"] = len(target.get("video_path", []))
                target["first_frame"] = min(
                    value
                    for value in (target.get("first_frame"), source.get("first_frame"))
                    if value is not None
                )
                target["last_frame"] = max(
                    value
                    for value in (target.get("last_frame"), source.get("last_frame"))
                    if value is not None
                )
                del tracks[source_id]
            elif correction.action == "split" and correction.split_frame is not None:
                split_frame = correction.split_frame
                split_track = deepcopy(source)
                split_track["track_id"] = next_track_id
                split_track["color"] = self._track_color(next_track_id)
                for path_key in ("video_path", "pitch_path"):
                    path = source.get(path_key, [])
                    source[path_key] = [point for point in path if int(point[0]) < split_frame]
                    split_track[path_key] = [point for point in path if int(point[0]) >= split_frame]
                self._refresh_layer_track(source)
                self._refresh_layer_track(split_track)
                if split_track.get("video_path"):
                    tracks[next_track_id] = split_track
                    next_track_id += 1
            elif correction.action == "change_team":
                source["team"] = correction.assigned_team_number
            elif correction.action == "assign_player" and correction.assigned_player_id:
                player = db.get(Player, correction.assigned_player_id)
                source["player_id"] = correction.assigned_player_id
                source["player_name"] = player.name if player is not None else None
                source["jersey_number"] = player.jersey_number if player is not None else None
        corrected["schema_version"] = max(2, int(corrected.get("schema_version", 1)))
        corrected["corrections_applied"] = len(corrections)
        corrected["tracks"] = [tracks[key] for key in sorted(tracks)]
        return corrected

    def _quality_from_legacy_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        tracks = []
        for track in summary.get("tracks", []):
            confidence = float(track.get("identity_confidence", 0.0))
            raw_ids = track.get("raw_ids_seen", [])
            transitions = max(0, len(raw_ids) - 1)
            risk = "high" if confidence < 0.52 else "medium" if confidence < 0.78 else "low"
            tracks.append(
                {
                    "track_id": track.get("track_id"),
                    "team": track.get("team"),
                    "identity_confidence": confidence,
                    "reid_confidence": min(1.0, float(track.get("appearance_references", 0)) / 5.0),
                    "motion_consistency": confidence,
                    "team_consistency": float(track.get("jersey_family_confidence", 0.0)),
                    "switch_risk": risk,
                    "fragment_count": 0,
                    "raw_id_transitions": transitions,
                    "first_frame": track.get("first_frame"),
                    "last_frame": track.get("last_frame"),
                    "observation_count": track.get("frames", 0),
                    "raw_track_ids": raw_ids,
                    "issue_codes": ["legacy_run_no_quality_artifact"],
                }
            )
        confidences = [track["identity_confidence"] for track in tracks]
        needing_review = sum(1 for track in tracks if track["switch_risk"] != "low")
        return {
            "engine": "legacy_summary_adapter",
            "tracker_runtime": {
                "engine": summary.get("tracker", "unknown"),
                "reid": {"active": False, "model": None},
            },
            "overview": {
                "status": "needs_review" if tracks else "pending",
                "average_identity_confidence": (
                    round(sum(confidences) / len(confidences), 4) if confidences else None
                ),
                "suspected_id_switches": 0,
                "fragmented_tracks": 0,
                "tracks_needing_review": needing_review,
            },
            "benchmark": {"status": "ground_truth_required"},
            "thresholds": QUALITY_THRESHOLDS,
            "tracks": tracks,
        }

    def _refresh_assessment_status(self, db: Session, run_id: int) -> None:
        assessment = (
            db.query(TrackingQualityAssessment)
            .filter(TrackingQualityAssessment.run_id == run_id)
            .first()
        )
        if assessment is None:
            return
        items = db.query(TrackReviewItem).filter(TrackReviewItem.run_id == run_id).all()
        open_items = [
            item
            for item in items
            if item.status == "pending" and item.switch_risk != "low"
        ]
        assessment.tracks_needing_review = len(open_items)
        assessment.status = "approved" if items and not open_items else "needs_review"
        assessment.reviewed_at = _utc_now() if assessment.status == "approved" else None

    def _get_item(self, db: Session, run_id: int, track_id: int) -> TrackReviewItem:
        item = (
            db.query(TrackReviewItem)
            .filter(TrackReviewItem.run_id == run_id)
            .filter(TrackReviewItem.track_id == track_id)
            .first()
        )
        if item is None:
            raise ValueError(f"Track {track_id} was not found in this run")
        return item

    def _serialize_assessment(self, item: TrackingQualityAssessment) -> dict[str, Any]:
        return {
            "id": item.id,
            "status": item.status,
            "tracker_engine": item.tracker_engine,
            "reid_enabled": item.reid_enabled,
            "reid_model": item.reid_model,
            "average_identity_confidence": item.average_identity_confidence,
            "suspected_id_switches": item.suspected_id_switches,
            "fragmented_tracks": item.fragmented_tracks,
            "tracks_needing_review": item.tracks_needing_review,
            "benchmark_status": item.benchmark_status,
            "id_switches": item.id_switches,
            "idf1": item.idf1,
            "hota": item.hota,
            "fragmentation": item.fragmentation,
            "predictions_object": item.predictions_object,
            "ground_truth_object": item.ground_truth_object,
            "metrics": item.metrics_json,
            "thresholds": item.thresholds_json,
            "updated_at": item.updated_at,
            "reviewed_at": item.reviewed_at,
        }

    def _serialize_item(self, item: TrackReviewItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "track_id": item.track_id,
            "canonical_track_id": item.canonical_track_id,
            "team": item.team_number,
            "assigned_player_id": item.assigned_player_id,
            "assigned_player": {
                "id": item.assigned_player.id,
                "name": item.assigned_player.name,
                "jersey_number": item.assigned_player.jersey_number,
            }
            if item.assigned_player is not None
            else None,
            "status": item.status,
            "identity_confidence": item.identity_confidence,
            "reid_confidence": item.reid_confidence,
            "motion_consistency": item.motion_consistency,
            "team_consistency": item.team_consistency,
            "switch_risk": item.switch_risk,
            "fragment_count": item.fragment_count,
            "raw_id_transitions": item.raw_id_transitions,
            "first_frame": item.first_frame,
            "last_frame": item.last_frame,
            "observation_count": item.observation_count,
            "raw_track_ids": item.raw_track_ids or [],
            "issue_codes": item.issue_codes or [],
            "crop_objects": item.crop_objects or [],
            "observations": item.observations_json or [],
        }

    def _serialize_correction(self, item: TrackReviewCorrection) -> dict[str, Any]:
        return {
            "id": item.id,
            "action": item.action,
            "source_track_id": item.source_track_id,
            "target_track_id": item.target_track_id,
            "split_frame": item.split_frame,
            "assigned_player_id": item.assigned_player_id,
            "assigned_team_number": item.assigned_team_number,
            "note": item.note,
            "undone": item.undone,
            "created_at": item.created_at,
        }

    def _snapshot_item(self, item: TrackReviewItem) -> dict[str, Any]:
        return {
            "canonical_track_id": item.canonical_track_id,
            "team_number": item.team_number,
            "assigned_player_id": item.assigned_player_id,
            "status": item.status,
        }

    def _restore_snapshot(self, item: TrackReviewItem, snapshot: dict[str, Any]) -> None:
        item.canonical_track_id = int(snapshot["canonical_track_id"])
        item.team_number = self._optional_int(snapshot.get("team_number"))
        item.assigned_player_id = self._optional_int(snapshot.get("assigned_player_id"))
        item.status = str(snapshot.get("status", "pending"))

    def _refresh_layer_track(self, track: dict[str, Any]) -> None:
        video_path = track.get("video_path", [])
        track["frames"] = len(video_path)
        track["first_frame"] = int(video_path[0][0]) if video_path else None
        track["last_frame"] = int(video_path[-1][0]) if video_path else None

    def _merge_paths(self, first: list[list[Any]], second: list[list[Any]]) -> list[list[Any]]:
        by_frame = {int(point[0]): point for point in first}
        for point in second:
            by_frame.setdefault(int(point[0]), point)
        return [by_frame[frame] for frame in sorted(by_frame)]

    def _track_color(self, track_id: int) -> str:
        hue = (track_id * 0.61803398875) % 1.0
        import colorsys

        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.94)
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    def _get_json(self, bucket: str, object_name: str) -> dict[str, Any]:
        response = client.get_object(bucket, object_name)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def _get_predictions(self, bucket: str, object_name: str) -> dict[str, Any]:
        if not object_name.endswith(".jsonl"):
            return self._get_json(bucket, object_name)
        response = client.get_object(bucket, object_name)
        try:
            observations = [
                json.loads(line)
                for line in response.read().decode("utf-8").splitlines()
                if line.strip()
            ]
            return {"observations": observations}
        finally:
            response.close()
            response.release_conn()

    def _put_json(self, bucket: str, object_name: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )

    def _optional_int(self, value: Any) -> int | None:
        return int(value) if value is not None else None

    def _optional_float(self, value: Any) -> float | None:
        return float(value) if value is not None else None
