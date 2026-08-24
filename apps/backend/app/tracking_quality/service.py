from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import io
import json
import math
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
from app.match_analysis_plus.analytics_v1 import AnalyticsRealV1
from app.match_analysis_plus.reports_v2 import ReportsV2Builder


QUALITY_THRESHOLDS = {
    "approve_identity_confidence": 0.82,
    "review_identity_confidence": 0.68,
    "high_risk_identity_confidence": 0.52,
    "high_risk_fragments": 2,
    "high_risk_raw_id_transitions": 4,
}
ALLOWED_PARTICIPANT_ROLES = {
    "player",
    "goalkeeper",
    "referee",
    "assistant_referee",
    "staff_outside_pitch",
}
ANALYTICS_PARTICIPANT_ROLES = {"player", "goalkeeper"}
MAX_PLAUSIBLE_PLAYER_SPEED_KMH = 45.0


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
        "change_role",
    }

    def __init__(self) -> None:
        self.analytics_engine = AnalyticsRealV1()
        self.report_builder = ReportsV2Builder()

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
        release_gate = (quality.get("benchmark") or {}).get("release_gate")
        if isinstance(release_gate, dict):
            assessment.release_gate_status = str(release_gate.get("status") or "not_ready")
            assessment.release_gate_json = release_gate

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
            if item.role_evidence_json != ["manual_track_review"]:
                item.role_name = str(track.get("role_name") or "player")
                item.role_confidence = float(track.get("role_confidence", 0.0))
                item.role_locked = bool(track.get("role_locked", False))
                item.role_evidence_json = [
                    str(value) for value in track.get("role_evidence", [])
                ]
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
        assigned_role_name = self._optional_role(payload.get("assigned_role_name"))

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
            existing_ids = [
                int(item.track_id)
                for item in db.query(TrackReviewItem)
                .filter(TrackReviewItem.run_id == run.id)
                .all()
            ]
            target_track_id = max(existing_ids, default=0) + 1
            original_last_frame = source.last_frame
            original_observations = source.observation_count
            total_span = max(1, (original_last_frame or split_frame) - (source.first_frame or 0) + 1)
            tail_ratio = max(0.0, min(1.0, ((original_last_frame or split_frame) - split_frame + 1) / total_span))
            tail_observations = max(1, int(round(original_observations * tail_ratio)))
            split_item = TrackReviewItem(
                run_id=run.id,
                track_id=target_track_id,
                canonical_track_id=target_track_id,
                team_number=source.team_number,
                role_name=source.role_name,
                role_confidence=source.role_confidence,
                role_locked=source.role_locked,
                role_evidence_json=list(source.role_evidence_json or []),
                assigned_player_id=None,
                status="pending",
                identity_confidence=source.identity_confidence,
                reid_confidence=source.reid_confidence,
                motion_consistency=source.motion_consistency,
                team_consistency=source.team_consistency,
                switch_risk=source.switch_risk,
                fragment_count=0,
                raw_id_transitions=0,
                first_frame=split_frame,
                last_frame=original_last_frame,
                observation_count=tail_observations,
                raw_track_ids=list(source.raw_track_ids or []),
                issue_codes=["manual_split_requires_review"],
                crop_objects=[
                    crop
                    for crop in (source.crop_objects or [])
                    if int(crop.get("frame", -1)) >= split_frame
                ],
                observations_json=[
                    observation
                    for observation in (source.observations_json or [])
                    if int(observation.get("frame", -1)) >= split_frame
                ],
            )
            source.last_frame = split_frame - 1
            source.observation_count = max(0, original_observations - tail_observations)
            source.crop_objects = [
                crop
                for crop in (source.crop_objects or [])
                if int(crop.get("frame", -1)) < split_frame
            ]
            source.observations_json = [
                observation
                for observation in (source.observations_json or [])
                if int(observation.get("frame", -1)) < split_frame
            ]
            source.status = "split"
            db.add(split_item)
        elif action == "assign_player":
            if assigned_player_id is None or db.get(Player, assigned_player_id) is None:
                raise ValueError("A valid assigned_player_id is required")
            source.assigned_player_id = assigned_player_id
            source.status = "approved"
        elif action == "change_team":
            if assigned_team_number not in {1, 2}:
                raise ValueError("assigned_team_number must be 1 or 2")
            source.team_number = assigned_team_number
        elif action == "change_role":
            if assigned_role_name is None:
                raise ValueError("assigned_role_name must be a supported participant role")
            source.role_name = assigned_role_name
            source.role_confidence = 1.0
            source.role_locked = True
            source.role_evidence_json = ["manual_track_review"]

        correction = TrackReviewCorrection(
            run_id=run.id,
            action=action,
            source_track_id=source_track_id,
            target_track_id=target_track_id,
            split_frame=split_frame,
            assigned_player_id=assigned_player_id,
            assigned_team_number=assigned_team_number,
            assigned_role_name=assigned_role_name,
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
        if correction.action == "split" and correction.target_track_id is not None:
            split_item = (
                db.query(TrackReviewItem)
                .filter(TrackReviewItem.run_id == run.id)
                .filter(TrackReviewItem.track_id == correction.target_track_id)
                .first()
            )
            if split_item is not None:
                db.delete(split_item)
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
        corrected = self._apply_layer_corrections(layers, corrections, db, run.id)
        prefix = original_object.rsplit("/", 1)[0]
        corrected_object = f"{prefix}/visual_layers.corrected.json"
        self._put_json(BUCKET_NAME, corrected_object, corrected)
        canonical_analytics = self._build_canonical_analytics(
            corrected,
            float(corrected.get("fps") or summary.get("fps") or 25.0),
        )
        canonical_possession = self._canonicalize_possession(
            summary.get("possession") or {},
            corrections,
            corrected,
        )
        canonical_analytics["possession"] = canonical_possession
        match = run.match
        team_context = {
            "analysis_scope": getattr(match, "analysis_scope", "both_teams_full"),
            "analyze_primary_players": bool(getattr(match, "analyze_primary_players", True)),
            "analyze_opponent_players": bool(getattr(match, "analyze_opponent_players", True)),
            "team_labels": {
                "1": getattr(match, "primary_team_name", None) or "Team 1",
                "2": getattr(match, "another_team_name", None)
                or getattr(match, "opponent_team_name", None)
                or "Team 2",
            },
            "formations": {
                "1": getattr(match, "formation", None),
                "2": getattr(match, "another_formation", None),
            },
        }
        analytics_real = self.analytics_engine.build(
            layers=corrected,
            possession=canonical_possession,
            pitch_gate=((summary.get("radar") or {}).get("quality_gate") or {}),
            ball_gate=((summary.get("ball_filter") or {}).get("quality_gate") or {}),
            team_identity=summary.get("team_classifier") or {},
            team_context=team_context,
        )
        analytics_object = f"{prefix}/canonical_analytics.json"
        report_object = f"{prefix}/canonical_report.json"
        canonical_report = self._build_canonical_report(run, canonical_analytics)
        self._put_json(BUCKET_NAME, analytics_object, canonical_analytics)
        self._put_json(BUCKET_NAME, report_object, canonical_report)
        analytics_real_object = f"{prefix}/analytics_v1.corrected.json"
        report_v2_object = f"{prefix}/reports-v2/report.corrected.json"
        report_v2_pdf_object = f"{prefix}/reports-v2/report.corrected.pdf"
        team_chart_object = f"{prefix}/reports-v2/team-overview.corrected.png"
        heatmap_atlas_object = f"{prefix}/reports-v2/player-heatmaps.corrected.png"
        report_v2 = self.report_builder.build(analytics_real, summary, team_context)
        report_v2["artifacts"] = {
            "json": report_v2_object,
            "pdf": report_v2_pdf_object,
            "team_chart": team_chart_object,
            "player_heatmaps": heatmap_atlas_object,
        }
        self._put_json(BUCKET_NAME, analytics_real_object, analytics_real)
        self._put_bytes(BUCKET_NAME, report_v2_object, json.dumps(report_v2, ensure_ascii=False).encode("utf-8"), "application/json")
        self._put_bytes(BUCKET_NAME, report_v2_pdf_object, self.report_builder.pdf(report_v2), "application/pdf")
        self._put_bytes(BUCKET_NAME, team_chart_object, self.report_builder.team_chart_png(report_v2), "image/png")
        self._put_bytes(BUCKET_NAME, heatmap_atlas_object, self.report_builder.heatmap_atlas_png(report_v2), "image/png")

        assessment = (
            db.query(TrackingQualityAssessment)
            .filter(TrackingQualityAssessment.run_id == run.id)
            .first()
        )
        corrected_predictions_object = None
        corrected_observations = 0
        if assessment is not None and assessment.predictions_object:
            predictions = self._get_predictions(BUCKET_NAME, assessment.predictions_object)
            corrected_predictions = self._apply_prediction_corrections(
                predictions,
                corrections,
                db,
                run.id,
            )
            corrected_predictions_object = f"{prefix}/tracking_quality_predictions.corrected.jsonl"
            self._put_jsonl(
                BUCKET_NAME,
                corrected_predictions_object,
                corrected_predictions.get("observations", []),
            )
            corrected_observations = len(corrected_predictions.get("observations", []))
        layer_summary = {
            **layer_summary,
            "status": "corrected" if corrections else layer_summary.get("status", "ready"),
            "object_name": corrected_object,
            "original_object_name": original_object,
            "corrections_applied": len(corrections),
            "tracks_count": len(corrected.get("tracks", [])),
            "canonical_identity_overlay": True,
        }
        summary["visual_layers"] = layer_summary
        summary["tracks"] = canonical_analytics["tracks"]
        summary["canonical_analytics"] = {
            **canonical_analytics,
            "object_name": analytics_object,
        }
        summary["possession"] = canonical_possession
        summary["canonical_report"] = {
            "status": "ready",
            "object_name": report_object,
            "schema_version": canonical_report["schema_version"],
            "teams_count": len(canonical_report["teams"]),
            "players_count": len(canonical_report["players"]),
        }
        summary["analytics_real_v1"] = {
            **analytics_real,
            "object_name": analytics_real_object,
        }
        summary["reports_v2"] = {
            "status": "ready",
            "schema_version": report_v2["schema_version"],
            "teams_count": len(report_v2["teams"]),
            "players_count": len(report_v2["players"]),
            "artifacts": report_v2["artifacts"],
            "canonical_corrections_applied": len(corrections),
        }
        summary["canonical_video"] = {
            "status": "ready",
            "strategy": "client_side_canonical_overlay",
            "base_object_name": run.output_object,
            "overlay_object_name": corrected_object,
            "note": "The review player renders corrected canonical identities over the saved analysis video.",
        }
        tracking_quality = deepcopy(summary.get("tracking_quality") or {})
        tracking_quality["corrected_layers_object"] = corrected_object
        tracking_quality["corrected_predictions_object"] = corrected_predictions_object
        tracking_quality["active_corrections"] = len(corrections)
        tracking_quality["canonical_tracks_count"] = len(canonical_analytics["tracks"])
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
            "corrected_predictions_object": corrected_predictions_object,
            "corrected_observations": corrected_observations,
            "canonical_analytics_object": analytics_object,
            "canonical_report_object": report_object,
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
        release_gate = metrics.get("release_gate") or {}
        assessment.release_gate_status = str(release_gate.get("status") or "not_ready")
        assessment.release_gate_json = release_gate
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

    def build_ground_truth_draft(
        self,
        db: Session,
        run: MatchAnalysisRun,
        start_frame: int,
        end_frame: int,
        sample_every_frames: int,
        track_ids: list[int] | None = None,
        scenario: str = "general",
        camera_style: str = "tactical",
        critical: bool = False,
    ) -> dict[str, Any]:
        if end_frame < start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if end_frame - start_frame > 3000:
            raise ValueError("A ground-truth clip cannot exceed 3000 frames")
        assessment = (
            db.query(TrackingQualityAssessment)
            .filter(TrackingQualityAssessment.run_id == run.id)
            .first()
        )
        if assessment is None:
            assessment = self.sync_from_summary(db, run, run.summary_json or {})
        if not assessment.predictions_object:
            raise ValueError("This run does not contain tracking prediction artifacts")

        predictions = self._get_predictions(BUCKET_NAME, assessment.predictions_object)
        selected_ids = set(track_ids or [])
        frames: dict[int, list[dict[str, Any]]] = {}
        source_frames: dict[int, int] = {}
        for observation in predictions.get("observations", []):
            frame_index = int(observation.get("frame", -1))
            track_id = int(observation.get("track_id", -1))
            if frame_index < start_frame or frame_index > end_frame:
                continue
            if (frame_index - start_frame) % sample_every_frames != 0:
                continue
            if selected_ids and track_id not in selected_ids:
                continue
            source_frame = observation.get("source_frame")
            if source_frame is not None:
                source_frames[frame_index] = int(source_frame)
            frames.setdefault(frame_index, []).append(
                {
                    "identity_id": f"identity-{track_id}",
                    "bbox": observation["bbox"],
                    "source_frame": int(source_frame) if source_frame is not None else None,
                    "source_track_id": track_id,
                    "source_raw_track_id": observation.get("raw_track_id"),
                    "team": observation.get("team"),
                    "role_name": observation.get("role_name"),
                    "review_state": "unverified",
                }
            )
        if not frames:
            raise ValueError("No tracking observations exist in the selected clip")

        payload = {
            "schema_version": "tracking_ground_truth.v2",
            "coverage": "all_visible_identities" if not selected_ids else "selected_identities",
            "verification": {
                "status": "draft",
                "annotator": None,
                "reviewed_at": None,
            },
            "source": {
                "match_id": run.match_id,
                "run_id": run.id,
                "predictions_object": assessment.predictions_object,
            },
            "clips": [
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    **(
                        {
                            "source_start_frame": min(source_frames.values()),
                            "source_end_frame": max(source_frames.values()),
                        }
                        if source_frames
                        else {}
                    ),
                    "sample_every_frames": sample_every_frames,
                    "coverage": "all_visible_identities" if not selected_ids else "selected_identities",
                    "scenario": str(scenario or "general").lower(),
                    "camera_style": str(camera_style or "tactical").lower(),
                    "critical": bool(critical),
                }
            ],
            "instructions": [
                "Correct every identity_id and bbox in the selected frames.",
                "Add missing visible identities and remove false detections.",
                "Set every review_state to verified, then set verification.status to verified.",
                "Use a stable identity_id for the same physical person across every clip.",
            ],
            "frames": [
                {"frame": frame_index, "objects": frames[frame_index]}
                for frame_index in sorted(frames)
            ],
        }
        prefix = assessment.predictions_object.rsplit("/", 1)[0]
        object_name = (
            f"{prefix}/ground-truth/draft_{start_frame}_{end_frame}_"
            f"step_{sample_every_frames}.json"
        )
        self._put_json(BUCKET_NAME, object_name, payload)
        return {
            "object_name": object_name,
            "frame_count": len(frames),
            "annotation_count": sum(len(items) for items in frames.values()),
            "ground_truth": payload,
        }

    def _apply_layer_corrections(
        self,
        layers: dict[str, Any],
        corrections: list[TrackReviewCorrection],
        db: Session,
        run_id: int,
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
                target["canonical_track_id"] = int(correction.target_track_id)
                del tracks[source_id]
            elif correction.action == "split" and correction.split_frame is not None:
                split_frame = correction.split_frame
                split_track = deepcopy(source)
                split_track_id = correction.target_track_id or next_track_id
                split_track["track_id"] = split_track_id
                split_track["canonical_track_id"] = split_track_id
                split_track["color"] = self._track_color(split_track_id)
                for path_key in ("video_path", "pitch_path"):
                    path = source.get(path_key, [])
                    source[path_key] = [point for point in path if int(point[0]) < split_frame]
                    split_track[path_key] = [point for point in path if int(point[0]) >= split_frame]
                self._refresh_layer_track(source)
                self._refresh_layer_track(split_track)
                if split_track.get("video_path"):
                    tracks[split_track_id] = split_track
                    next_track_id = max(next_track_id, split_track_id + 1)
            elif correction.action == "change_team":
                source["team"] = correction.assigned_team_number
            elif correction.action == "change_role" and correction.assigned_role_name:
                source["role_name"] = correction.assigned_role_name
                source["role_confidence"] = 1.0
                source["role_locked"] = True
                source["role_evidence"] = ["manual_track_review"]
            elif correction.action == "assign_player" and correction.assigned_player_id:
                player = db.get(Player, correction.assigned_player_id)
                source["player_id"] = correction.assigned_player_id
                source["player_name"] = player.name if player is not None else None
                source["jersey_number"] = player.jersey_number if player is not None else None
        review_items = {
            int(item.track_id): item
            for item in db.query(TrackReviewItem)
            .filter(TrackReviewItem.run_id == run_id)
            .all()
        }
        for track_id, track in tracks.items():
            item = review_items.get(track_id)
            track["canonical_track_id"] = int(
                item.canonical_track_id if item is not None else track_id
            )
            if item is None:
                continue
            track["team"] = item.team_number
            track["role_name"] = item.role_name
            track["role_confidence"] = item.role_confidence
            track["role_locked"] = item.role_locked
            track["role_evidence"] = list(item.role_evidence_json or [])
            if item.assigned_player is not None:
                track["player_id"] = item.assigned_player.id
                track["player_name"] = item.assigned_player.name
                track["jersey_number"] = item.assigned_player.jersey_number
        corrected["schema_version"] = max(2, int(corrected.get("schema_version", 1)))
        corrected["corrections_applied"] = len(corrections)
        corrected["tracks"] = [tracks[key] for key in sorted(tracks)]
        return corrected

    def _apply_prediction_corrections(
        self,
        predictions: dict[str, Any],
        corrections: list[TrackReviewCorrection],
        db: Session,
        run_id: int,
    ) -> dict[str, Any]:
        review_items = {
            int(item.track_id): item
            for item in db.query(TrackReviewItem)
            .filter(TrackReviewItem.run_id == run_id)
            .all()
        }
        corrected_by_frame_identity: dict[tuple[int, int], dict[str, Any]] = {}
        for raw in predictions.get("observations", []):
            observation = deepcopy(raw)
            original_track_id = self._optional_int(observation.get("track_id"))
            if original_track_id is None:
                continue
            frame = int(observation.get("frame", -1))
            canonical_track_id = original_track_id
            rejected = False
            for correction in corrections:
                source_id = correction.source_track_id
                if source_id is None or canonical_track_id != source_id:
                    continue
                if correction.action == "reject":
                    rejected = True
                    break
                if correction.action == "merge" and correction.target_track_id is not None:
                    canonical_track_id = int(correction.target_track_id)
                elif (
                    correction.action == "split"
                    and correction.target_track_id is not None
                    and correction.split_frame is not None
                    and frame >= int(correction.split_frame)
                ):
                    canonical_track_id = int(correction.target_track_id)
            if rejected:
                continue

            item = review_items.get(canonical_track_id)
            observation["source_track_id"] = original_track_id
            observation["track_id"] = canonical_track_id
            observation["canonical_track_id"] = canonical_track_id
            if item is not None:
                observation["team"] = item.team_number
                observation["role_name"] = item.role_name
                observation["role_confidence"] = item.role_confidence
                observation["role_locked"] = item.role_locked
                observation["assigned_player_id"] = item.assigned_player_id
            key = (frame, canonical_track_id)
            existing = corrected_by_frame_identity.get(key)
            if existing is None or self._observation_score(observation) > self._observation_score(existing):
                corrected_by_frame_identity[key] = observation

        return {
            "schema_version": "canonical_tracking_predictions.v1",
            "corrections_applied": len(corrections),
            "observations": [
                corrected_by_frame_identity[key]
                for key in sorted(corrected_by_frame_identity)
            ],
        }

    def _build_canonical_analytics(
        self,
        layers: dict[str, Any],
        fps: float,
    ) -> dict[str, Any]:
        safe_fps = max(1.0, fps)
        tracks: list[dict[str, Any]] = []
        team_totals: dict[str, dict[str, Any]] = {}
        role_counts: dict[str, int] = {}
        excluded_roles: dict[str, int] = {}

        for layer_track in layers.get("tracks", []):
            track_id = int(layer_track.get("canonical_track_id") or layer_track["track_id"])
            role_name = str(layer_track.get("role_name") or "player")
            role_counts[role_name] = role_counts.get(role_name, 0) + 1
            pitch_path = sorted(
                [point for point in layer_track.get("pitch_path", []) if len(point) >= 3],
                key=lambda point: int(point[0]),
            )
            distance_m = 0.0
            active_seconds = 0.0
            valid_speeds: list[float] = []
            rejected_metric_steps = 0
            for previous, current in zip(pitch_path, pitch_path[1:]):
                frame_delta = int(current[0]) - int(previous[0])
                if frame_delta <= 0 or frame_delta > safe_fps * 2.0:
                    rejected_metric_steps += 1
                    continue
                step_m = math.hypot(
                    float(current[1]) - float(previous[1]),
                    float(current[2]) - float(previous[2]),
                ) / 100.0
                seconds = frame_delta / safe_fps
                speed_kmh = step_m / max(seconds, 1e-6) * 3.6
                if not math.isfinite(speed_kmh) or speed_kmh > MAX_PLAUSIBLE_PLAYER_SPEED_KMH:
                    rejected_metric_steps += 1
                    continue
                distance_m += step_m
                active_seconds += seconds
                valid_speeds.append(speed_kmh)

            track_summary = {
                "track_id": track_id,
                "canonical_track_id": track_id,
                "source_track_id": int(layer_track["track_id"]),
                "team": layer_track.get("team"),
                "role_name": role_name,
                "player_id": layer_track.get("player_id"),
                "player_name": layer_track.get("player_name"),
                "jersey_number": layer_track.get("jersey_number"),
                "frames": int(layer_track.get("frames", len(layer_track.get("video_path", [])))),
                "first_frame": layer_track.get("first_frame"),
                "last_frame": layer_track.get("last_frame"),
                "distance_m": round(distance_m, 3),
                "average_speed_kmh": round(distance_m / active_seconds * 3.6, 3) if active_seconds else 0.0,
                "max_speed_kmh": round(max(valid_speeds), 3) if valid_speeds else 0.0,
                "movement_samples": len(layer_track.get("video_path", [])),
                "heatmap_samples": len(pitch_path),
                "metric_steps": len(valid_speeds),
                "rejected_metric_steps": rejected_metric_steps,
                "identity_confidence": layer_track.get("identity_confidence"),
                "switch_risk": layer_track.get("switch_risk"),
            }
            tracks.append(track_summary)
            if role_name not in ANALYTICS_PARTICIPANT_ROLES:
                excluded_roles[role_name] = excluded_roles.get(role_name, 0) + 1
                continue
            team_key = str(layer_track.get("team") or "unassigned")
            aggregate = team_totals.setdefault(
                team_key,
                {
                    "team": layer_track.get("team"),
                    "players_count": 0,
                    "total_distance_m": 0.0,
                    "movement_samples": 0,
                    "heatmap_samples": 0,
                },
            )
            aggregate["players_count"] += 1
            aggregate["total_distance_m"] += distance_m
            aggregate["movement_samples"] += track_summary["movement_samples"]
            aggregate["heatmap_samples"] += track_summary["heatmap_samples"]

        for aggregate in team_totals.values():
            aggregate["total_distance_m"] = round(float(aggregate["total_distance_m"]), 3)
        return {
            "schema_version": "canonical_analytics.v1",
            "coordinate_system": "pitch_centimeters",
            "fps": round(safe_fps, 4),
            "corrections_applied": int(layers.get("corrections_applied", 0)),
            "tracks_count": len(tracks),
            "analytics_tracks_count": sum(
                1 for track in tracks if track["role_name"] in ANALYTICS_PARTICIPANT_ROLES
            ),
            "role_counts": role_counts,
            "excluded_roles": excluded_roles,
            "teams": team_totals,
            "tracks": sorted(tracks, key=lambda track: int(track["track_id"])),
        }

    def _build_canonical_report(
        self,
        run: MatchAnalysisRun,
        analytics: dict[str, Any],
    ) -> dict[str, Any]:
        match = run.match
        labels = {
            "1": getattr(match, "primary_team_name", None) or "Team 1",
            "2": (
                getattr(match, "another_team_name", None)
                or getattr(match, "opponent_team_name", None)
                or "Team 2"
            ),
            "unassigned": "Unassigned",
        }
        teams = []
        for key, values in analytics.get("teams", {}).items():
            teams.append({"team_key": key, "name": labels.get(key, key), **values})
        return {
            "schema_version": "canonical_match_report.v1",
            "run_id": run.id,
            "match_id": run.match_id,
            "generated_at": _utc_now().isoformat() + "Z",
            "corrections_applied": analytics.get("corrections_applied", 0),
            "teams": teams,
            "players": [
                track
                for track in analytics.get("tracks", [])
                if track.get("role_name") in ANALYTICS_PARTICIPANT_ROLES
            ],
            "possession": analytics.get("possession", {}),
            "participants_excluded_from_player_analytics": analytics.get("excluded_roles", {}),
            "quality": {
                "coordinate_system": analytics.get("coordinate_system"),
                "maximum_plausible_speed_kmh": MAX_PLAUSIBLE_PLAYER_SPEED_KMH,
                "canonical_identity_required": True,
            },
        }

    def _canonicalize_possession(
        self,
        possession: dict[str, Any],
        corrections: list[TrackReviewCorrection],
        layers: dict[str, Any],
    ) -> dict[str, Any]:
        track_metadata = {
            int(track.get("canonical_track_id") or track["track_id"]): track
            for track in layers.get("tracks", [])
        }
        events = []
        for raw_event in possession.get("events", []):
            event = deepcopy(raw_event)
            frame = int(event.get("frame", 0))
            start_frame = int(event.get("start_frame", frame))
            from_track_id = self._canonical_track_id_at_frame(
                self._optional_int(event.get("from_track_id")),
                start_frame,
                corrections,
            )
            to_track_id = self._canonical_track_id_at_frame(
                self._optional_int(event.get("to_track_id")),
                frame,
                corrections,
            )
            if from_track_id is None or to_track_id is None or from_track_id == to_track_id:
                continue
            event["from_track_id"] = from_track_id
            event["to_track_id"] = to_track_id
            event["from_team"] = track_metadata.get(from_track_id, {}).get(
                "team", event.get("from_team")
            )
            event["to_team"] = track_metadata.get(to_track_id, {}).get(
                "team", event.get("to_team")
            )
            same_team = (
                event.get("from_team") is not None
                and event.get("from_team") == event.get("to_team")
            )
            if same_team and event.get("travel_m") is not None and float(event["travel_m"]) >= 1.2:
                event["type"] = "completed_pass"
            elif (
                event.get("from_team") is not None
                and event.get("to_team") is not None
                and event.get("from_team") != event.get("to_team")
            ):
                event["type"] = "turnover"
            else:
                event["type"] = "possession_change"
            events.append(event)

        player_frames: dict[str, int] = {}
        for raw_track_id, frames in (possession.get("player_frames") or {}).items():
            canonical_track_id = self._canonical_track_id_at_frame(
                self._optional_int(raw_track_id),
                0,
                corrections,
            )
            if canonical_track_id is None:
                continue
            key = str(canonical_track_id)
            player_frames[key] = player_frames.get(key, 0) + int(frames)
        return {
            **possession,
            "engine": "canonical_metric_ball_possession_and_pass_detection_v2",
            "player_frames": player_frames,
            "transitions": len(events),
            "completed_passes": sum(1 for event in events if event["type"] == "completed_pass"),
            "turnovers": sum(1 for event in events if event["type"] == "turnover"),
            "events": events,
            "canonical_track_ids": True,
        }

    def _canonical_track_id_at_frame(
        self,
        track_id: int | None,
        frame: int,
        corrections: list[TrackReviewCorrection],
    ) -> int | None:
        if track_id is None:
            return None
        canonical_track_id = track_id
        for correction in corrections:
            if correction.source_track_id != canonical_track_id:
                continue
            if correction.action == "reject":
                return None
            if correction.action == "merge" and correction.target_track_id is not None:
                canonical_track_id = int(correction.target_track_id)
            elif (
                correction.action == "split"
                and correction.target_track_id is not None
                and correction.split_frame is not None
                and frame >= int(correction.split_frame)
            ):
                canonical_track_id = int(correction.target_track_id)
        return canonical_track_id

    def _observation_score(self, observation: dict[str, Any]) -> float:
        return float(
            observation.get("identity_confidence")
            or observation.get("confidence")
            or 0.0
        )

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
            "release_gate_status": item.release_gate_status,
            "release_gate": item.release_gate_json,
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
            "role_name": item.role_name,
            "role_confidence": item.role_confidence,
            "role_locked": item.role_locked,
            "role_evidence": item.role_evidence_json or [],
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
            "assigned_role_name": item.assigned_role_name,
            "note": item.note,
            "undone": item.undone,
            "created_at": item.created_at,
        }

    def _snapshot_item(self, item: TrackReviewItem) -> dict[str, Any]:
        return {
            "canonical_track_id": item.canonical_track_id,
            "team_number": item.team_number,
            "role_name": item.role_name,
            "role_confidence": item.role_confidence,
            "role_locked": item.role_locked,
            "role_evidence_json": item.role_evidence_json,
            "assigned_player_id": item.assigned_player_id,
            "status": item.status,
            "first_frame": item.first_frame,
            "last_frame": item.last_frame,
            "observation_count": item.observation_count,
            "crop_objects": list(item.crop_objects or []),
            "observations_json": list(item.observations_json or []),
        }

    def _restore_snapshot(self, item: TrackReviewItem, snapshot: dict[str, Any]) -> None:
        item.canonical_track_id = int(snapshot["canonical_track_id"])
        item.team_number = self._optional_int(snapshot.get("team_number"))
        item.role_name = str(snapshot.get("role_name") or "player")
        item.role_confidence = float(snapshot.get("role_confidence", 0.0))
        item.role_locked = bool(snapshot.get("role_locked", False))
        item.role_evidence_json = list(snapshot.get("role_evidence_json") or [])
        item.assigned_player_id = self._optional_int(snapshot.get("assigned_player_id"))
        item.status = str(snapshot.get("status", "pending"))
        item.first_frame = self._optional_int(snapshot.get("first_frame"))
        item.last_frame = self._optional_int(snapshot.get("last_frame"))
        item.observation_count = int(snapshot.get("observation_count", 0))
        item.crop_objects = list(snapshot.get("crop_objects") or [])
        item.observations_json = list(snapshot.get("observations_json") or [])

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

    def _put_bytes(
        self,
        bucket: str,
        object_name: str,
        payload: bytes,
        content_type: str,
    ) -> None:
        client.put_object(
            bucket,
            object_name,
            io.BytesIO(payload),
            length=len(payload),
            content_type=content_type,
        )

    def _put_jsonl(
        self,
        bucket: str,
        object_name: str,
        rows: list[dict[str, Any]],
    ) -> None:
        data = (
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        ).encode("utf-8")
        client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="application/x-ndjson",
        )

    def _optional_int(self, value: Any) -> int | None:
        return int(value) if value is not None else None

    def _optional_float(self, value: Any) -> float | None:
        return float(value) if value is not None else None

    def _optional_role(self, value: Any) -> str | None:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in ALLOWED_PARTICIPANT_ROLES else None
