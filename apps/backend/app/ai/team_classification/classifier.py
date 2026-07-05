from typing import Any

from app.services.minio_client import BUCKET_NAME, client


class TeamClassifier:
    name = "team_classifier_v2_color_refs"

    def assign(
        self,
        tracks: list[dict[str, Any]],
        match_context: dict[str, Any] | None = None,
        frames: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        context = match_context or {}
        kit_reference_analysis = self._analyze_kit_references(
            context.get("kit_references", {})
        )
        frame_lookup = {
            frame.get("frame_index"): frame
            for frame in frames or []
        }
        enriched_tracks = [
            self._assign_track(
                track,
                context,
                frame_lookup,
                kit_reference_analysis,
            )
            for track in tracks
        ]

        return {
            "status": "ok",
            "data": {
                "tracks": enriched_tracks,
                "tracks_count": len(enriched_tracks),
                "kit_references": context.get("kit_references", {}),
                "kit_reference_analysis": kit_reference_analysis,
            },
            "meta": {
                "engine": self.name,
                "current_method": "compare_player_jersey_crop_to_kit_reference_colors",
                "fallback_method": "stub_object_key",
                "match_context": context,
            },
        }

    def _assign_track(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
        frame_lookup: dict[int, dict[str, Any]],
        kit_reference_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        if track.get("class_name") == "ball":
            return {
                **track,
                "team_context": "ball",
                "team_label": "ball",
                "team_assignment_source": "class_name",
            }

        if context.get("match_type") in {"internal_scrimmage", "academy_match"}:
            return {
                **track,
                "team_context": "club_internal",
                "team_label": "club_internal",
                "team_assignment_source": "match_type",
                "team_assignment_confidence": 0.8,
                "team_confidence": 0.8,
            }

        crop_analysis = self._analyze_track_jersey_crops(track, frame_lookup)
        primary_reference = kit_reference_analysis.get("primary_team_selected_kit")
        opponent_reference = kit_reference_analysis.get("opponent_team_kit")
        crop_assignment = self._assign_from_crop_color(
            crop_analysis,
            primary_reference,
            opponent_reference,
        )

        if crop_assignment is not None:
            return {
                **track,
                "team_context": crop_assignment["team_context"],
                "team_label": crop_assignment["team_context"],
                "team_assignment_source": "jersey_crop_color",
                "team_assignment_confidence": crop_assignment["confidence"],
                "team_confidence": crop_assignment["confidence"],
                "kit_match_score": crop_assignment["kit_match_score"],
                "dominant_colors": crop_analysis.get("dominant_colors", []),
                "jersey_crop_analysis": crop_analysis,
            }

        fallback_team = self._infer_team_context(track, context)
        return {
            **track,
            "team_context": fallback_team,
            "team_label": fallback_team,
            "team_assignment_source": "visual_unknown_fallback",
            "team_assignment_confidence": 0.25 if fallback_team != "unknown" else 0.0,
            "team_confidence": 0.25 if fallback_team != "unknown" else 0.0,
            "kit_match_score": None,
            "dominant_colors": crop_analysis.get("dominant_colors", []) if crop_analysis else [],
            "jersey_crop_analysis": crop_analysis,
        }

    def _analyze_track_jersey_crops(
        self,
        track: dict[str, Any],
        frame_lookup: dict[int, dict[str, Any]],
    ) -> dict[str, Any] | None:
        analyses = [
            analysis
            for analysis in [
                self._analyze_track_jersey_crop(frame, frame_lookup)
                for frame in self._sample_track_frames(track)
            ]
            if analysis is not None and analysis.get("status") != "failed"
        ]
        if not analyses:
            return None

        colors = []
        for analysis in analyses:
            colors.extend(analysis.get("dominant_colors", []))
        if not colors:
            return None

        return {
            "status": "ok",
            "samples_count": len(analyses),
            "dominant_colors": self._merge_color_samples(colors),
            "samples": analyses,
        }

    def _sample_track_frames(self, track: dict[str, Any]) -> list[dict[str, Any]]:
        frames = [
            frame
            for frame in track.get("frames", [])
            if frame.get("bbox_xyxy")
        ]
        if len(frames) <= 3:
            return frames
        return [
            frames[0],
            frames[len(frames) // 2],
            frames[-1],
        ]

    def _analyze_kit_references(
        self,
        kit_references: dict[str, Any],
    ) -> dict[str, Any]:
        primary_image = kit_references.get("primary_team_primary_kit_image")
        alternate_image = kit_references.get("primary_team_alternate_kit_image")
        selected_image = kit_references.get("primary_team_selected_kit_image")

        return {
            "primary_team_primary_kit": self._analyze_reference_image(primary_image),
            "primary_team_alternate_kit": self._analyze_reference_image(alternate_image),
            "primary_team_selected_kit": self._analyze_reference_image(selected_image),
            "opponent_team_kit": self._analyze_reference_image(
                kit_references.get("opponent_team_kit_image")
            ),
        }

    def _analyze_reference_image(
        self,
        object_name: str | None,
    ) -> dict[str, Any] | None:
        if not object_name:
            return None

        try:
            image = self._load_image_from_minio(object_name)
            colors = self._extract_dominant_colors(image)
            return {
                "object_name": object_name,
                "status": "ok",
                "dominant_colors": colors,
            }
        except Exception as exc:
            return {
                "object_name": object_name,
                "status": "failed",
                "error": str(exc),
                "dominant_colors": [],
            }

    def _analyze_track_jersey_crop(
        self,
        track_frame: dict[str, Any],
        frame_lookup: dict[int, dict[str, Any]],
    ) -> dict[str, Any] | None:
        try:
            import cv2

            frame_index = track_frame.get("frame_index")
            frame = frame_lookup.get(frame_index)
            if frame is None or not frame.get("image_path"):
                return None

            image = cv2.imread(frame["image_path"])
            if image is None:
                return None

            crop = self._crop_jersey_region(image, track_frame.get("bbox_xyxy"))
            if crop is None:
                return None

            colors = self._extract_dominant_colors(crop, clusters=2)
            return {
                "frame_index": frame_index,
                "bbox_xyxy": track_frame.get("bbox_xyxy"),
                "dominant_colors": colors,
                "crop_path": track_frame.get("jersey_crop_path"),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
            }

    def _crop_jersey_region(
        self,
        image,
        bbox_xyxy: list[float] | None,
    ):
        if not bbox_xyxy:
            return None

        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        box_height = y2 - y1
        jersey_y1 = y1 + int(box_height * 0.18)
        jersey_y2 = y1 + int(box_height * 0.62)
        crop = image[jersey_y1:jersey_y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _assign_from_crop_color(
        self,
        crop_analysis: dict[str, Any] | None,
        primary_reference: dict[str, Any] | None,
        opponent_reference: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not crop_analysis or crop_analysis.get("status") == "failed":
            return None
        if not primary_reference or primary_reference.get("status") != "ok":
            return None

        crop_colors = crop_analysis.get("dominant_colors") or []
        primary_colors = primary_reference.get("dominant_colors") or []
        opponent_colors = (
            opponent_reference.get("dominant_colors") or []
            if opponent_reference and opponent_reference.get("status") == "ok"
            else []
        )
        if not crop_colors or not primary_colors:
            return None

        primary_distance = min(
            self._color_distance(crop_color["rgb"], reference_color["rgb"])
            for crop_color in crop_colors
            for reference_color in primary_colors
        )

        opponent_distance = None
        if opponent_colors:
            opponent_distance = min(
                self._color_distance(crop_color["rgb"], reference_color["rgb"])
                for crop_color in crop_colors
                for reference_color in opponent_colors
            )

        if opponent_distance is not None:
            margin = abs(primary_distance - opponent_distance)
            if margin < 18:
                return {
                    "team_context": "unknown",
                    "confidence": 0.2,
                    "kit_match_score": round(max(0.0, 1 - min(primary_distance, opponent_distance) / 255), 4),
                }

            if primary_distance < opponent_distance:
                kit_match_score = round(max(0.0, 1 - primary_distance / 255), 4)
                return {
                    "team_context": "primary_team",
                    "confidence": round(max(0.45, min(0.95, kit_match_score)), 4),
                    "kit_match_score": kit_match_score,
                }

            kit_match_score = round(max(0.0, 1 - opponent_distance / 255), 4)
            return {
                "team_context": "opponent_team",
                "confidence": round(max(0.45, min(0.95, kit_match_score)), 4),
                "kit_match_score": kit_match_score,
            }

        if primary_distance <= 85:
            kit_match_score = round(max(0.0, 1 - primary_distance / 255), 4)
            return {
                "team_context": "primary_team",
                "confidence": round(max(0.45, kit_match_score), 4),
                "kit_match_score": kit_match_score,
            }

        if primary_distance >= 125:
            kit_match_score = round(min(1.0, primary_distance / 255), 4)
            return {
                "team_context": "opponent_team",
                "confidence": round(min(0.85, kit_match_score), 4),
                "kit_match_score": kit_match_score,
            }

        return None

    def _color_distance(
        self,
        first: list[int],
        second: list[int],
    ) -> float:
        return sum(
            (first[index] - second[index]) ** 2
            for index in range(3)
        ) ** 0.5

    def _load_image_from_minio(self, object_name: str):
        import cv2
        import numpy as np

        response = client.get_object(BUCKET_NAME, object_name)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()

        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode kit reference image")
        return image

    def _extract_dominant_colors(
        self,
        image,
        clusters: int = 3,
    ) -> list[dict[str, Any]]:
        import cv2
        import numpy as np

        resized = cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape((-1, 3)).astype(np.float32)

        # Drop very bright background-like pixels so white backdrops do not dominate.
        brightness = pixels.mean(axis=1)
        filtered = pixels[brightness < 245]
        if len(filtered) >= clusters * 10:
            pixels = filtered

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0,
        )
        _, labels, centers = cv2.kmeans(
            pixels,
            clusters,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )

        counts = np.bincount(labels.flatten(), minlength=clusters)
        total = int(counts.sum()) or 1
        ordered = sorted(
            zip(centers, counts),
            key=lambda item: int(item[1]),
            reverse=True,
        )

        return [
            {
                "rgb": [int(channel) for channel in center],
                "hex": self._rgb_to_hex(center),
                "coverage": round(int(count) / total, 4),
            }
            for center, count in ordered
        ]

    def _merge_color_samples(
        self,
        colors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            colors,
            key=lambda color: float(color.get("coverage") or 0),
            reverse=True,
        )
        merged: list[dict[str, Any]] = []
        for color in ordered:
            rgb = color.get("rgb")
            if not rgb:
                continue
            if any(
                self._color_distance(rgb, existing["rgb"]) < 35
                for existing in merged
            ):
                continue
            merged.append(color)
            if len(merged) >= 4:
                break
        return merged

    def _rgb_to_hex(self, color) -> str:
        red, green, blue = [int(channel) for channel in color]
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _infer_team_context(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        if track.get("class_name") == "ball":
            return "ball"

        if context.get("match_type") in {"internal_scrimmage", "academy_match"}:
            return "club_internal"

        object_key = track.get("object_key") or ""
        if object_key.endswith("left"):
            return "primary_team"
        if object_key.endswith("right"):
            return "opponent_team"
        return "unknown"
