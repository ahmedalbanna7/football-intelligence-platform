from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

from app.core.config import settings


class YOLODetector:
    name = "yolo_detector"
    stub_name = "yolo_stub_v2"
    real_name = "ultralytics_yolo"
    model_name = settings.YOLO_MODEL_PATH
    classes = {
        0: "player",
        1: "ball",
        2: "referee",
        3: "goalkeeper",
    }

    def __init__(
        self,
        mode: str | None = None,
        model_path: str | None = None,
        confidence: float | None = None,
        image_size: int | None = None,
        device: str | None = None,
        max_detections: int | None = None,
        allowed_classes: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.mode = (mode or settings.YOLO_MODE).lower().strip()
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.confidence = confidence or settings.YOLO_CONFIDENCE
        self.image_size = image_size or settings.YOLO_IMAGE_SIZE
        self.device = device or settings.YOLO_DEVICE
        self.max_detections = max_detections or settings.YOLO_MAX_DETECTIONS
        self.batch_size = batch_size or settings.YOLO_BATCH_SIZE
        self.allowed_class_names = self._parse_allowed_classes(
            allowed_classes or settings.YOLO_CLASSES
        )
        self._model = None
        self._model_error: str | None = None
        if self.mode not in {"auto", "real", "stub"}:
            self._model_error = f"Unsupported YOLO_MODE={self.mode}; using stub"
            self.mode = "stub"

    def detect(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        started_at = perf_counter()
        if self.mode in {"real", "auto"}:
            real_result = self._detect_with_ultralytics(frames, started_at)
            if real_result is not None:
                return real_result

            if self.mode == "real":
                return {
                    "status": "failed",
                    "data": {
                        "detections": [],
                        "frames_processed": len(frames),
                        "detections_count": 0,
                    },
                    "meta": {
                        "engine": self.real_name,
                        "model": self.model_path,
                        "mode": self.mode,
                        "error": self._model_error,
                        "elapsed_ms": self._elapsed_ms(started_at),
                    },
                }

        return self._detect_with_stub(frames, started_at)

    def _detect_with_ultralytics(
        self,
        frames: list[dict[str, Any]],
        started_at: float,
    ) -> dict[str, Any] | None:
        model = self._load_model()
        if model is None:
            return None

        detections: list[dict[str, Any]] = []
        frames_with_images = [
            frame
            for frame in frames
            if frame.get("image_path")
        ]
        image_paths = [
            frame["image_path"]
            for frame in frames_with_images
        ]

        if not image_paths:
            self._model_error = "No extracted frame image_path values available"
            return None

        try:
            results = model.predict(
                source=image_paths,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                max_det=self.max_detections,
                batch=self.batch_size,
                verbose=False,
            )
        except Exception as exc:
            self._model_error = str(exc)
            return None

        for frame, result in zip(frames_with_images, results, strict=False):
            detections.extend(self._convert_result(frame, result))

        summary = self._summarize_detections(detections)

        return {
            "status": "ok",
            "data": {
                "detections": detections,
                "frames_processed": len(frames_with_images),
                "frames_requested": len(frames),
                "frames_skipped": len(frames) - len(frames_with_images),
                "detections_count": len(detections),
                "class_counts": summary["class_counts"],
                "raw_class_counts": summary["raw_class_counts"],
                "confidence": summary["confidence"],
            },
            "meta": {
                "engine": self.real_name,
                "model": self.model_path,
                "mode": self.mode,
                "confidence": self.confidence,
                "image_size": self.image_size,
                "device": self.device,
                "max_detections_per_frame": self.max_detections,
                "allowed_class_names": sorted(self.allowed_class_names),
                "classes": self.classes,
                "model_loaded": True,
                "model_file_exists": self._model_file_exists(),
                "elapsed_ms": self._elapsed_ms(started_at),
                "output_contract": {
                    "bbox_xyxy": "pixel coordinates [x1, y1, x2, y2]",
                    "bbox_normalized": "relative coordinates in [0, 1]",
                    "center": "pixel coordinates [x, y]",
                },
            },
        }

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            return self._model
        except Exception as exc:
            self._model_error = str(exc)
            return None

    def _convert_result(
        self,
        frame: dict[str, Any],
        result,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        names = getattr(result, "names", {}) or {}
        width = int(frame.get("width") or 1)
        height = int(frame.get("height") or 1)
        frame_index = int(frame.get("frame_index", 0))
        timestamp_seconds = frame.get("timestamp_seconds")
        timestamp_ms = (
            round(float(timestamp_seconds) * 1000, 2)
            if timestamp_seconds is not None
            else None
        )

        for detection_index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            raw_class_name = names.get(class_id, str(class_id))
            if not self._is_allowed_raw_class(raw_class_name):
                continue

            mapped_class = self._map_class(raw_class_name)
            if mapped_class is None:
                continue

            confidence = float(box.conf[0].item())
            xyxy = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]
            x1, y1, x2, y2 = xyxy
            bbox_width = max(x2 - x1, 0)
            bbox_height = max(y2 - y1, 0)
            bbox_center = [
                round((x1 + x2) / 2, 2),
                round((y1 + y2) / 2, 2),
            ]
            bbox_area = round(bbox_width * bbox_height, 2)

            detections.append(
                {
                    "id": f"{frame_index}-{detection_index}",
                    "detector": self.real_name,
                    "source": "yolo",
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "timestamp_ms": timestamp_ms,
                    "class_id": self._contract_class_id(mapped_class),
                    "raw_class_id": class_id,
                    "raw_class_name": raw_class_name,
                    "class_name": mapped_class,
                    "object_role": mapped_class,
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                    "bbox_xywh": [
                        round(x1, 2),
                        round(y1, 2),
                        round(bbox_width, 2),
                        round(bbox_height, 2),
                    ],
                    "bbox_center": bbox_center,
                    "bbox_area": bbox_area,
                    "normalized_bbox": [
                        round(x1 / width, 6),
                        round(y1 / height, 6),
                        round(x2 / width, 6),
                        round(y2 / height, 6),
                    ],
                    "bbox_normalized": [
                        round(x1 / width, 6),
                        round(y1 / height, 6),
                        round(x2 / width, 6),
                        round(y2 / height, 6),
                    ],
                    "center": bbox_center,
                    "area": bbox_area,
                }
            )

        return detections

    def _map_class(self, raw_class_name: str) -> str | None:
        normalized = raw_class_name.lower().strip()
        if normalized in {"person", "player", "football player", "soccer player"}:
            return "player"
        if normalized in {"sports ball", "ball", "soccer ball", "football"}:
            return "ball"
        if normalized in {"referee"}:
            return "referee"
        if normalized in {"goalkeeper", "keeper"}:
            return "goalkeeper"
        return None

    def _contract_class_id(self, class_name: str) -> int:
        for class_id, name in self.classes.items():
            if name == class_name:
                return class_id
        return -1

    def _detect_with_stub(
        self,
        frames: list[dict[str, Any]],
        started_at: float | None = None,
    ) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []

        for frame in frames:
            detections.extend(self._detect_frame(frame))

        summary = self._summarize_detections(detections)

        return {
            "status": "ok",
            "data": {
                "detections": detections,
                "frames_processed": len(frames),
                "frames_requested": len(frames),
                "frames_skipped": 0,
                "detections_count": len(detections),
                "class_counts": summary["class_counts"],
                "raw_class_counts": summary["raw_class_counts"],
                "confidence": summary["confidence"],
            },
            "meta": {
                "engine": self.stub_name,
                "model": "stub-yolo-football-v1",
                "classes": self.classes,
                "mode": "deterministic_stub",
                "fallback_reason": self._model_error,
                "model_file_exists": self._model_file_exists(),
                "elapsed_ms": self._elapsed_ms(started_at),
                "output_contract": {
                    "bbox_xyxy": "pixel coordinates [x1, y1, x2, y2]",
                    "bbox_normalized": "relative coordinates in [0, 1]",
                },
            },
        }

    def _detect_frame(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        frame_index = int(frame.get("frame_index", 0))
        timestamp_seconds = frame.get("timestamp_seconds")
        width = int(frame.get("width") or 1920)
        height = int(frame.get("height") or 1080)

        player_a_x = 0.18 + ((frame_index // 30) % 20) * 0.01
        player_b_x = 0.62 - ((frame_index // 30) % 16) * 0.008
        ball_x = (player_a_x + player_b_x) / 2

        specs = [
            {
                "class_id": 0,
                "class_name": "player",
                "confidence": 0.88,
                "center": [player_a_x, 0.56],
                "size": [0.055, 0.18],
                "stub_object_id": "player-left",
            },
            {
                "class_id": 0,
                "class_name": "player",
                "confidence": 0.84,
                "center": [player_b_x, 0.52],
                "size": [0.055, 0.18],
                "stub_object_id": "player-right",
            },
            {
                "class_id": 1,
                "class_name": "ball",
                "confidence": 0.72,
                "center": [ball_x, 0.49],
                "size": [0.022, 0.022],
                "stub_object_id": "ball",
            },
        ]

        return [
            self._build_detection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                width=width,
                height=height,
                detection_index=index,
                spec=spec,
            )
            for index, spec in enumerate(specs)
        ]

    def _build_detection(
        self,
        frame_index: int,
        timestamp_seconds: float | None,
        width: int,
        height: int,
        detection_index: int,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        center_x, center_y = spec["center"]
        box_width, box_height = spec["size"]

        x1_norm = max(center_x - box_width / 2, 0)
        y1_norm = max(center_y - box_height / 2, 0)
        x2_norm = min(center_x + box_width / 2, 1)
        y2_norm = min(center_y + box_height / 2, 1)

        bbox_xyxy = [
            round(x1_norm * width, 2),
            round(y1_norm * height, 2),
            round(x2_norm * width, 2),
            round(y2_norm * height, 2),
        ]
        bbox_width = round(bbox_xyxy[2] - bbox_xyxy[0], 2)
        bbox_height = round(bbox_xyxy[3] - bbox_xyxy[1], 2)
        bbox_center = [
            round((bbox_xyxy[0] + bbox_xyxy[2]) / 2, 2),
            round((bbox_xyxy[1] + bbox_xyxy[3]) / 2, 2),
        ]
        bbox_area = round(max(bbox_width, 0) * max(bbox_height, 0), 2)
        timestamp_ms = (
            round(float(timestamp_seconds) * 1000, 2)
            if timestamp_seconds is not None
            else None
        )

        return {
            "id": f"{frame_index}-{detection_index}",
            "detector": self.stub_name,
            "source": "yolo_stub",
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "timestamp_ms": timestamp_ms,
            "class_id": spec["class_id"],
            "class_name": spec["class_name"],
            "object_role": spec["class_name"],
            "confidence": spec["confidence"],
            "bbox_xyxy": bbox_xyxy,
            "bbox_xywh": [
                bbox_xyxy[0],
                bbox_xyxy[1],
                bbox_width,
                bbox_height,
            ],
            "bbox_center": bbox_center,
            "bbox_area": bbox_area,
            "normalized_bbox": [
                round(x1_norm, 6),
                round(y1_norm, 6),
                round(x2_norm, 6),
                round(y2_norm, 6),
            ],
            "bbox_normalized": [
                round(x1_norm, 6),
                round(y1_norm, 6),
                round(x2_norm, 6),
                round(y2_norm, 6),
            ],
            "center": bbox_center,
            "area": bbox_area,
            "stub_object_id": spec["stub_object_id"],
        }

    def health(self) -> dict[str, Any]:
        model = self._load_model() if self.mode in {"real", "auto"} else None
        return {
            "status": "ok" if self.mode == "stub" or model is not None else "failed",
            "engine": self.real_name if model is not None else self.stub_name,
            "mode": self.mode,
            "model": self.model_path,
            "model_file_exists": self._model_file_exists(),
            "model_loaded": model is not None,
            "error": self._model_error,
            "confidence": self.confidence,
            "image_size": self.image_size,
            "device": self.device,
                "max_detections_per_frame": self.max_detections,
                "batch_size": self.batch_size,
                "allowed_class_names": sorted(self.allowed_class_names),
            "classes": self.classes,
        }

    def _parse_allowed_classes(self, value: str) -> set[str]:
        classes = {
            item.strip().lower()
            for item in value.split(",")
            if item.strip()
        }
        return classes or {"person", "sports ball"}

    def _is_allowed_raw_class(self, raw_class_name: str) -> bool:
        normalized = raw_class_name.lower().strip()
        return normalized in self.allowed_class_names

    def _summarize_detections(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        class_counts = Counter(
            detection.get("class_name", "unknown")
            for detection in detections
        )
        raw_class_counts = Counter(
            detection.get("raw_class_name", detection.get("class_name", "unknown"))
            for detection in detections
        )
        confidences = [
            float(detection.get("confidence", 0))
            for detection in detections
        ]
        return {
            "class_counts": dict(class_counts),
            "raw_class_counts": dict(raw_class_counts),
            "confidence": {
                "min": round(min(confidences), 6) if confidences else None,
                "max": round(max(confidences), 6) if confidences else None,
                "avg": (
                    round(sum(confidences) / len(confidences), 6)
                    if confidences
                    else None
                ),
            },
        }

    def _model_file_exists(self) -> bool:
        path = Path(self.model_path)
        if path.is_absolute():
            return path.exists()
        return Path.cwd().joinpath(path).exists()

    def _elapsed_ms(self, started_at: float | None) -> float | None:
        if started_at is None:
            return None
        return round((perf_counter() - started_at) * 1000, 2)
