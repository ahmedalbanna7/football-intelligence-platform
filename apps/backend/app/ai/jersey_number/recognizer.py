from typing import Any

from app.services.minio_client import BUCKET_NAME
from app.services.minio_client import client


class JerseyNumberRecognizer:
    name = "jersey_number_ocr_light_v1"

    def recognize(
        self,
        tracks: list[dict[str, Any]],
        match_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = match_context or {}
        enriched_tracks = [
            self._recognize_track(track, context)
            for track in tracks
        ]

        return {
            "status": "ok",
            "data": {
                "tracks": enriched_tracks,
                "tracks_count": len(enriched_tracks),
                "recognized_count": len(
                    [
                        track
                        for track in enriched_tracks
                        if track.get("recognized_shirt_number") is not None
                    ]
                ),
            },
            "meta": {
                "engine": self.name,
                "current_method": "template_matching_on_saved_jersey_crops",
                "fallback_method": "return_null_when_uncertain",
                "match_context": context,
            },
        }

    def _recognize_track(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if track.get("class_name") != "player":
            return {
                **track,
                "recognized_shirt_number": None,
                "shirt_number_confidence": None,
                "shirt_number_source": "not_player",
            }

        number, confidence, source = self._ocr_track(track)
        return {
            **track,
            "recognized_shirt_number": number,
            "shirt_number_confidence": confidence,
            "shirt_number_source": source,
        }

    def _ocr_track(
        self,
        track: dict[str, Any],
    ) -> tuple[int | None, float | None, str]:
        for sample in track.get("crop_samples", [])[:4]:
            object_name = sample.get("jersey_crop_path")
            if not object_name:
                continue
            result = self._ocr_image_object(object_name)
            if result[0] is not None:
                return result
        return None, None, "no_readable_jersey_crop"

    def _ocr_image_object(
        self,
        object_name: str,
    ) -> tuple[int | None, float | None, str]:
        try:
            image = self._load_image_from_minio(object_name)
            return self._template_match_digits(image)
        except Exception:
            return None, None, "ocr_error"

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
            raise ValueError("Unable to decode jersey crop")
        return image

    def _template_match_digits(self, image) -> tuple[int | None, float | None, str]:
        import cv2
        import numpy as np

        if image.size == 0:
            return None, None, "empty_crop"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = 96 / max(gray.shape[:2])
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.equalizeHist(gray)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            8,
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        height, width = binary.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if h < height * 0.18 or w < width * 0.04:
                continue
            if area < 20:
                continue
            candidates.append((x, y, w, h))

        candidates = sorted(candidates, key=lambda item: item[0])[:2]
        if not candidates:
            return None, None, "no_digit_contours"

        digits = []
        scores = []
        templates = self._digit_templates()
        for x, y, w, h in candidates:
            crop = binary[y:y + h, x:x + w]
            crop = cv2.resize(crop, (28, 42), interpolation=cv2.INTER_AREA)
            best_digit = None
            best_score = -1.0
            for digit, template in templates.items():
                score = float(np.corrcoef(crop.flatten(), template.flatten())[0, 1])
                if score > best_score:
                    best_score = score
                    best_digit = digit
            if best_digit is not None and best_score >= 0.18:
                digits.append(str(best_digit))
                scores.append(best_score)

        if not digits:
            return None, None, "low_confidence_digit_match"

        number = int("".join(digits[:2]))
        confidence = round(max(0.0, min(0.92, sum(scores) / len(scores))), 4)
        if confidence < 0.22:
            return None, confidence, "low_confidence_digit_match"
        return number, confidence, "template_ocr_jersey_crop"

    def _digit_templates(self):
        import cv2
        import numpy as np

        templates = {}
        for digit in range(10):
            canvas = np.zeros((42, 28), dtype=np.uint8)
            cv2.putText(
                canvas,
                str(digit),
                (2, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.15,
                255,
                2,
                cv2.LINE_AA,
            )
            templates[digit] = canvas
        return templates
