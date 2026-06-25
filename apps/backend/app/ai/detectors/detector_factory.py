from app.ai.detectors.yolo_detector import YOLODetector


class DetectorFactory:
    @staticmethod
    def create(engine: str = "yolo") -> YOLODetector:
        if engine == "yolo":
            return YOLODetector()

        raise ValueError(f"Unsupported detector engine: {engine}")
