from fastapi import APIRouter

from app.ai.detectors.yolo_detector import YOLODetector

router = APIRouter()


@router.get("/yolo/status")
def get_yolo_status():
    detector = YOLODetector()
    return detector.health()
