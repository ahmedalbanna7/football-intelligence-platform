from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_NAME: str

    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    REDIS_URL: str
    RABBITMQ_URL: str

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str

    YOLO_MODE: str = "auto"
    YOLO_MODEL_PATH: str = "yolo11n.pt"
    YOLO_CONFIDENCE: float = 0.25
    YOLO_IMAGE_SIZE: int = 640
    YOLO_DEVICE: str = "cpu"
    YOLO_MAX_DETECTIONS: int = 300
    YOLO_CLASSES: str = "person,sports ball"
    YOLO_BATCH_SIZE: int = 8
    YOLO_ARTIFACT_SAMPLE_LIMIT: int = 25

    FRAME_SAMPLE_RATE: int = 45
    FRAME_MAX_FRAMES: int = 120

    TRACKER_HIGH_THRESH: float = 0.25
    TRACKER_LOW_THRESH: float = 0.10
    TRACKER_NEW_TRACK_THRESH: float = 0.25
    TRACKER_MATCH_THRESH: float = 0.80
    TRACKER_BUFFER: int = 30
    TRACKER_MIN_PLAYER_OBSERVATIONS: int = 3
    TRACKER_MERGE_GAP_FRAMES: int = 120
    TRACKER_MERGE_DISTANCE: float = 140.0

    SAVE_PLAYER_CROPS: bool = True
    CROP_EVERY_N_FRAMES: int = 10
    MAX_CROPS_PER_TRACK: int = 8

    FIRST_ANALYSIS_MAX_FRAMES: int = 450
    MATCH_ANALYSIS_WORKER_URL: str = "http://match-analysis-worker:8010"
    MATCH_ANALYSIS_WORKER_TIMEOUT_SECONDS: int = 3600
    MATCH_ANALYSIS_AUTO_QUEUE_ON_UPLOAD: bool = True
    MATCH_ANALYSIS_DEFAULT_MODE: str = "PLAYER_TRACKING"
    MATCH_ANALYSIS_DEFAULT_MAX_FRAMES: int = 450

    class Config:
        env_file = ".env"


settings = Settings()
