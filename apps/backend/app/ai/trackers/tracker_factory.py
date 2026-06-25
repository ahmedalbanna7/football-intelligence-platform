from app.ai.trackers.bytetrack_tracker import ByteTrackTracker


class TrackerFactory:
    @staticmethod
    def create(engine: str = "bytetrack") -> ByteTrackTracker:
        if engine == "bytetrack":
            return ByteTrackTracker()

        raise ValueError(f"Unsupported tracker engine: {engine}")
