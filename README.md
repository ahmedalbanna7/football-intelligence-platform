# Sports Intelligence Platform

Open-source football video intelligence platform for match upload, computer-vision processing, player tracking, tactical analysis, reporting, and coach-facing workflows.

The project is built as a practical foundation for football clubs, academies, analysts, and developers who want to turn raw match video into structured football intelligence. It combines a FastAPI backend, RabbitMQ workers, MinIO object storage, PostgreSQL data models, YOLO-based detection, tracking, team classification, tactical identity layers, and a React dashboard.

> Status: active development. The platform already includes the core upload, queue, analysis, and dashboard foundation. Some advanced football intelligence modules are intentionally staged and will become more accurate as stronger tracking, OCR, re-identification, and analytics models are added.

## Highlights

- Match video upload API with MinIO storage.
- RabbitMQ-based asynchronous processing.
- Dedicated `Match Analysis +` worker flow.
- YOLO / Ultralytics integration for player and ball observations.
- Stable track ID layer on top of detector/tracker outputs.
- Static-field-marker ball filtering to reduce false ball detections.
- Team profile and opponent team management.
- Player roster, shirt number, tactical zone, and assignment data models.
- First-analysis integration inspired by an external football analysis project.
- Match analysis run history with saved artifacts.
- React + TypeScript dashboard for uploads, teams, analysis, reports, and agent workflows.
- Open-source architecture designed for extension.

## Product Surface

The frontend is organized around the workflows an analyst naturally needs:

- **Dashboard**: high-level operational view.
- **Settings**: platform configuration surface.
- **My Team**: primary club profile, kits, roster, player metadata.
- **Teams**: opponent and external team history, profiles, kits, players, and related matches.
- **Matches**: video upload and match metadata.
- **Analysis**: pipeline outputs, detections, tracks, tactical identity, and processing summaries.
- **First Analysis**: lightweight annotated-video analysis flow.
- **Match Analysis +**: primary worker-based match analysis flow with saved runs.
- **Reports**: JSON/PDF/reporting foundation.
- **Agent**: future coach/analyst assistant surface.
- **Recommendations**: future season, team, player, and match recommendation area.

## Architecture

```mermaid
flowchart LR
    User["Analyst / Coach"] --> Frontend["React + TypeScript Dashboard"]
    Frontend --> API["FastAPI Backend"]
    API --> Postgres["PostgreSQL"]
    API --> MinIO["MinIO Object Storage"]
    API --> RabbitMQ["RabbitMQ"]
    API --> Redis["Redis"]

    RabbitMQ --> MatchWorker["Match Analysis + Worker"]
    RabbitMQ --> VideoWorker["Video Worker (optional profile)"]

    MatchWorker --> YOLO["YOLO / Ultralytics"]
    MatchWorker --> Tracking["Tracking + Stable IDs"]
    MatchWorker --> TeamClassifier["Team Classification"]
    MatchWorker --> Analytics["Analytics + Reports"]
    MatchWorker --> MinIO
    MatchWorker --> Postgres
```

## Match Upload Flow

```mermaid
sequenceDiagram
    participant UI as React Dashboard
    participant API as FastAPI
    participant Storage as MinIO
    participant DB as PostgreSQL
    participant MQ as RabbitMQ
    participant Worker as Match Analysis Worker

    UI->>API: POST /matches/upload
    API->>Storage: Store source video
    API->>DB: Create match + video/run records
    API->>MQ: Publish match_analysis.requested
    MQ->>Worker: Deliver analysis job
    Worker->>Storage: Download source video
    Worker->>Worker: Detect, track, classify, annotate
    Worker->>Storage: Save video + summary artifacts
    Worker->>DB: Update run and match status
    UI->>API: Fetch run history and artifacts
```

## Video Intelligence Pipeline

```mermaid
flowchart TD
    Video["Uploaded Match Video"] --> FrameRead["Frame Reader / OpenCV"]
    FrameRead --> Detector["YOLO Detector"]
    Detector --> RawObjects["Frame Observations"]
    RawObjects --> BallFilter["Ball Static Marker Filter"]
    RawObjects --> Tracker["YOLO/ByteTrack Output"]
    Tracker --> Stabilizer["Stable Track ID Layer"]
    Stabilizer --> TeamClassifier["Team Classifier"]
    TeamClassifier --> TacticalIdentity["Tactical Identity Layer"]
    TacticalIdentity --> Analytics["Analytics Engine"]
    Analytics --> Reports["Reports + Artifacts"]
    Reports --> Dashboard["Frontend Review"]
```

## Data Model Overview

```mermaid
erDiagram
    MATCH ||--o{ MATCH_VIDEO : has
    MATCH ||--o{ MATCH_ANALYSIS_RUN : produces
    MATCH ||--o{ VIDEO_PROCESSING_JOB : queues
    MATCH ||--o{ TRACK_PLAYER_ASSIGNMENT : maps
    MATCH ||--o{ TRACK_IDENTITY_ASSIGNMENT : resolves
    MATCH ||--o{ MATCH_LINEUP : uses
    MATCH ||--o{ MATCH_SUBSTITUTION : includes

    TEAM ||--o{ PLAYER : owns
    TEAM ||--o{ PLAYER_ROSTER_ENTRY : registers
    PRIMARY_TEAM_PROFILE ||--o{ PLAYER_ROSTER_ENTRY : defines

    MATCH_ANALYSIS_RUN {
        int id
        int match_id
        string mode
        string status
        string artifact_prefix
    }

    TRACK_PLAYER_ASSIGNMENT {
        int track_id
        int player_id
        int shirt_number
        string assignment_source
    }
```

## Queue Topology

```mermaid
flowchart LR
    API["FastAPI Publisher"] --> Exchange["RabbitMQ Topic Exchange"]
    Exchange --> VideoQueue["video.processing"]
    Exchange --> MatchQueue["match_analysis.processing"]
    VideoQueue --> VideoWorker["Video Worker"]
    MatchQueue --> MatchWorker["Match Analysis + Worker"]
```

## Stack

### Backend

- **Python 3.12**
- **FastAPI** for HTTP APIs.
- **Uvicorn** for ASGI serving.
- **SQLAlchemy 2** for ORM/data access.
- **Alembic** for database migrations.
- **Pydantic Settings** for environment-driven config.
- **OpenCV** for frame reading, annotation, and video processing.
- **Ultralytics YOLO** for player and ball detection.
- **PyTorch CPU runtime** for model execution.
- **RabbitMQ / aio-pika** for async job dispatching.
- **MinIO Python SDK** for object storage.
- **Redis** as infrastructure foundation for cache/session/future async workflows.
- **LangChain / OpenAI packages** are included for coach/LLM and agent-facing extensions.

### Frontend

- **React 19**
- **TypeScript**
- **Vite**
- **Lucide React** icons
- Plain CSS design system focused on operational dashboard workflows.

### Infrastructure

- **Docker Compose**
- **PostgreSQL 16**
- **RabbitMQ 3 Management**
- **Redis 7**
- **MinIO**
- Optional worker profile for the legacy video pipeline.
- Dedicated `match-analysis-worker` service for `Match Analysis +`.

## Services

| Service | Purpose | Default Port |
| --- | --- | --- |
| `frontend` | React dashboard | `5173` |
| `backend` | FastAPI API | `8000` |
| `postgres` | relational database | dynamic host port |
| `rabbitmq` | queues + management UI | `5672`, `15672` |
| `redis` | cache/future async support | `6379` |
| `minio` | video/artifact object storage | `9000`, `9001` |
| `match-analysis-worker` | primary match analysis worker | internal |
| `video-worker` | optional legacy pipeline worker profile | internal |

## Key API Areas

| Area | Example Routes |
| --- | --- |
| Matches | `POST /matches/upload`, `GET /matches`, `DELETE /matches/{id}` |
| Artifacts | `GET /matches/artifacts/object` |
| Processing | `GET /matches/{id}/processing` |
| Reports | `GET /matches/{id}/report`, `GET /matches/{id}/report.pdf` |
| Primary team | `GET /primary-team`, `POST /primary-team`, `POST /primary-team/players` |
| Teams | `GET /teams`, `POST /teams`, `POST /teams/{id}/players` |
| First analysis | `POST /first-analysis/{match_id}/run`, `GET /first-analysis/{match_id}` |
| Match Analysis + | `POST /match-analysis-plus/{match_id}/run`, `GET /match-analysis-plus/{match_id}` |
| AI status | `GET /ai/yolo/status` |

## Repository Layout

```text
.
├── apps
│   ├── backend
│   │   ├── app
│   │   │   ├── ai
│   │   │   ├── match_analysis_plus
│   │   │   ├── modules
│   │   │   ├── pipeline
│   │   │   ├── queues
│   │   │   ├── services
│   │   │   └── workers
│   │   ├── alembic
│   │   └── football_analysis-main
│   ├── frontend
│   │   └── src
│   └── match-analysis-worker
│       └── sports-main
├── docker-compose.yml
└── README.md
```

## Local Development

### 1. Create backend environment file

Create `apps/backend/.env`:

```env
APP_NAME=Sports Intelligence Platform

POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=sports
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

REDIS_URL=redis://redis:6379/0
RABBITMQ_URL=amqp://guest:guest@rabbitmq/

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=minio123

YOLO_MODE=auto
YOLO_MODEL_PATH=yolo11n.pt
YOLO_CONFIDENCE=0.25
YOLO_IMAGE_SIZE=640
YOLO_DEVICE=cpu

MATCH_ANALYSIS_AUTO_QUEUE_ON_UPLOAD=true
MATCH_ANALYSIS_DEFAULT_MODE=PLAYER_TRACKING
MATCH_ANALYSIS_DEFAULT_MAX_FRAMES=450
```

### 2. Start the platform

```bash
docker compose up -d --build
```

### 3. Open the dashboard

- Frontend: <http://localhost:5173>
- Backend API: <http://localhost:8000>
- Swagger docs: <http://localhost:8000/docs>
- RabbitMQ UI: <http://localhost:15672>
- MinIO console: <http://localhost:9001>

### 4. Run a quick analysis

1. Open `Matches`.
2. Upload a video.
3. Open `Match Analysis +`.
4. Select the uploaded match.
5. Run analysis with a small `max_frames` value first.
6. Review the generated annotated video, summary, tracks, and run history.

## Current Analysis Capabilities

```mermaid
mindmap
  root((Football Intelligence))
    Detection
      Players
      Ball
      YOLO model status
    Tracking
      Tracklets
      Stable track ID layer
      Manual assignment foundation
    Team Understanding
      My team profile
      Opponent profiles
      Kit image references
      Jersey color classifier
    Tactical Identity
      Roster links
      Shirt number metadata
      Zone model
      Confidence reasons
    Analytics
      Distance
      Speed
      Movement summaries
      Ball control foundation
    Reports
      JSON summaries
      PDF endpoint foundation
      Future charts and heatmaps
```

## Roadmap

```mermaid
gantt
    title Football Analysis Roadmap
    dateFormat  YYYY-MM-DD
    section Foundation
    Upload + storage + queues       :done, f1, 2026-06-01, 7d
    React dashboard                 :done, f2, 2026-06-08, 10d
    Match Analysis + worker          :done, f3, 2026-06-18, 10d
    section Vision
    Stronger tracking / ReID         :active, v1, 2026-07-01, 21d
    Jersey OCR                       :v2, after v1, 21d
    TeamClassifier validation        :v3, after v1, 14d
    section Intelligence
    Real analytics v1                :i1, after v2, 21d
    Reports v2 charts and heatmaps   :i2, after i1, 21d
    LLM coach recommendations        :i3, after i1, 21d
    section Live
    RTSP/live processing             :l1, after i2, 30d
```

## Development Philosophy

This project is intentionally modular:

- The queue layer keeps upload APIs fast and worker execution isolated.
- The analysis worker can evolve independently from the backend API.
- Detection, tracking, team classification, identity, analytics, reports, and LLM coaching are separate layers.
- Early modules can use stubs or lightweight heuristics while the public API remains stable.
- Each layer is designed to become more intelligent without breaking the surrounding workflow.

## Known Limitations

- Default YOLO models are general-purpose and may confuse field markers, logos, or white spots with a ball.
- Stable track IDs are improved by a custom stabilization layer, but long occlusions still require stronger player re-identification.
- Jersey number OCR is foundational and needs more validation on real kit crops.
- Some analytics and reports are still early-stage and will become more accurate as tracking improves.
- CPU inference is supported, but GPU inference is recommended for long videos and production workloads.

## Contributing

Contributions are welcome. Useful areas include:

- Better player re-identification.
- Football-specific object detection models.
- Jersey number OCR.
- Ball tracking validation.
- Team classification from kit images and crops.
- Tactical analytics.
- Report charts, heatmaps, and visualizations.
- Live RTSP processing.
- Frontend review tools for analysts.

## License

This project is open-source under the MIT License. See [LICENSE](LICENSE).
