# Runtime and Dependency Audit

This audit separates production requirements from optional development tools
and confirmed unused coupling. It intentionally avoids deleting a future-facing
dependency merely because the current release does not call it.

## Required for the current application

- `frontend`, `backend`, and `match-analysis-worker` serve the UI, API, and
  football analysis pipeline.
- `postgres` stores matches, runs, corrections, reports, and quality metadata.
- `rabbitmq` carries durable video and match-analysis jobs.
- `minio` stores uploaded videos, model artifacts, rendered videos, detections,
  visual layers, and reports.
- `migrate` is a one-shot schema migration service and must run before the API.
- `model-setup` is a one-shot model downloader. It is required for a new
  collaborator who does not already have the `.pt` files.
- `video-worker` is required when uploaded videos are processed asynchronously.
  Its Compose profile only controls when it is started.
- All football `.pt` model candidates referenced by the model registry are
  intentional fallbacks or specialized models; none should be deleted from a
  collaborator release without updating the registry and model manifest.

## Optional but intentionally retained

- `model-trainer` and `training/` are not needed to run inference. They are
  retained for future dataset preparation, retraining, and model export.
- Redis is not called by current application code. Only `REDIS_URL`, Compose
  service declarations, and future-infrastructure documentation reference it.
  It can be removed today if cache, sessions, rate limiting, and future Redis
  workflows are out of scope. Keep it if any of those roadmap items remain.
- LangChain, LangGraph, ChromaDB, OpenAI, ONNX Runtime, and the Python Redis
  client have no direct imports under `apps/backend/app` in this release. They
  are candidates for a dedicated requirements-slimming pass, but should be
  removed only with a clean Docker rebuild because the current requirements
  file also contains transitive packages.

## Confirmed unused runtime coupling

- `apps/match-analysis-worker/sports-main` is a 3.7 MB vendored reference tree.
  The worker imports the native runner from
  `apps/backend/app/match_analysis_plus`; no production module imports the
  vendored package. Its `/opt/sports-main` volume and extra `PYTHONPATH` entry
  are unnecessary for runtime. The source tree can be archived or deleted once
  its license/reference value is no longer wanted.
- Run labels and summary notes still use the historical `sports-main` name.
  They are metadata, not executable dependencies, and can be migrated in a
  separate compatibility-safe cleanup.
- `.codex-*.json`, `.codex-*.jsonl`, and generated local validation layers are
  temporary evidence exports. They are not application inputs and must not be
  committed.
- Docker build cache is disposable. The latest audit found about 15 GB
  reclaimable with `docker builder prune`; pruning affects rebuild speed, not
  source code or runtime data. Named Postgres and MinIO volumes are not build
  cache and must not be deleted.

## Generalization checks

The runtime source contains no hardcoded references to the validated match
filenames, match IDs, player/goalkeeper IDs, or critical frame numbers. The
release suite covers five verified identity cases across 360p tactical, 1080p
tactical, and 4K close/moving video. Ball tracking uses normalized geometry and
confidence gates; measured cross-video ball accuracy still requires additional
verified ball labels.
