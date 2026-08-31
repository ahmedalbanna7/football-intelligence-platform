# Match Analysis Performance

The platform keeps accuracy-sensitive tracking ordered while accelerating the
parts that can be reused safely.

## Detection cache

Every completed Match Analysis run stores the raw detector observations as
`performance/detections.jsonl`. A later run can select **Reuse detections from
run #...** to skip player and ball YOLO inference while rebuilding tracking,
team identity, pitch calibration, analytics, rendering, and reports.

The cache is accepted only for the same uploaded video and requested frame
range. A partial cache is rejected instead of silently mixing incompatible
observations.

## CPU mode

The default Compose file builds the CPU PyTorch wheel and runs with:

```bash
docker compose up -d --build
```

This is the portable development profile. Long videos are expected to be slow.

## NVIDIA GPU mode

Requirements:

- NVIDIA GPU with a compatible driver
- Docker Desktop GPU support / NVIDIA Container Toolkit
- Enough VRAM for the selected model and image size

Build and start the CUDA worker with:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build match-analysis-worker
```

The override installs the CUDA 12.4 PyTorch wheels, exposes all GPUs to the
analysis worker, and sets `YOLO_DEVICE=0`. The API and frontend can remain on
their normal CPU images.

Verify the worker runtime:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec match-analysis-worker python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Why tracking batch size remains one

Stable identity association is stateful and frame ordered. Batching arbitrary
tracking frames would change the temporal contract and can increase identity
switches. The run summary therefore reports a tracking batch size of one while
keeping `YOLO_BATCH_SIZE` available for independent auxiliary inference and
future detector precomputation.

## Progress and diagnostics

Queued and processing runs expose stage, processed frames, percentage, current
FPS, cache hits, and ETA through `analysis_config.runtime_progress`. The
frontend polls active runs and displays the same values in Run Details.

The final summary records:

- configured device and CUDA availability
- cache hits and actual YOLO inference frames
- detector and rendering time
- processing FPS
- reusable detection-cache object
