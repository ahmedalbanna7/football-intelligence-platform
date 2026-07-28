# Football Intelligence Model Training

This directory contains the reproducible training and evaluation pipeline for the
versioned model bundle used by Match Analysis +.

The production weights in `apps/backend/models` are never overwritten by a
training command. New weights are written under `training/runs` and remain
candidate models until they pass dataset validation and video regression tests.

## Model bundle

The analysis bundle deliberately uses separate tasks:

| Model | Task | Classes / output |
| --- | --- | --- |
| `football-objects-v2` | Object detection | ball, player, goalkeeper, referee, other |
| `football-ball-v2` | Small-object detection | ball |
| `football-pitch-v2` | Pose/keypoint detection | pitch landmarks |
| `stable-kit-appearance-v3` | Per-match appearance classification | team 1, team 2, official, unknown |

A single weight file should not be trained from the three source datasets. The
ball-only images contain football players without player annotations, and the
pitch dataset uses keypoints rather than detection boxes. Combining the raw
labels would create false negatives and an invalid YOLO task.

Team identity is not trained as permanent color classes. "Red team" and "blue
team" would change meaning from one match to another. The runtime appearance
classifier uses jersey color, HSV histograms, texture, temporal voting, and
stored kit references to assign the two teams after object detection.

The active `production-v2-hybrid` bundle keeps all v1 weights as local
fallbacks. Object and ball v2 weights are the preferred detectors. Pitch v1 and
v2 are both retained because each generalizes differently; a three-frame
geometry gate selects the model with better wide-view coverage, RANSAC inliers,
and reprojection error for the current video.

## GPU training container

Training uses a dedicated Compose profile so the API and analysis worker keep
their existing runtime:

```powershell
docker compose --profile training build model-trainer
docker compose --profile training run --rm model-trainer nvidia-smi
```

When Docker registry downloads are slow, a project-local Windows environment is
supported and remains entirely on the workspace drive:

```powershell
py -3.12 -m venv training/.venv
training\.venv\Scripts\python.exe -m pip install `
  torch==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cu124
training\.venv\Scripts\python.exe -m pip install -r training/requirements.txt
```

## Source datasets

The datasets are public Roboflow Universe projects licensed under CC BY 4.0.
Their attribution and immutable version information are recorded in
`config/datasets.yaml`.

Place downloaded archives in `training/downloads` and extract them into:

```text
training/datasets/
  players-v20/
  player-others-v1/
  ball-v4/
  pitch-v17/
  objects-v2-merged/
  ball-v4-grouped/
  pitch-v17-grouped/
```

Each extracted directory must contain its Roboflow `data.yaml`.

## Validate labels

```powershell
python training/scripts/validate_dataset.py training/datasets/players-v20
python training/scripts/validate_dataset.py training/datasets/ball-v4
python training/scripts/validate_dataset.py training/datasets/pitch-v17 --task pose
```

Validation checks split paths, image-label pairs, class indexes, normalized
coordinates, pose column counts, empty labels, and class distribution.

Build the derived datasets before training:

```powershell
python training/scripts/merge_object_datasets.py `
  --primary training/datasets/players-v20 `
  --auxiliary training/datasets/player-others-v1 `
  --output training/datasets/objects-v2-merged

python training/scripts/resplit_detection_by_group.py `
  training/datasets/ball-v4 `
  training/datasets/ball-v4-grouped `
  --val-groups 2 --test-groups 2

python training/scripts/resplit_detection_by_group.py `
  training/datasets/pitch-v17 `
  training/datasets/pitch-v17-grouped `
  --val-groups 7 --test-groups 7 `
  --group-pattern "^(?P<group>.+)_\d+_png$"
```

`audit_detection_dataset.py` checks out-of-frame boxes, duplicate boxes,
tiny-object distribution, and perceptual leakage between splits.

Create the rare-class training manifest after building `objects-v2-merged`:

```powershell
python training/scripts/build_balanced_manifest.py `
  training/datasets/objects-v2-merged `
  --repeat other=4 --repeat goalkeeper=2 --repeat referee=1
```

Only the training list is repeated. Validation and test images remain untouched.

## Train candidates

The defaults fit a 6 GB NVIDIA GPU conservatively:

```powershell
python training/scripts/train_yolo.py `
  --task detect `
  --data training/datasets/objects-v2-merged/data-balanced.yaml `
  --model yolo11s.pt `
  --name football-objects-v2-candidate `
  --imgsz 960 --batch 4 --epochs 120

python training/scripts/train_yolo.py `
  --task detect `
  --data training/datasets/ball-v4-grouped/data-balanced.yaml `
  --model yolo11s.pt `
  --name football-ball-v2-candidate `
  --imgsz 960 --batch 4 --epochs 120

python training/scripts/train_yolo.py `
  --task pose `
  --data training/datasets/pitch-v17-grouped/data.yaml `
  --model apps/backend/models/football-pitch-detection.pt `
  --name football-pitch-v2-candidate `
  --imgsz 640 --batch 1 --epochs 100
```

Use `--device cpu` only for a smoke test. Full training on CPU is not practical.
The scripts enable checkpointing, early stopping, deterministic seeds, and CSV
metrics.

Run the complete versioned plan sequentially after CUDA verification:

```powershell
training\.venv\Scripts\python.exe training/scripts/train_bundle.py
```

The exact models, datasets, image sizes, and 6 GB GPU batches live in
`config/training-plan.yaml`. Progress is checkpointed in
`training/cache/bundle-training-state.json`.

Verify the active bundle's versioned SHA256 manifest before starting a worker:

```powershell
training\.venv\Scripts\python.exe training/scripts/verify_model_bundle.py
```

## Match-specific hard negatives

Export evenly distributed frames from a local copy of a difficult match:

```powershell
docker compose --profile training run --rm model-trainer `
  python training/scripts/extract_video_samples.py `
  --video training/downloads/match-12.mp4 `
  --output training/datasets/hard-negatives/match-12 `
  --count 120
```

These frames must be annotated or reviewed before training. Empty labels are
valid only when the image genuinely contains none of the target classes.

## Promotion gate

Candidate weights are evaluated on their untouched dataset test splits and on
the same sampled intervals from:

- Match `#12 08fd33_4.mp4`
- the existing academy regression clip
- the Croatia tactical-camera regression clip

Promotion requires:

- no regression in class precision/recall or mAP50-95;
- fewer people outside the playing surface;
- fewer penalty-spot/static-object ball false positives;
- higher on-pitch player coverage;
- stable or improved processing speed;
- stable or improved downstream IDF1/HOTA when ground truth is available;
- valid pitch calibration confidence and reprojection error.

Only then is a candidate copied to `apps/backend/models` under a new versioned
name and activated through the model registry.
