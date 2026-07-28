# Model Bundle Evaluation

The `production-v2-hybrid` bundle was trained from isolated, group-clean
Roboflow exports and compared with the original local production weights.
Generated JSON reports live under `training/cache/evaluations` and
`training/cache/benchmarks`.

## Dataset test splits

| Component | Model | mAP50 | mAP50-95 | Inference |
| --- | --- | ---: | ---: | ---: |
| Football roles | v1 | 0.746 | 0.366 | 130.8 ms/image |
| Football roles | v2 | **0.859** | **0.569** | **22.3 ms/image** |
| Ball | v1 | 0.382 | 0.163 | 131.8 ms/image |
| Ball | v2 | **0.859** | **0.524** | **20.1 ms/image** |
| Pitch pose | v1 | 0.057 pose mAP50 | 0.012 pose mAP50-95 | 73.2 ms/image |
| Pitch pose | v2 | **0.899 pose mAP50** | **0.554 pose mAP50-95** | 73.8 ms/image |

The role comparison uses the untouched four-class source test split so the v1
model is not penalized for the new `other` class it was never trained to emit.

## Match 12 video regression

The local `#12 08fd33_4.mp4` copy contains 750 frames. Component benchmarks
scan up to 1,000 source frames and therefore cover the complete clip.

- Objects v2 selected 34 valid participant candidates in the first preview
  frame, versus 22 for v1 and 27 generic `person` detections.
- Visual inspection confirmed that v2 boxes belong to people. The metric
  pitch-occupancy layer remains responsible for removing touchline personnel
  before stable identity assignment.
- Ball v2 found substantially more true candidates and runs faster, but may
  produce multiple raw candidates. Runtime keeps only one ball through the
  static-marker, penalty-spot, motion, player-proximity, and Kalman layers.
- Pitch v1 remained more stable on this specific tactical-camera clip:
  approximately 43 cm median preview reprojection error versus 92 cm for v2.
  The runtime geometry gate correctly retained v1 for this video.

Pitch v2 is not discarded. Its isolated test result is much stronger and it can
generalize better to other camera layouts. Runtime evaluates v1 and v2 on three
preview frames and chooses by wide-view coverage, homography success, RANSAC
inliers, and reprojection error. The selected model and all candidate scores are
saved in every analysis summary.

## Team separation

Team assignment is intentionally outside the detector:

1. Detect players, goalkeepers, referees, and the ball.
2. Load the match's stored primary and alternate kit images when available.
3. Extract background-resistant jersey palettes.
4. Compare player torso crops by color, HSV histogram, and texture.
5. Stabilize the team label through per-track temporal voting.
6. Exclude referees from team anchors and treat goalkeepers as separate roles.
7. Fall back to online two-team appearance clustering when kit references are
   missing.

Training fixed classes such as `red_team` and `blue_team` would assign the wrong
identity whenever clubs change kits, so it is not used.

## Docker end-to-end acceptance

The generated weights are versioned and v1 weights remain intact. Docker
acceptance was completed against Match 12 because BoT-SORT depends on `lap`,
which is part of the backend image:

```powershell
docker compose up -d backend frontend rabbitmq minio postgres redis match-analysis-worker
docker compose exec match-analysis-worker python -m unittest test_track_id_stabilizer.py test_match_analysis_worker.py
```

Run 47 covered the complete 750-frame Match 12 clip:

- Objects v2 and Ball v2 were selected.
- The geometry gate selected the more stable Pitch v1 fallback.
- 618/750 frames had reliable metric pitch calibration.
- 293 participant candidates outside the calibrated pitch were rejected.
- The full CPU run completed in 28.73 minutes with two worker CPUs.

This acceptance exposed the RabbitMQ 30-minute consumer acknowledgement timeout.
The Compose service now persists a 24-hour timeout, and the worker handles
redelivery idempotently so a completed run is never executed twice.

Run 48 is the focused 200-frame regression after fixing long ball-detector gaps:

- 51 measured ball frames and 59 short interpolated frames.
- 3 expired tracks reset cleanly and the ball was reacquired 4 times.
- 35 implausible motion candidates were rejected.
- The worker completed without a restart and the RabbitMQ queue returned to
  zero ready and zero unacknowledged messages.

The browser artifact endpoint was also verified with byte ranges. The result is
an H.264 MP4 at 1920x1080, 25 fps, and supports HTTP `206 Partial Content` for
normal playback and seeking.

Tracking quality remains a measured review gate, not a promise of perfect
identity. Run 48 is marked `needs_review`; IDF1 and HOTA remain empty until
human-labelled ground truth is supplied through the benchmark workflow.
