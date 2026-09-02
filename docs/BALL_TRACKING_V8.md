# Ball Tracking v8

Ball Tracking v8 is the generic ground/air state pipeline used by `Match
Analysis`. It combines dedicated detections, 2D Kalman motion, Lucas-Kanade
optical flow, competing reacquisition hypotheses, pitch calibration, and a
confidence-gated monocular ballistic estimate.

The release policy is precision first: a temporarily missing marker is better
than a marker on a boot, player body, penalty spot, or stale goalkeeper
location.

## Display contract

- A yellow triangle means a confirmed or short, guarded ground-ball path.
- A cyan ring means an airborne or ground-unknown path.
- An airborne ball is omitted from the 2D pitch radar and cannot own possession.
- A missing marker means that no candidate passed the temporal safety gates.
- After landing, yellow returns only after ballistic contact or fresh,
  multi-frame ground evidence.

## Generic safety gates

1. Candidates are associated across frames by position, velocity, acceleration,
   size, direction, and metric pitch motion.
2. Optical flow bridges short detector gaps but is rejected when it drifts from
   the image or ballistic trajectory.
3. Remote candidates become parallel hypotheses; they cannot replace the live
   ball until repeated motion evidence wins.
4. Candidate paths attached to player bodies are rejected. Foot proximity may
   support ground contact but never overrides trajectory consistency.
5. Monocular `(X, Y, Z)` is metadata with confidence. A weak 3D projection
   cannot move the visible 2D marker.
6. Reacquisition does not restart the total airborne episode. An unresolved
   episode expires after five seconds (scaled by video FPS), hides the marker,
   and requires fresh temporal evidence. It never converts directly to a
   ground owner.

No video name, coordinate, team color, track number, goalkeeper position, or
frame range is hardcoded in these rules.

## Run 115 acceptance

Run 115 processed frames 0-749 of `08fd33_4.mp4` using the same cached
detections as the earlier regression runs, isolating tracker behavior.

| Metric | Result |
| --- | ---: |
| Processing | 750 / 750 frames |
| GPU active | yes |
| Ball Quality Gate | passed |
| Observed ball frames | 287 |
| Interpolated ball frames | 209 |
| Total image-path frames | 496 |
| Maximum interpolation streak | 10 |
| Airborne episode limit | 125 frames at 25 FPS |
| Expired unresolved airborne episodes | 2 |
| Confirmed ground contacts | 2 |
| Player-body pass-throughs | 0 |
| Airborne frames excluded from possession | 254 |
| Confirmed possession frames | 132 |
| Possession readiness | passed |

Visual checks covered the two episode expirations and the ground transitions at
frames 196, 489, 559, and 666. The cyan marker disappears before uncertain
reacquisition; yellow returns only with a temporally confirmed ground candidate.

## Limitations and validation

Height from one tactical camera is an estimate, not an absolute measurement.
The system therefore exposes `trajectory_3d_confidence`, blocks low-confidence
3D from rendering, and keeps Ball Ground Truth as the release evidence for new
camera styles. Cross-match precision/recall, continuity, airborne-state
accuracy, and optional height error must be measured in the built-in annotation
editor before claiming production accuracy for a new broadcast domain.

See [Ground Truth Annotation Editor](GROUND_TRUTH_ANNOTATION_EDITOR.md) for the
review and benchmark workflow.
