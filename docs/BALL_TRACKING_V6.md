# Ball Tracking v6

Ball Tracking v6 is the guarded multi-frame single-ball pipeline used by
`Match Analysis +`. It combines dedicated detections, image motion, metric
pitch motion, and a confidence-gated monocular 3D estimate. The release rule is
deliberately conservative: a temporarily missing marker is better than a
confident marker on a player, boot, penalty spot, or unrelated object.

## Pipeline

1. The dedicated ball detector produces candidates at analysis resolution.
2. Static pitch features, implausible sizes, pitch exits, and participant-body
   patches are rejected before association.
3. A 2D Kalman state scores position, velocity, acceleration, size continuity,
   and direction over multiple frames.
4. Forward/backward Lucas-Kanade optical flow bridges short detector gaps.
5. The live track cannot be replaced by a remote candidate. Competing
   hypotheses are created only after the track is genuinely dormant.
6. Dormant reacquisition keeps multiple candidates in parallel. A candidate
   must show repeated observations and plausible motion before promotion.
7. Candidate-to-participant ownership is tracked while reacquiring, preventing
   one body patch from being linked through different nearby players.
8. Metric pitch coordinates provide a second motion gate when calibration is
   valid. Image continuity remains authoritative when metric projection is
   weak or conflicts with an airborne observation.
9. The trajectory state estimates `(X, Y, Z)` from one camera using pitch
   homography, camera candidates, temporal motion, and a ballistic prior.
10. A 3D projection can move the visible marker only when its confidence is at
    least the release threshold. Otherwise the system keeps the 2D/optical-flow
    result or omits the marker.

## Marker semantics

- A filled yellow triangle is an observed ball position.
- An outlined yellow triangle is a Kalman or optical-flow prediction.
- The tip of the triangle points to the top of the ball bounding box.
- No triangle means no position passed the safety gates for that frame.
- The triangle is the ball marker; it is not a separate possession-owner icon.

## Airborne and 3D behavior

The 3D height is an estimate because a single tactical camera does not provide
true depth. Each sample therefore carries `trajectory_3d_confidence`. Low
confidence 3D estimates cannot override the visible 2D location. Real,
temporally consistent detections correct the trajectory immediately after an
occlusion.

Ground and airborne states use different interpolation budgets. Verified
airborne motion can survive a longer detector gap, while an unverified ground
track expires quickly. Height, direction, acceleration, player-body overlap,
and camera projection plausibility all contribute to the airborne gate.

## Possession safeguards

- Only eligible players and goalkeepers can own the ball.
- Referees, assistant referees, staff, and outside-pitch people are excluded.
- Ownership needs repeated observed frames; predicted frames can hold an owner
  briefly but cannot create a transfer.
- Fast moving balls use a smaller control radius and must be approaching the
  candidate player.
- An airborne ball above the release height has no owner.
- Possession and passes remain blocked until the ball gate passes and confirmed
  owner coverage reaches 15% of analyzed frames.

## Run 94 validation

Run 94 processed the first 750 frames of `08fd33_4.mp4` on CUDA using the same
cached dedicated detections as the earlier regression runs.

| Metric | Run 94 |
| --- | ---: |
| Processing | 750 / 750 frames |
| Processing rate | 3.533 FPS |
| GPU active | yes |
| Observed ball frames | 304 |
| Interpolated ball frames | 163 |
| Tracked ball frames | 467 |
| Peak dormant hypotheses | 5 |
| Dormant hypothesis promotions | 6 |
| Optical-flow successes | 623 |
| Player-body rejections | 5 |
| Metric jump rejections | 418 |
| Low-confidence 3D image fallbacks | 34 |
| 3D observation projection rejections | 9 |
| Ball Quality Gate | passed |

The critical frames 626-678 were checked against the source video at full
resolution. The ball path advances continuously from approximately
`(915, 539)` to `(1050, 614)`. Frames 638-640 are predicted through the player
occlusion, and frame 641 reacquires the observed ball without jumping to a
distant player or goalkeeper.

Possession coverage in this 30-second segment is only 5.2%. The pipeline
therefore keeps possession and pass analytics blocked instead of presenting
low-coverage estimates as verified facts.

## Final Run 101 regression

Run 101 repeated the same 750 source frames after the final generic identity,
cache-provenance, and resolution-scaling fixes. The ball quality gate still
passes with 328 observed frames, 173 interpolated frames, 661 successful
optical-flow updates, five player-body false-candidate rejections, and 34
low-confidence 3D-to-2D fallbacks.

The critical path remained unchanged: frame 626 starts near `(915.2, 539.0)`,
frames 638-640 bridge the player occlusion, frame 641 reacquires near
`(949.1, 554.6)`, and frame 678 ends near `(1050.2, 613.9)`. It does not jump
to the nearby goalkeeper or a remote player. This confirms that the tracking
identity fixes did not regress the guarded ball path.

## Generalization boundaries

No coordinates, shirt colors, player IDs, or goalkeeper locations from this
match are hardcoded. The gates use normalized geometry, temporal evidence,
camera calibration confidence, and participant roles. Track identity has been
benchmarked across 360p, 1080p, and 4K videos from tactical and close/moving
cameras. Ball behavior is generic in code and has guarded regression coverage,
but additional manually labeled ball ground truth from other matches is still
required before claiming measured ball accuracy across every broadcast style.

## Ball Ground Truth

The built-in `Ground Truth > Ball` editor now creates and resumes server-side
ball annotation documents against the original source video. It records visible,
occluded, out-of-frame, and uncertain states, center coordinates, airborne state,
and optional height. Generated model candidates always begin unverified.

Only a fully reviewed document with an annotator can run the ball benchmark.
The benchmark measures localization precision/recall/F1, center error,
continuity gaps, airborne accuracy, and height error when height labels exist.
Its default release gate needs at least 20 evaluated frames, including 10
visible-ball frames, before accuracy thresholds can pass.
This is the evidence required to replace guarded runtime observations with a
measured cross-video accuracy claim. See `docs/GROUND_TRUTH_ANNOTATION_EDITOR.md`.
