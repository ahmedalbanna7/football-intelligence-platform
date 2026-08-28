# Ball Tracking v4.1

> Superseded by [Ball Tracking v6](BALL_TRACKING_V6.md). The v6 validation
> rechecked the source video at full resolution and corrected the historical
> interpretation of frames 626-678 below.

Ball Tracking v4 is the guarded single-ball pipeline used by `Match Analysis +`.
Its priority is identity correctness: when the ball cannot be located reliably,
the overlay is intentionally omitted instead of attaching the marker to a
distant player or a static pitch feature.

## Pipeline

1. A dedicated football-ball model runs on every frame at high resolution.
2. Static field markers, penalty spots, and out-of-pitch candidates are filtered.
3. Candidate size, shape, player-body overlap, and player-foot proximity are checked.
4. Image-space Kalman motion and metric pitch-space motion score each candidate.
5. Ground and airborne motion are tracked as separate states. An airborne ball
   follows image continuity even when its ground-plane projection is misleading.
6. Forward/backward Lucas-Kanade optical flow bridges detector gaps. Ground
   predictions remain limited to six frames; verified airborne flow may bridge
   up to 36 frames.
7. A ballistic trajectory gate rejects candidates that drift away from the
   last verified flight path, including white boots and kit fragments.
8. A dormant track keeps its identity for up to 90 frames. A distant candidate
   can replace it only after two strong observations with real displacement;
   a stationary high-confidence candidate cannot take over the track.
9. Cross-pitch jumps and ambiguous reacquisitions are rejected.

Measured observations use a filled yellow triangle. Optical-flow or Kalman
predictions use an outlined yellow triangle. No triangle means that the
pipeline did not have a safe ball location for that frame.

## Possession safeguards

- A player must remain the closest eligible participant within 180 cm for three
  observed frames before ownership is confirmed.
- Referees, assistant referees, staff, outside-pitch people, and predicted player
  positions cannot receive possession.
- An observed ball far from the current owner releases ownership immediately.
- A gap longer than three seconds starts a new possession instead of linking two
  unrelated observations.
- Transfers implying a ball speed above 50 m/s are stored as
  `unverified_reacquisition`; they are not counted as passes or turnovers.
- Possession and pass analytics are blocked unless at least 15% of the analyzed
  frames have a confirmed owner and the Ball Quality Gate passes.

## Validation snapshot

The same 750-frame `08fd33_4.mp4` segment was processed before and after v4.

| Metric | Run 69 | Run 74 |
| --- | ---: | ---: |
| Observed ball frames | 158 | 377 |
| Tracked ball frames | 376 | 510 |
| Metric jump rejections | unavailable | 270 |
| Player-body false-positive rejections | unavailable | 75 |
| Optical-flow successes | unavailable | 510 |
| Confirmed-owner coverage | not gated | 29.07% |

Run 74 passed the ball gate. Two geometrically impossible transitions were kept
only as low-confidence `unverified_reacquisition` diagnostics and excluded from
the pass and turnover totals.

### Airborne regression validation

Run 82 processed all 750 frames of `08fd33_4.mp4` with the v4.1 tracker and the
cached dedicated-ball detections. The critical pass in frames 564-749 was
checked against the rendered video and image-space path:

| Metric | Run 82 |
| --- | ---: |
| Observed ball frames | 307 |
| Tracked ball frames | 584 |
| Interpolated frames | 277 |
| Airborne observed frames | 41 |
| Maximum airborne prediction streak | 35 / 36 |
| Airborne body pass-throughs | 13 |
| Ballistic trajectory rejections | 7 |
| Dormant-track challenger promotions | 4 |

The marker reacquired the real ball at frame 580 and crossed in front of the
player without attaching to the player body. A later full-resolution review in
v6 established that the path around image coordinates `x=925..1050` in frames
626-678 is the real ball moving at the player's feet, not a false player-body
path. This correction is covered by a dedicated v6 regression test.

## Remaining limitation

A single tactical camera cannot see an occluded ball in every frame. Improving
coverage further requires more labeled small-ball examples and ball ground truth.
The current release favors a missing marker over a confidently wrong marker.
