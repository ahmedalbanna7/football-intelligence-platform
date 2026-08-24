# Ball Tracking v4

Ball Tracking v4 is the guarded single-ball pipeline used by `Match Analysis +`.
Its priority is identity correctness: when the ball cannot be located reliably,
the overlay is intentionally omitted instead of attaching the marker to a
distant player or a static pitch feature.

## Pipeline

1. A dedicated football-ball model runs on every frame at high resolution.
2. Static field markers, penalty spots, and out-of-pitch candidates are filtered.
3. Candidate size, shape, player-body overlap, and player-foot proximity are checked.
4. Image-space Kalman motion and metric pitch-space motion score each candidate.
5. Forward/backward Lucas-Kanade optical flow bridges short detector gaps.
6. Cross-pitch jumps and ambiguous reacquisitions are rejected.
7. Predictions are limited to six frames and cannot create a new possession owner.

Measured observations use a filled yellow triangle. Short predictions use an
outlined yellow triangle. No triangle means that the pipeline did not have a
safe ball location for that frame.

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

## Remaining limitation

A single tactical camera cannot see an occluded ball in every frame. Improving
coverage further requires more labeled small-ball examples and ball ground truth.
The current release favors a missing marker over a confidently wrong marker.
