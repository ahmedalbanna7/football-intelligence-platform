# Ground Truth Annotation Editor

The Ground Truth Annotation Editor is the analyst-facing validation surface for
identity tracking and ball tracking. It lives inside `Match Analysis` and uses
the original uploaded video, not the rendered overlay, so generated labels do
not hide the evidence being reviewed.

## Why It Exists

Runtime quality signals can find suspicious motion, fragments, crossings, and
weak ball candidates, but they cannot prove that an identity or ball location is
correct. A manually reviewed reference is required to calculate release metrics.

The editor keeps three states separate:

- **Model candidate**: a generated starting point. It is not ground truth.
- **Draft**: saved analyst work that may contain unreviewed frames.
- **Verified**: every selected frame was checked and an annotator was recorded.
  Only this state can be evaluated.

The backend rejects a document marked `verified` when any frame or object is
still unverified, the annotator is missing, or the coordinates are invalid.

## Open the Editor

1. Open `Match Analysis` and select a processed run.
2. Open `Tracking Quality Gate`.
3. In `Quality Overview`, set `Clip start`, `Clip end`, and `Sample every`.
4. Select the scenario and camera style, then enable `Critical clip` when the
   range contains crossing, crowding, occlusion, or re-entry.
5. Choose `Identity editor` or `Ball editor`.
6. The app opens the `Ground Truth` tab and seeks the original source video to
   the selected frame.

Use `Saved identities` or `Saved ball` to resume work already stored for the
run. The Ground Truth tab badge counts both server-side annotation documents,
even before they are loaded into the browser.

Saved annotation documents belong to one analysis run. `Load saved` never
searches another run: the Ground Truth toolbar shows whether the selected run
has a saved identity or ball document, its status, and its frame count. When no
document exists, create a draft from the range selected in `Quality Overview`.
Loading, success, and missing-document results are shown directly under the
toolbar so the action remains visible inside the editor.

## Identity Workflow

For every selected frame:

1. Select each box and move or resize it to the visible person.
2. Keep one `Stable identity` value for the same physical person across every
   frame and clip. Never use the current Track ID as truth when it switched.
3. Correct `Team` and `Participant role`.
4. Use the draw tool to add a missed person.
5. Use `Remove false detection` for a coach, spectator, field fixture, duplicate,
   or other false participant candidate.
6. Choose `Verify frame` only after every visible participant has been handled.

The copy-previous control is useful for adjacent frames, but copied objects are
made unverified so the analyst must check motion and occlusion. Undo restores the
last local edit. Download exports the current JSON as an additional backup.

## Ball Workflow

For every selected frame:

1. Click or drag the marker to the center of the visible ball.
2. Select exactly one state:
   - `Visible`: the ball center is directly observable.
   - `Occluded`: the ball is hidden but the frame was reviewed.
   - `Out Of Frame`: the ball is outside the image.
   - `Uncertain`: the analyst cannot establish a reliable state yet.
3. Enable `Airborne` only when the ball is visibly above playable ground level.
4. Enter `Estimated height (cm)` only when a useful estimate can be made. A
   monocular estimate is allowed, but it remains explicit rather than invented.
5. Clear a wrong generated marker instead of accepting a remote candidate.
6. Choose `Verify frame` after the state and marker have been checked.

A frame can be verified without coordinates when its state is `occluded` or
`out_of_frame`. A `visible` frame must have a finite in-image center.

## Save and Evaluate

- `Save draft` persists work without making an accuracy claim.
- `Verify all & save` remains disabled until every frame was explicitly checked
  with `Verify frame`. It then records the annotator and promotes the complete
  document to verified without silently approving untouched frames.
- `Evaluate` requires a fully verified document. It saves the verified reference
  and calculates the corresponding quality metrics.

Identity evaluation reports IDF1, HOTA, exact ID switches, fragmentation,
cross-team identity reuse, and per-clip release conditions.

Ball evaluation reports precision, recall, F1, mean/median/p95 center error,
observed and interpolated matches, maximum miss gap, airborne-state accuracy,
height MAE when height labels exist, and a release decision.

The default ball release gate also requires at least 20 evaluated frames and at
least 10 visible-ball frames. This prevents a tiny or mostly uncertain sample
from producing a misleading pass.

## Frame Coordinates

Editor coordinates use source-video pixels in `[x1, y1, x2, y2]` order for
identity boxes and `[x, y]` for ball centers. A run created with a non-zero
`start_frame` stores both local `frame` and absolute `source_frame`. The video
seek adds the run's source offset so labels stay aligned with future reruns.

## Storage

Documents are stored in MinIO beside the analysis run:

```text
matches/{match_id}/match-analysis-plus/runs/{run_id}/tracking-quality/ground_truth.json
matches/{match_id}/match-analysis-plus/runs/{run_id}/ball_ground_truth.json
```

Run summary metadata records document status, frame counts, verified frame
counts, state counts, readiness for evaluation, and update time. Identity draft
generation also stores a range-specific artifact; `Save draft` persists the
current identity or ball document as the run's resumable canonical annotation.

## API

```text
POST /match-analysis-plus/{match_id}/runs/{run_id}/quality/ground-truth/draft
GET  /match-analysis-plus/{match_id}/runs/{run_id}/quality/ground-truth
PUT  /match-analysis-plus/{match_id}/runs/{run_id}/quality/ground-truth
POST /match-analysis-plus/{match_id}/runs/{run_id}/quality/benchmark

POST /match-analysis-plus/{match_id}/runs/{run_id}/quality/ball-ground-truth/draft
GET  /match-analysis-plus/{match_id}/runs/{run_id}/quality/ball-ground-truth
PUT  /match-analysis-plus/{match_id}/runs/{run_id}/quality/ball-ground-truth
POST /match-analysis-plus/{match_id}/runs/{run_id}/quality/ball-ground-truth/benchmark
```

## Recommended Validation Set

Use 500-1000 frame processed windows and annotate high-value samples rather than
blindly labeling every frame. Include tactical and close/moving cameras, both
teams, both goalkeepers, officials, crossings, dense penalty-area play, long
occlusion/re-entry, ground passes, high balls, and genuine out-of-frame periods.
The release suite should combine verified cases from more than one source video.
