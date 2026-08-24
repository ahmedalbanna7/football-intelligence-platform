# Tracking Accuracy v2

Tracking Accuracy v2 treats identity as a measured, reviewable contract. A low
track count is not considered success when one ID has been assigned to multiple
physical people.

## Association order

1. Reject impossible image-plane foot displacement before appearance scoring.
2. Preserve a locally plausible native BoT-SORT ID.
3. Reject locked kit-family conflicts and strong color conflicts.
4. Use 2D foot trajectory, depth proxy, direction, body scale, and robust
   appearance-gallery similarity in a global one-to-one assignment.
5. Suppress ambiguous overlap observations instead of drawing predicted IDs in
   empty space.
6. After a long gap, restore an ID only when native BoT-SORT Re-ID retained the
   raw identity and the visual evidence is compatible. Otherwise create a new
   fragment for manual merge.
7. Resolve and lock `player`, `goalkeeper`, `referee`, `assistant_referee`, and
   `staff_outside_pitch` roles in a separate temporal classifier. Role labels do
   not participate in identity matching.

Starting a new fragment is safer than assigning an old identity to another
person. A fragment can be merged after review; an identity switch corrupts
distance, speed, possession, heatmaps, and player reports.

## Ground-truth protocol

Ground-truth templates are generated from the Quality Overview for selected
frame ranges. Generated labels are drafts, not measurements.

For each draft:

1. Correct every bounding box and identity label.
2. Add missed people and remove false detections.
3. Use the same `identity_id` for the same physical person across all clips.
4. Set each object's `review_state` to `verified`.
5. Set `verification.status` to `verified`, add `verification.annotator`, and add
   `verification.reviewed_at`.

The benchmark API rejects unverified files. Predictions outside annotated frames
are excluded so a short clip is not incorrectly compared with a full run.

## Match 12 regression clips

Source: `08fd33_4.mp4`

| Frames | Purpose | Recommended sample step |
| --- | --- | --- |
| 0-60 | Initial global identity permutation | 1 |
| 480-540 | Crowding and re-entry | 1 |
| 570-630 | Camera movement and cross-team association | 1 |
| 700-749 | Long disappearance and late re-entry | 1 |

At minimum, manually verify every player, goalkeeper, and referee visible in
these clips. Include a second tactical-camera match and a close-camera clip
before approving a release.

## Current measured baseline

The repository includes five verified partial-identity fixtures:

- `benchmarks/tracking/match12_identity_a_verified.json` targets the first
  confirmed global identity permutation.
- `benchmarks/tracking/match12_multi_identity_verified.json` covers 17 physical
  people from both teams, including an assistant referee, at frames 15, 27, and
  39. Its 51 identity observations were visually reviewed before verification.
- `benchmarks/tracking/match12_late_multi_identity_verified.json` covers source
  frames 480-749. It contains 102 verified observations for 17 people across six
  camera, crowding, and re-entry checkpoints.
- `benchmarks/tracking/match10_close_moving_verified.json` covers a separate
  close/moving-camera clip with visually verified identities from both teams.
- `benchmarks/tracking/match11_mid_tactical_verified.json` covers a 500-frame
  mid-match tactical segment at original source frames 90003-90403. Five
  persistent identities from both teams were verified through camera motion.

On the same fixture and source frames:

| Run | Stable identity layer | ID switches | IDF1 | HOTA |
| --- | --- | ---: | ---: | ---: |
| 47 | Previous v4 association | 3 | 90.000 | 89.370 |
| 50 | v5 conservative Re-ID + raw-motion conflict guard | 0 | 99.029 | 99.034 |

The complete 200-frame Run 50 prediction stream contains 4,170 accepted person
observations across 29 stable tracks. An independent image-motion audit found
zero accepted hard-motion violations across all tracks. The raw-motion conflict
guard rejected eight native-ID ownership conflicts before assignment.

Run 51 evaluated source frames 480-749 as a separate late-match segment. The
verified 17-person fixture measured zero ID switches, zero fragmentation,
`IDF1 99.512`, and `HOTA 99.513`. Across all 5,950 accepted observations, the
integrated motion gate reported zero tracks over the hard-motion limit and
rejected 19 conflicting native-ID ownership claims.

The expanded release suite measured the current candidate on all four fixtures:

| Camera | Scenario | ID switches | IDF1 | HOTA | Cross-team transfers |
| --- | --- | ---: | ---: | ---: | ---: |
| Tactical | Crossing | 0 | 100.000 | 100.000 | 0 |
| Tactical | Crowding | 0 | 99.030 | 99.030 | 0 |
| Tactical | Re-entry | 0 | 99.510 | 98.200 | 0 |
| Close/moving | Crowding | 0 | 100.000 | 100.000 | 0 |
| Tactical mid-match | Camera motion | 0 | 100.000 | 100.000 | 0 |

The aggregate release conditions pass with minimum `IDF1 99.029`, minimum
`HOTA 98.2`, zero critical ID switches, zero cross-team transfers, and both
required camera styles across five measured cases. These are selected-identity benchmark results, not a
claim of perfect full-match tracking. The manifest in
`benchmarks/tracking/release-gate-manifest.json` records the exact coverage and
remaining all-visible annotation recommendations.

## Release quality gate

- Zero ID switches in manually verified critical crossing clips.
- Zero accepted hard-motion violations.
- Zero team-family identity transfers in reviewed clips.
- Zero player/referee role changes after role lock.
- Report IDF1, HOTA, exact ID switches, and fragmentation from verified ground
  truth only.
- Compare the candidate run against the previous released run on the same clips.
- Require both tactical and close/moving camera cases in the same release suite.
- Keep officials and outside staff out of player analytics while retaining them
  in Track Review for role correction.

Perfect identity cannot be guaranteed from every single-camera video, especially
through long occlusion or a camera cut. The production rule is therefore:
conservative automatic identity, explicit confidence, measurable benchmarks,
and manual correction for unresolved fragments.
