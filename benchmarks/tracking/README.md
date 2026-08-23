# Tracking Release Benchmarks

This directory contains manually verified ground-truth fixtures for the
Tracking Accuracy v2 release gate. Automatically generated drafts must never be
treated as benchmark truth.

## Coverage

`release-gate-manifest.json` is the source of truth for the expected scenarios,
camera styles, thresholds, and verification state. Current verified coverage
includes:

- tactical-camera crossing;
- tactical-camera crowding;
- tactical-camera disappearance and re-entry;
- close/moving-camera crowding;
- a separate 500-frame tactical segment with camera motion;
- verified team and participant-role metadata for cross-team checks.

Each fixture declares whether it covers every visible participant or only a
selected set of identities. Scores from selected-identity fixtures apply only
to those reviewed identities and frames.

## Release Conditions

- zero ID switches in critical clips;
- `IDF1 >= 95`;
- `HOTA >= 90`;
- zero stable-identity transfers between teams;
- every reported fragment resolved or explicitly reviewed;
- crossing, crowding, and re-entry scenarios present;
- tactical and close/moving camera styles present.

## Workflow

1. Build 500-1000-frame run requests from the Tracking Quality Gate UI.
2. Run each planned clip and open Track Review.
3. Generate a draft for a critical frame range.
4. Correct boxes, stable identities, teams, and participant roles manually.
5. Set every object to `review_state: verified` and complete the verification
   metadata.
6. Submit the prediction/ground-truth cases to the release-suite endpoint.
7. Commit verified fixtures and update the manifest with their exact coverage.

Predictions and ground truth carry both run-local `frame` and original-video
`source_frame` coordinates. This keeps an annotation aligned when the same
source segment is processed in a different run.
