import {
  ArrowLeft,
  ArrowRight,
  Check,
  CircleDot,
  Copy,
  Download,
  MousePointer2,
  Plus,
  Save,
  ShieldCheck,
  Trash2,
  Undo2
} from "lucide-react";
import {
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import type {
  BallGroundTruthDocument,
  BallGroundTruthFrame,
  BallGroundTruthState,
  TrackingGroundTruthDocument,
  TrackingGroundTruthFrame,
  TrackingGroundTruthObject
} from "./types";

type AnnotationMode = "tracking" | "ball";
type AnnotationDocument = TrackingGroundTruthDocument | BallGroundTruthDocument;
type EditorTool = "select" | "draw";

type GroundTruthAnnotationEditorProps = {
  mode: AnnotationMode;
  document: AnnotationDocument;
  videoSrc: string;
  videoFrameOffset?: number;
  fps: number;
  busy?: boolean;
  metrics?: Record<string, unknown> | null;
  onChange: (document: AnnotationDocument) => void;
  onSave: (document: AnnotationDocument) => Promise<void>;
  onEvaluate: (document: AnnotationDocument) => Promise<void>;
};

type DragState = {
  kind: "draw" | "move" | "resize" | "ball";
  objectIndex?: number;
  corner?: "nw" | "ne" | "sw" | "se";
  start: [number, number];
  original?: [number, number, number, number];
};

const ROLE_OPTIONS: Array<NonNullable<TrackingGroundTruthObject["role_name"]>> = [
  "player",
  "goalkeeper",
  "referee",
  "assistant_referee",
  "staff_outside_pitch"
];

function cloneDocument<T>(document: T): T {
  return structuredClone(document);
}

function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function trackingDocument(document: AnnotationDocument): document is TrackingGroundTruthDocument {
  return document.schema_version.startsWith("tracking_ground_truth");
}

function frameVerified(mode: AnnotationMode, frame: TrackingGroundTruthFrame | BallGroundTruthFrame) {
  if (mode === "ball") return (frame as BallGroundTruthFrame).review_state === "verified";
  const trackingFrame = frame as TrackingGroundTruthFrame;
  const objectsVerified = trackingFrame.objects.every((item) => item.review_state === "verified");
  if (trackingFrame.review_state != null) {
    return trackingFrame.review_state === "verified" && objectsVerified;
  }
  return trackingFrame.objects.length > 0 && objectsVerified;
}

export function GroundTruthAnnotationEditor({
  mode,
  document,
  videoSrc,
  videoFrameOffset = 0,
  fps,
  busy = false,
  metrics,
  onChange,
  onSave,
  onEvaluate
}: GroundTruthAnnotationEditorProps) {
  const [frameIndex, setFrameIndex] = useState(0);
  const [selectedObject, setSelectedObject] = useState<number | null>(null);
  const [tool, setTool] = useState<EditorTool>(mode === "ball" ? "draw" : "select");
  const [annotator, setAnnotator] = useState(document.verification.annotator || "");
  const [drag, setDrag] = useState<DragState | null>(null);
  const [previewBox, setPreviewBox] = useState<[number, number, number, number] | null>(null);
  const [newIdentity, setNewIdentity] = useState("");
  const historyRef = useRef<AnnotationDocument[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const frames = document.frames;
  const safeFrameIndex = clamp(frameIndex, 0, Math.max(0, frames.length - 1));
  const currentFrame = frames[safeFrameIndex];
  const resolution = document.resolution?.length === 2
    ? document.resolution
    : [1920, 1080] as [number, number];
  const width = Math.max(1, Number(resolution[0]));
  const height = Math.max(1, Number(resolution[1]));
  const verifiedCount = useMemo(
    () => frames.filter((frame) => frameVerified(mode, frame)).length,
    [frames, mode]
  );
  const progress = frames.length ? verifiedCount / frames.length : 0;
  const allFramesVerified = frames.length > 0 && verifiedCount === frames.length;
  const documentKey = [
    document.schema_version,
    document.source?.run_id,
    document.clips?.[0]?.start_frame,
    document.clips?.[0]?.end_frame,
    document.clips?.[0]?.source_start_frame,
    document.verification.annotator,
    document.verification.reviewed_at
  ].join(":");

  useEffect(() => {
    setFrameIndex(0);
    setSelectedObject(null);
    setAnnotator(document.verification.annotator || "");
    historyRef.current = [];
  }, [documentKey]);

  useEffect(() => {
    setTool(mode === "ball" ? "draw" : "select");
    setSelectedObject(null);
  }, [mode]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !currentFrame) return;
    const target = sourceFrameFor(currentFrame) / Math.max(fps || document.fps || 25, 0.001);
    if (Math.abs(video.currentTime - target) > 0.005) video.currentTime = target;
    video.pause();
  }, [currentFrame?.frame, currentFrame?.source_frame, document.fps, documentKey, fps, videoFrameOffset]);

  function checkpoint() {
    historyRef.current = [...historyRef.current.slice(-39), cloneDocument(document)];
  }

  function update(next: AnnotationDocument, saveHistory = true) {
    if (saveHistory) checkpoint();
    onChange(next);
  }

  function undo() {
    const previous = historyRef.current.pop();
    if (previous) onChange(previous);
  }

  function pointFromEvent(event: ReactPointerEvent<SVGSVGElement | SVGElement>): [number, number] {
    const svg = svgRef.current;
    if (!svg) return [0, 0];
    const bounds = svg.getBoundingClientRect();
    return [
      clamp((event.clientX - bounds.left) * width / Math.max(1, bounds.width), 0, width),
      clamp((event.clientY - bounds.top) * height / Math.max(1, bounds.height), 0, height)
    ];
  }

  function replaceTrackingFrame(nextFrame: TrackingGroundTruthFrame, saveHistory = true) {
    if (!trackingDocument(document)) return;
    const next = cloneDocument(document);
    if (next.verification.status === "verified") {
      next.verification = { ...next.verification, status: "draft", reviewed_at: null };
    }
    next.frames[safeFrameIndex] = nextFrame;
    update(next, saveHistory);
  }

  function replaceBallFrame(nextFrame: BallGroundTruthFrame, saveHistory = true) {
    if (trackingDocument(document)) return;
    const next = cloneDocument(document);
    if (next.verification.status === "verified") {
      next.verification = { ...next.verification, status: "draft", reviewed_at: null };
    }
    next.frames[safeFrameIndex] = nextFrame;
    update(next, saveHistory);
  }

  function startCanvasAction(event: ReactPointerEvent<SVGRectElement>) {
    const point = pointFromEvent(event);
    if (mode === "ball") {
      checkpoint();
      setDrag({ kind: "ball", start: point });
      setBallPosition(point, false);
      return;
    }
    if (tool !== "draw") {
      setSelectedObject(null);
      return;
    }
    checkpoint();
    setDrag({ kind: "draw", start: point });
    setPreviewBox([point[0], point[1], point[0], point[1]]);
  }

  function startObjectMove(event: ReactPointerEvent<SVGRectElement>, objectIndex: number) {
    event.stopPropagation();
    if (!trackingDocument(document) || tool !== "select") return;
    checkpoint();
    setSelectedObject(objectIndex);
    setDrag({
      kind: "move",
      objectIndex,
      start: pointFromEvent(event),
      original: [...document.frames[safeFrameIndex].objects[objectIndex].bbox]
    });
  }

  function startResize(
    event: ReactPointerEvent<SVGCircleElement>,
    objectIndex: number,
    corner: DragState["corner"]
  ) {
    event.stopPropagation();
    if (!trackingDocument(document)) return;
    checkpoint();
    setDrag({
      kind: "resize",
      objectIndex,
      corner,
      start: pointFromEvent(event),
      original: [...document.frames[safeFrameIndex].objects[objectIndex].bbox]
    });
  }

  function handlePointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (!drag || !currentFrame) return;
    const point = pointFromEvent(event);
    if (drag.kind === "draw") {
      setPreviewBox([
        Math.min(drag.start[0], point[0]),
        Math.min(drag.start[1], point[1]),
        Math.max(drag.start[0], point[0]),
        Math.max(drag.start[1], point[1])
      ]);
      return;
    }
    if (drag.kind === "ball") {
      setBallPosition(point, false);
      return;
    }
    if (!trackingDocument(document) || drag.objectIndex == null || !drag.original) return;
    const nextFrame = cloneDocument(document.frames[safeFrameIndex]);
    const [x1, y1, x2, y2] = drag.original;
    const dx = point[0] - drag.start[0];
    const dy = point[1] - drag.start[1];
    let nextBox: [number, number, number, number];
    if (drag.kind === "move") {
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;
      const nextX1 = clamp(x1 + dx, 0, width - boxWidth);
      const nextY1 = clamp(y1 + dy, 0, height - boxHeight);
      nextBox = [nextX1, nextY1, nextX1 + boxWidth, nextY1 + boxHeight];
    } else {
      nextBox = [x1, y1, x2, y2];
      if (drag.corner?.includes("n")) nextBox[1] = clamp(y1 + dy, 0, y2 - 3);
      if (drag.corner?.includes("s")) nextBox[3] = clamp(y2 + dy, y1 + 3, height);
      if (drag.corner?.includes("w")) nextBox[0] = clamp(x1 + dx, 0, x2 - 3);
      if (drag.corner?.includes("e")) nextBox[2] = clamp(x2 + dx, x1 + 3, width);
    }
    nextFrame.objects[drag.objectIndex].bbox = nextBox.map((value) => Number(value.toFixed(2))) as TrackingGroundTruthObject["bbox"];
    nextFrame.objects[drag.objectIndex].review_state = "unverified";
    nextFrame.review_state = "unverified";
    replaceTrackingFrame(nextFrame, false);
  }

  function sourceFrameFor(frame: TrackingGroundTruthFrame | BallGroundTruthFrame) {
    if (frame.source_frame != null) return Number(frame.source_frame);
    const clip = document.clips?.find((item) => {
      const start = Number(item.start_frame ?? Number.NEGATIVE_INFINITY);
      const end = Number(item.end_frame ?? Number.POSITIVE_INFINITY);
      return frame.frame >= start && frame.frame <= end;
    });
    const clipStart = Number(clip?.start_frame ?? 0);
    const sourceStart = Number(clip?.source_start_frame ?? clipStart + videoFrameOffset);
    return sourceStart + frame.frame - clipStart;
  }

  function finishPointerAction() {
    if (drag?.kind === "draw" && previewBox) {
      const [x1, y1, x2, y2] = previewBox;
      if (x2 - x1 >= 4 && y2 - y1 >= 4 && trackingDocument(document)) {
        const nextFrame = cloneDocument(document.frames[safeFrameIndex]);
        const identity = newIdentity.trim() || `identity-new-${Date.now().toString(36)}`;
        nextFrame.objects.push({
          identity_id: identity,
          bbox: previewBox.map((value) => Number(value.toFixed(2))) as TrackingGroundTruthObject["bbox"],
          source_frame: sourceFrameFor(currentFrame),
          team: null,
          role_name: "player",
          review_state: "unverified"
        });
        nextFrame.review_state = "unverified";
        setSelectedObject(nextFrame.objects.length - 1);
        replaceTrackingFrame(nextFrame, false);
        setTool("select");
      }
    }
    setDrag(null);
    setPreviewBox(null);
  }

  function setBallPosition(point: [number, number], saveHistory = true) {
    if (trackingDocument(document)) return;
    const frame = cloneDocument(document.frames[safeFrameIndex]);
    const radius = Math.max(4, Math.hypot(width, height) * 0.004);
    frame.state = "visible";
    frame.review_state = "unverified";
    frame.ball = {
      ...frame.ball,
      center: [Number(point[0].toFixed(2)), Number(point[1].toFixed(2))],
      bbox: [
        Math.max(0, point[0] - radius),
        Math.max(0, point[1] - radius),
        Math.min(width, point[0] + radius),
        Math.min(height, point[1] + radius)
      ]
        .map((value) => Number(value.toFixed(2))) as [number, number, number, number],
      airborne: frame.ball?.airborne ?? false,
      height_cm: frame.ball?.height_cm ?? 0
    };
    replaceBallFrame(frame, saveHistory);
  }

  function updateSelectedObject(changes: Partial<TrackingGroundTruthObject>) {
    if (!trackingDocument(document) || selectedObject == null) return;
    const frame = cloneDocument(document.frames[safeFrameIndex]);
    frame.objects[selectedObject] = { ...frame.objects[selectedObject], ...changes };
    frame.review_state = "unverified";
    replaceTrackingFrame(frame);
  }

  function removeSelectedObject() {
    if (!trackingDocument(document) || selectedObject == null) return;
    const frame = cloneDocument(document.frames[safeFrameIndex]);
    frame.objects.splice(selectedObject, 1);
    frame.review_state = "unverified";
    replaceTrackingFrame(frame);
    setSelectedObject(null);
  }

  function updateBallFrame(changes: Partial<BallGroundTruthFrame>) {
    if (trackingDocument(document)) return;
    replaceBallFrame({
      ...cloneDocument(document.frames[safeFrameIndex]),
      ...changes,
      review_state: "unverified"
    });
  }

  function setBallState(state: BallGroundTruthState) {
    if (trackingDocument(document)) return;
    const frame = cloneDocument(document.frames[safeFrameIndex]);
    frame.state = state;
    frame.review_state = "unverified";
    if (state !== "visible") frame.ball = null;
    replaceBallFrame(frame);
  }

  function verifyCurrentFrame() {
    if (trackingDocument(document)) {
      const frame = cloneDocument(document.frames[safeFrameIndex]);
      frame.review_state = "verified";
      frame.objects = frame.objects.map((item) => ({ ...item, review_state: "verified" }));
      replaceTrackingFrame(frame);
    } else {
      const frame = cloneDocument(document.frames[safeFrameIndex]);
      if (frame.state === "visible" && !frame.ball) return;
      frame.review_state = "verified";
      replaceBallFrame(frame);
    }
  }

  function copyPreviousFrame() {
    if (safeFrameIndex === 0) return;
    if (trackingDocument(document)) {
      const previous = document.frames[safeFrameIndex - 1];
      const frame = cloneDocument(document.frames[safeFrameIndex]);
      frame.objects = previous.objects.map((item) => ({
        ...cloneDocument(item),
        bbox: [...item.bbox],
        source_frame: sourceFrameFor(frame),
        review_state: "unverified"
      }));
      frame.review_state = "unverified";
      replaceTrackingFrame(frame);
    } else {
      const previous = document.frames[safeFrameIndex - 1];
      const frame = cloneDocument(document.frames[safeFrameIndex]);
      frame.state = previous.state;
      frame.ball = previous.ball ? structuredClone(previous.ball) : null;
      frame.review_state = "unverified";
      replaceBallFrame(frame);
    }
  }

  function verifiedDocument(): AnnotationDocument | null {
    if (!annotator.trim() || !allFramesVerified) return null;
    const next = cloneDocument(document);
    next.verification = {
      status: "verified",
      annotator: annotator.trim(),
      reviewed_at: new Date().toISOString()
    };
    return next;
  }

  async function saveDraft() {
    const next = cloneDocument(document);
    next.verification = { status: "draft", annotator: annotator.trim() || null, reviewed_at: null };
    onChange(next);
    await onSave(next);
  }

  async function verifyAndSave() {
    const next = verifiedDocument();
    if (!next) return;
    onChange(next);
    await onSave(next);
  }

  async function evaluate() {
    const next = document.verification.status === "verified" && allFramesVerified
      ? document
      : verifiedDocument();
    if (!next) return;
    onChange(next);
    await onEvaluate(next);
  }

  function download() {
    const blob = new Blob([JSON.stringify(document, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = window.document.createElement("a");
    link.href = url;
    link.download = `${mode}-ground-truth-${String(document.source?.run_id ?? "run")}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const trackingFrame = trackingDocument(document) ? document.frames[safeFrameIndex] : null;
  const ballFrame = !trackingDocument(document) ? document.frames[safeFrameIndex] : null;
  const selected = trackingFrame && selectedObject != null ? trackingFrame.objects[selectedObject] : null;
  const ballMetrics = metrics || null;

  return (
    <div className="ground-truth-editor">
      <div className="annotation-toolbar">
        <div className="annotation-progress">
          <span>{mode === "tracking" ? "Identity labels" : "Ball labels"}</span>
          <strong>{verifiedCount} / {frames.length} frames verified</strong>
          <span className="annotation-progress-track"><span style={{ width: `${progress * 100}%` }} /></span>
        </div>
        <div className="annotation-tool-group">
          {mode === "tracking" ? (
            <>
              <button className={tool === "select" ? "active" : ""} onClick={() => setTool("select")} title="Select and move annotation" type="button"><MousePointer2 size={16} /></button>
              <button className={tool === "draw" ? "active" : ""} onClick={() => setTool("draw")} title="Draw missing identity" type="button"><Plus size={16} /></button>
            </>
          ) : <button className="active" title="Mark ball position" type="button"><CircleDot size={16} /></button>}
          <button disabled={!historyRef.current.length} onClick={undo} title="Undo annotation edit" type="button"><Undo2 size={16} /></button>
          <button disabled={safeFrameIndex === 0} onClick={copyPreviousFrame} title="Copy previous frame annotations" type="button"><Copy size={16} /></button>
          <button onClick={download} title="Download annotation JSON" type="button"><Download size={16} /></button>
        </div>
      </div>

      <div className="annotation-workspace">
        <aside className="annotation-frame-list">
          {frames.map((frame, index) => (
            <button
              className={`${index === safeFrameIndex ? "active" : ""} ${frameVerified(mode, frame) ? "verified" : ""}`}
              key={frame.frame}
              onClick={() => { setFrameIndex(index); setSelectedObject(null); }}
              type="button"
            >
              <span>F{frame.frame}</span>
              <small>{mode === "tracking" ? `${(frame as TrackingGroundTruthFrame).objects.length} objects` : titleCase((frame as BallGroundTruthFrame).state)}</small>
              {frameVerified(mode, frame) ? <Check size={13} /> : <span className="annotation-pending" />}
            </button>
          ))}
        </aside>

        <div className="annotation-canvas-column">
          <div className="annotation-stage" style={{ aspectRatio: `${width} / ${height}` }}>
            <video muted preload="auto" ref={videoRef} src={videoSrc} />
            <svg
              onPointerMove={handlePointerMove}
              onPointerUp={finishPointerAction}
              onPointerLeave={finishPointerAction}
              preserveAspectRatio="none"
              ref={svgRef}
              viewBox={`0 0 ${width} ${height}`}
            >
              <rect className="annotation-hit-area" height={height} onPointerDown={startCanvasAction} width={width} x="0" y="0" />
              {trackingFrame?.objects.map((item, index) => {
                const [x1, y1, x2, y2] = item.bbox;
                const active = selectedObject === index;
                const color = item.team === 1 ? "#ef553f" : item.team === 2 ? "#f4f7f9" : "#f5c542";
                const handleSize = Math.max(4, width * 0.004);
                return (
                  <g className={active ? "active" : ""} key={`${item.identity_id}-${index}`}>
                    <rect
                      className="annotation-person-box"
                      fill={`${color}18`}
                      height={y2 - y1}
                      onPointerDown={(event) => startObjectMove(event, index)}
                      stroke={color}
                      width={x2 - x1}
                      x={x1}
                      y={y1}
                    />
                    <rect className="annotation-label-bg" fill={color} height={Math.max(16, height * 0.022)} width={Math.max(64, item.identity_id.length * 8)} x={x1} y={Math.max(0, y1 - Math.max(16, height * 0.022))} />
                    <text className="annotation-label" x={x1 + 4} y={Math.max(12, y1 - 4)}>{item.identity_id}</text>
                    {active ? ([
                      ["nw", x1, y1], ["ne", x2, y1], ["sw", x1, y2], ["se", x2, y2]
                    ] as const).map(([corner, x, y]) => (
                      <circle className="annotation-resize-handle" cx={x} cy={y} key={corner} onPointerDown={(event) => startResize(event, index, corner)} r={handleSize} />
                    )) : null}
                  </g>
                );
              })}
              {previewBox ? <rect className="annotation-preview-box" height={previewBox[3] - previewBox[1]} width={previewBox[2] - previewBox[0]} x={previewBox[0]} y={previewBox[1]} /> : null}
              {ballFrame?.ball ? (
                <g className="annotation-ball-marker">
                  <circle cx={ballFrame.ball.center[0]} cy={ballFrame.ball.center[1]} r={Math.max(7, width * 0.006)} />
                  <line x1={ballFrame.ball.center[0] - width * 0.012} x2={ballFrame.ball.center[0] + width * 0.012} y1={ballFrame.ball.center[1]} y2={ballFrame.ball.center[1]} />
                  <line x1={ballFrame.ball.center[0]} x2={ballFrame.ball.center[0]} y1={ballFrame.ball.center[1] - width * 0.012} y2={ballFrame.ball.center[1] + width * 0.012} />
                </g>
              ) : null}
            </svg>
          </div>
          <div className="annotation-navigation">
            <button className="button icon-button" disabled={safeFrameIndex === 0} onClick={() => setFrameIndex((value) => Math.max(0, value - 1))} title="Previous annotated frame" type="button"><ArrowLeft size={16} /></button>
            <span><strong>Frame {currentFrame?.frame ?? "-"}</strong><small>{((currentFrame?.frame || 0) / Math.max(fps, 0.001)).toFixed(2)}s</small></span>
            <button className="button icon-button" disabled={safeFrameIndex >= frames.length - 1} onClick={() => setFrameIndex((value) => Math.min(frames.length - 1, value + 1))} title="Next annotated frame" type="button"><ArrowRight size={16} /></button>
            <button className="button" onClick={verifyCurrentFrame} type="button"><Check size={16} /> Verify frame</button>
          </div>
        </div>

        <aside className="annotation-inspector">
          {mode === "tracking" ? (
            selected ? (
              <>
                <div><span className="eyebrow">Selected annotation</span><h3>{selected.identity_id}</h3></div>
                <label><span>Stable identity</span><input className="input" onChange={(event) => updateSelectedObject({ identity_id: event.target.value, review_state: "unverified" })} value={selected.identity_id} /></label>
                <div className="review-control"><span>Team</span><div className="segmented-control"><button className={selected.team === 1 ? "active" : ""} onClick={() => updateSelectedObject({ team: 1, review_state: "unverified" })} type="button">Team 1</button><button className={selected.team === 2 ? "active" : ""} onClick={() => updateSelectedObject({ team: 2, review_state: "unverified" })} type="button">Team 2</button></div></div>
                <label><span>Participant role</span><select className="select" onChange={(event) => updateSelectedObject({ role_name: event.target.value as TrackingGroundTruthObject["role_name"], review_state: "unverified" })} value={selected.role_name || "player"}>{ROLE_OPTIONS.map((role) => <option key={role} value={role}>{titleCase(role)}</option>)}</select></label>
                <label><span>New identity for draw</span><input className="input" onChange={(event) => setNewIdentity(event.target.value)} placeholder="identity-player-name" value={newIdentity} /></label>
                <button className="button danger" onClick={removeSelectedObject} type="button"><Trash2 size={16} /> Remove false detection</button>
              </>
            ) : (
              <div className="annotation-empty"><MousePointer2 size={20} /><span>Select a box or draw a missing identity.</span></div>
            )
          ) : ballFrame ? (
            <>
              <div><span className="eyebrow">Ball state</span><h3>Frame {ballFrame.frame}</h3></div>
              <div className="ball-state-grid">
                {(["visible", "occluded", "out_of_frame", "uncertain"] as BallGroundTruthState[]).map((state) => (
                  <button className={ballFrame.state === state ? "active" : ""} key={state} onClick={() => setBallState(state)} type="button">{titleCase(state)}</button>
                ))}
              </div>
              {ballFrame.ball ? (
                <>
                  <div className="annotation-coordinate-grid"><span>X<strong>{ballFrame.ball.center[0].toFixed(1)}</strong></span><span>Y<strong>{ballFrame.ball.center[1].toFixed(1)}</strong></span></div>
                  <label className="toggle-row"><span>Airborne</span><input checked={Boolean(ballFrame.ball.airborne)} onChange={(event) => updateBallFrame({ ball: { ...ballFrame.ball!, airborne: event.target.checked } })} type="checkbox" /></label>
                  <label><span>Estimated height (cm)</span><input className="input" min="0" onChange={(event) => updateBallFrame({ ball: { ...ballFrame.ball!, height_cm: Number(event.target.value) } })} type="number" value={ballFrame.ball.height_cm ?? 0} /></label>
                  <button className="button danger" onClick={() => updateBallFrame({ state: "uncertain", ball: null })} type="button"><Trash2 size={16} /> Clear marker</button>
                </>
              ) : null}
            </>
          ) : null}
        </aside>
      </div>

      {ballMetrics ? (
        <div className="ball-ground-truth-metrics">
          {(["precision", "recall", "f1", "median_center_error_pixels", "p95_center_error_pixels"] as const).map((key) => (
            <div key={key}><span>{titleCase(key)}</span><strong>{String(ballMetrics[key] ?? "-")}{key.includes("pixels") && ballMetrics[key] != null ? " px" : key.includes("precision") || key === "recall" || key === "f1" ? "%" : ""}</strong></div>
          ))}
        </div>
      ) : null}

      <div className="annotation-footer">
        <label><span>Annotator</span><input className="input" onChange={(event) => setAnnotator(event.target.value)} placeholder="Name or email" value={annotator} /></label>
        <span className={`annotation-document-state ${document.verification.status}`}>{titleCase(document.verification.status)}</span>
        <button className="button" disabled={busy} onClick={() => void saveDraft()} type="button"><Save size={16} /> Save draft</button>
        <button className="button" disabled={busy || !annotator.trim() || !allFramesVerified} onClick={() => void verifyAndSave()} title="Available after every frame is manually verified" type="button"><ShieldCheck size={16} /> Verify all & save</button>
        <button className="button primary" disabled={busy || !annotator.trim() || !allFramesVerified} onClick={() => void evaluate()} type="button"><ShieldCheck size={16} /> Evaluate</button>
      </div>
    </div>
  );
}
