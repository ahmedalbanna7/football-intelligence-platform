import {
  AlertTriangle,
  CircleDot,
  CheckCircle2,
  Eye,
  FileCheck2,
  GitMerge,
  History,
  Plus,
  RefreshCw,
  RotateCcw,
  Scissors,
  ShieldCheck,
  Undo2,
  Upload,
  UserCheck,
  UserRoundCog,
  XCircle
} from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { GroundTruthAnnotationEditor } from "./GroundTruthAnnotationEditor";
import { ActionWithHelp, HelpTip, LabelWithHelp, PaginatedTable } from "./UiPrimitives";
import type {
  BallGroundTruthDocument,
  TrackingGroundTruthDocument,
  TrackReviewItem,
  TrackingQualityResponse
} from "./types";

type QualityTab = "overview" | "annotation" | "review" | "history";

type TrackingQualityPanelProps = {
  matchId: number;
  runId: number;
  videoObject: string;
  fps: number;
  onLayersChanged?: () => void;
  onRunQueued?: () => void;
};

function percent(value?: number | null) {
  return value == null ? "-" : `${(value * 100).toFixed(1)}%`;
}

function metricPercent(value?: number | null) {
  return value == null ? "Ground truth required" : `${value.toFixed(1)}%`;
}

function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function normalizeTrackingGroundTruth(payload: Record<string, unknown>): TrackingGroundTruthDocument {
  if (Array.isArray(payload.frames)) return payload as TrackingGroundTruthDocument;
  if (!Array.isArray(payload.observations)) {
    throw new Error("Tracking ground truth must contain frames or observations.");
  }
  const grouped = new Map<number, TrackingGroundTruthDocument["frames"][number]>();
  for (const value of payload.observations) {
    const item = value as Record<string, unknown>;
    const frame = Number(item.frame);
    const bbox = (item.bbox || item.bbox_xyxy) as number[];
    if (!Number.isFinite(frame) || !Array.isArray(bbox) || bbox.length !== 4) continue;
    const frameItem = grouped.get(frame) || { frame, objects: [] };
    frameItem.objects.push({
      identity_id: String(item.identity_id ?? ""),
      bbox: bbox.map(Number) as [number, number, number, number],
      source_frame: item.source_frame == null ? null : Number(item.source_frame),
      team: item.team == null ? null : Number(item.team),
      role_name: (item.role_name as TrackReviewItem["role_name"]) || "player",
      review_state: item.review_state === "verified" ? "verified" : "unverified"
    });
    grouped.set(frame, frameItem);
  }
  return {
    ...payload,
    schema_version: "tracking_ground_truth.v2",
    verification: (payload.verification || { status: "draft" }) as TrackingGroundTruthDocument["verification"],
    frames: [...grouped.values()].sort((first, second) => first.frame - second.frame)
  } as TrackingGroundTruthDocument;
}

function QualityBar({ value }: { value: number }) {
  const normalized = Math.max(0, Math.min(1, value));
  const level = normalized >= 0.82 ? "good" : normalized >= 0.68 ? "review" : "risk";
  return (
    <span className="quality-bar" title={`${(normalized * 100).toFixed(1)}%`}>
      <span className={`quality-bar-fill ${level}`} style={{ width: `${normalized * 100}%` }} />
    </span>
  );
}

function RiskBadge({ risk }: { risk: TrackReviewItem["switch_risk"] }) {
  return <span className={`quality-risk ${risk}`}>{risk}</span>;
}

export function TrackingQualityPanel({
  matchId,
  runId,
  videoObject,
  fps,
  onLayersChanged,
  onRunQueued
}: TrackingQualityPanelProps) {
  const [data, setData] = useState<TrackingQualityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [tab, setTab] = useState<QualityTab>("overview");
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [riskFilter, setRiskFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [teamFilter, setTeamFilter] = useState("all");
  const [roleFilter, setRoleFilter] = useState("all");
  const [mergeTarget, setMergeTarget] = useState<number | null>(null);
  const [splitFrame, setSplitFrame] = useState<number | null>(null);
  const [playerId, setPlayerId] = useState<number | null>(null);
  const [participantRole, setParticipantRole] = useState<TrackReviewItem["role_name"]>("player");
  const [note, setNote] = useState("");
  const [groundTruth, setGroundTruth] = useState<TrackingGroundTruthDocument | null>(null);
  const [ballGroundTruth, setBallGroundTruth] = useState<BallGroundTruthDocument | null>(null);
  const [annotationKind, setAnnotationKind] = useState<"tracking" | "ball">("tracking");
  const [ballGroundTruthMetrics, setBallGroundTruthMetrics] = useState<Record<string, unknown> | null>(null);
  const [groundTruthName, setGroundTruthName] = useState<string | null>(null);
  const [iouThreshold, setIouThreshold] = useState(0.5);
  const [groundTruthStart, setGroundTruthStart] = useState(0);
  const [groundTruthEnd, setGroundTruthEnd] = useState(0);
  const [groundTruthStep, setGroundTruthStep] = useState(5);
  const [groundTruthScenario, setGroundTruthScenario] = useState("crossing");
  const [groundTruthCamera, setGroundTruthCamera] = useState("tactical");
  const [groundTruthCritical, setGroundTruthCritical] = useState(true);
  const [releaseClipSize, setReleaseClipSize] = useState(750);
  const [releaseOverlap, setReleaseOverlap] = useState(0);
  const [releasePlan, setReleasePlan] = useState<Awaited<ReturnType<typeof api.buildTrackingReleasePlan>> | null>(null);
  const [criticalRanges, setCriticalRanges] = useState<Awaited<ReturnType<typeof api.suggestTrackingCriticalRanges>> | null>(null);
  const [releaseRunningIndex, setReleaseRunningIndex] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  async function load() {
    setLoading(true);
    try {
      const response = await api.getTrackingQuality(matchId, runId);
      setData(response);
      setMessage(null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not load tracking quality.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setData(null);
    setSelectedTrackId(null);
    setGroundTruth(null);
    setBallGroundTruth(null);
    setBallGroundTruthMetrics(null);
    setAnnotationKind("tracking");
    void load();
  }, [matchId, runId]);

  useEffect(() => {
    if (!data?.tracks.length || selectedTrackId != null) return;
    const firstReview = data.tracks.find((track) => track.switch_risk !== "low");
    setSelectedTrackId((firstReview || data.tracks[0]).track_id);
  }, [data, selectedTrackId]);

  const filteredTracks = useMemo(() => {
    return (data?.tracks || []).filter((track) => {
      if (riskFilter !== "all" && track.switch_risk !== riskFilter) return false;
      if (statusFilter !== "all" && track.status !== statusFilter) return false;
      if (teamFilter !== "all" && String(track.team ?? "unknown") !== teamFilter) return false;
      if (roleFilter !== "all" && track.role_name !== roleFilter) return false;
      return true;
    });
  }, [data, riskFilter, statusFilter, teamFilter, roleFilter]);

  const selectedTrack = data?.tracks.find((track) => track.track_id === selectedTrackId) || null;
  const lastAvailableFrame = useMemo(
    () => Math.max(0, ...(data?.tracks.map((track) => track.last_frame ?? 0) || [0])),
    [data]
  );

  useEffect(() => {
    setGroundTruthStart(0);
    setGroundTruthEnd(lastAvailableFrame);
  }, [runId, lastAvailableFrame]);

  useEffect(() => {
    if (!selectedTrack) return;
    setSplitFrame(selectedTrack.first_frame ?? null);
    setPlayerId(selectedTrack.assigned_player_id ?? null);
    setParticipantRole(selectedTrack.role_name);
    setMergeTarget(
      data?.tracks.find((track) => track.track_id !== selectedTrack.track_id)?.track_id ?? null
    );
  }, [selectedTrackId]);

  function seekToFrame(frame?: number | null) {
    if (frame == null || !videoRef.current) return;
    videoRef.current.currentTime = frame / Math.max(fps, 0.001);
    void videoRef.current.play();
  }

  async function applyCorrection(
    action: string,
    extra: {
      target_track_id?: number | null;
      split_frame?: number | null;
      assigned_player_id?: number | null;
      assigned_team_number?: number | null;
      assigned_role_name?: string | null;
    } = {}
  ) {
    if (!selectedTrack) return;
    setBusy(true);
    setMessage(`Applying ${titleCase(action)}...`);
    try {
      const response = await api.createTrackCorrection(matchId, runId, {
        action,
        source_track_id: selectedTrack.track_id,
        note: note || null,
        ...extra
      });
      setData(response);
      setMessage(`${titleCase(action)} saved. Corrected layers are ready.`);
      onLayersChanged?.();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Correction failed.");
    } finally {
      setBusy(false);
    }
  }

  async function recalculate() {
    setBusy(true);
    setMessage("Rebuilding corrected visual layers...");
    try {
      const response = await api.recalculateTrackingQuality(matchId, runId);
      setData(response.quality);
      setMessage(`${response.corrections_applied} corrections applied to ${response.tracks_count} tracks.`);
      onLayersChanged?.();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recalculation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function undo(correctionId: number) {
    setBusy(true);
    try {
      const response = await api.undoTrackCorrection(matchId, runId, correctionId);
      setData(response);
      setMessage("Correction undone and visual layers rebuilt.");
      onLayersChanged?.();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Undo failed.");
    } finally {
      setBusy(false);
    }
  }

  async function selectGroundTruth(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()) as Record<string, unknown>;
      if (parsed.schema_version === "ball_ground_truth.v1") {
        setBallGroundTruth(parsed as BallGroundTruthDocument);
        setAnnotationKind("ball");
      } else {
        setGroundTruth(normalizeTrackingGroundTruth(parsed));
        setAnnotationKind("tracking");
      }
      setGroundTruthName(file.name);
      setTab("annotation");
      setMessage("Ground truth loaded in the annotation editor.");
    } catch {
      setGroundTruth(null);
      setGroundTruthName(null);
      setMessage("Ground truth must be a valid JSON file.");
    }
  }

  async function benchmark(document: TrackingGroundTruthDocument | null = groundTruth) {
    if (!document) return;
    setBusy(true);
    setMessage("Measuring IDF1, HOTA, switches, and fragmentation...");
    try {
      const response = await api.benchmarkTrackingQuality(
        matchId,
        runId,
        document,
        iouThreshold
      );
      setData(response.quality);
      setMessage("Ground-truth benchmark completed.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Benchmark failed.");
    } finally {
      setBusy(false);
    }
  }

  async function openTrackingGroundTruthDraft() {
    setBusy(true);
    setMessage("Building selected ground-truth clip...");
    try {
      const response = await api.buildTrackingGroundTruthDraft(matchId, runId, {
        start_frame: groundTruthStart,
        end_frame: groundTruthEnd,
        sample_every_frames: groundTruthStep,
        scenario: groundTruthScenario,
        camera_style: groundTruthCamera,
        critical: groundTruthCritical
      });
      setGroundTruth(response.ground_truth as TrackingGroundTruthDocument);
      setAnnotationKind("tracking");
      setTab("annotation");
      setMessage(`${response.frame_count} identity frames opened with ${response.annotation_count} annotations.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Ground-truth draft failed.");
    } finally {
      setBusy(false);
    }
  }

  async function openBallGroundTruthDraft() {
    setBusy(true);
    setMessage("Building ball annotation frames...");
    try {
      const response = await api.buildBallGroundTruthDraft(matchId, runId, {
        start_frame: groundTruthStart,
        end_frame: groundTruthEnd,
        sample_every_frames: groundTruthStep,
        scenario: groundTruthScenario,
        camera_style: groundTruthCamera,
        critical: groundTruthCritical
      });
      setBallGroundTruth(response.ground_truth);
      setBallGroundTruthMetrics(null);
      setAnnotationKind("ball");
      setTab("annotation");
      setMessage(`${response.frame_count} ball frames opened with ${response.candidate_count} model candidates.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Ball ground-truth draft failed.");
    } finally {
      setBusy(false);
    }
  }

  async function loadSavedAnnotations(kind: "tracking" | "ball") {
    setBusy(true);
    try {
      if (kind === "tracking") {
        const response = await api.getTrackingGroundTruth(matchId, runId);
        setGroundTruth(response.ground_truth);
      } else {
        const response = await api.getBallGroundTruth(matchId, runId);
        setBallGroundTruth(response.ground_truth);
        setBallGroundTruthMetrics(response.metrics || null);
      }
      setAnnotationKind(kind);
      setTab("annotation");
      setMessage(`Saved ${kind} ground truth loaded.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Saved annotation could not be loaded.");
    } finally {
      setBusy(false);
    }
  }

  async function saveTrackingAnnotations(document: TrackingGroundTruthDocument) {
    setBusy(true);
    try {
      const response = await api.saveTrackingGroundTruth(matchId, runId, document);
      setGroundTruth(response.ground_truth);
      await load();
      setMessage(`${response.validation.verified_frames}/${response.validation.frame_count} identity frames verified and saved.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Tracking annotations could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  async function saveBallAnnotations(document: BallGroundTruthDocument) {
    setBusy(true);
    try {
      const response = await api.saveBallGroundTruth(matchId, runId, document);
      setBallGroundTruth(response.ground_truth);
      await load();
      setMessage(`${response.validation.verified_frames}/${response.validation.frame_count} ball frames verified and saved.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Ball annotations could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  async function evaluateBallAnnotations(document: BallGroundTruthDocument) {
    setBusy(true);
    setMessage("Measuring ball localization, coverage, continuity, and airborne state...");
    try {
      const response = await api.benchmarkBallGroundTruth(matchId, runId, document);
      setBallGroundTruth(document);
      setBallGroundTruthMetrics(response.metrics);
      await load();
      setMessage(`Ball benchmark completed: ${String(response.metrics.f1 ?? "-")}% F1.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Ball benchmark failed.");
    } finally {
      setBusy(false);
    }
  }

  async function buildReleasePlan() {
    setBusy(true);
    setMessage("Building full-video release test plan...");
    try {
      const response = await api.buildTrackingReleasePlan(matchId, {
        clip_size: releaseClipSize,
        overlap_frames: releaseOverlap,
        camera_style: groundTruthCamera,
        critical_ranges: criticalRanges?.ranges.map((range) => ({
          start_frame: range.start_frame,
          end_frame: range.end_frame,
          scenario: range.scenarios[0] || "critical"
        }))
      });
      setReleasePlan(response);
      setMessage(`${response.clips_count} release clips planned across ${response.source_frames} source frames.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Release plan failed.");
    } finally {
      setBusy(false);
    }
  }

  async function suggestCriticalRanges() {
    setBusy(true);
    setMessage("Finding crossings, crowding, re-entry, and identity-risk windows...");
    try {
      const response = await api.suggestTrackingCriticalRanges(matchId, runId);
      setCriticalRanges(response);
      setMessage(`${response.ranges.length} critical windows found from ${response.events_detected} review signals.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not suggest critical clips.");
    } finally {
      setBusy(false);
    }
  }

  function selectCriticalRange(range: NonNullable<typeof criticalRanges>["ranges"][number]) {
    setGroundTruthStart(range.start_frame);
    setGroundTruthEnd(range.end_frame);
    setGroundTruthScenario(range.scenarios[0] || "crossing");
    setGroundTruthCritical(true);
    seekToFrame(range.peak_frame);
    setMessage(`Ground-truth window selected: F${range.start_frame}-F${range.end_frame}.`);
  }

  async function queueReleaseClip(clip: NonNullable<typeof releasePlan>["clips"][number]) {
    setBusy(true);
    setReleaseRunningIndex(clip.index);
    setMessage(`Queueing release clip ${clip.index + 1}...`);
    try {
      const queued = await api.runMatchAnalysisPlus(matchId, clip.run_request);
      setMessage(`Clip ${clip.index + 1} queued as run #${queued.id}. Its result is saved with this match.`);
      onRunQueued?.();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not queue release clip.");
    } finally {
      setBusy(false);
      setReleaseRunningIndex(null);
    }
  }

  if (loading && !data) {
    return <section className="tracking-quality-shell quality-loading">Loading tracking quality...</section>;
  }

  if (!data) {
    return (
      <section className="tracking-quality-shell quality-loading">
        <AlertTriangle size={18} /> {message || "Tracking quality is unavailable."}
      </section>
    );
  }

  const assessment = data.assessment;
  const benchmarkReady = assessment.benchmark_status === "measured";

  return (
    <section className="tracking-quality-shell">
      <div className="quality-heading">
        <div>
          <span className="eyebrow">Identity assurance</span>
          <h2 className="section-title title-with-help">Tracking Quality Gate <HelpTip text="Measures identity stability, builds verified ground truth, and blocks release when tracking quality is not proven." /></h2>
        </div>
        <div className="quality-heading-actions">
          <span className={`quality-state ${assessment.status}`}>
            {assessment.status === "approved" ? <ShieldCheck size={15} /> : <AlertTriangle size={15} />}
            {titleCase(assessment.status)}
          </span>
          <ActionWithHelp help="Reload the saved quality state without changing run artifacts."><button className="button icon-button" disabled={busy} onClick={() => void load()} title="Refresh quality" type="button"><RefreshCw size={16} /></button></ActionWithHelp>
          <ActionWithHelp help="Recompute tracking confidence, risks and review flags from the saved run data."><button className="button" disabled={busy} onClick={() => void recalculate()} type="button"><RotateCcw size={16} /> Recalculate</button></ActionWithHelp>
        </div>
      </div>

      <div className="quality-tabs" role="tablist">
        <button className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")} role="tab" type="button">
          <FileCheck2 size={16} /> Quality Overview
        </button>
        <button className={tab === "annotation" ? "active" : ""} onClick={() => setTab("annotation")} role="tab" type="button">
          <CircleDot size={16} /> Ground Truth <span>{
            Number(Boolean(groundTruth || data?.annotations?.tracking))
              + Number(Boolean(ballGroundTruth || data?.annotations?.ball))
          }</span>
        </button>
        <button className={tab === "review" ? "active" : ""} onClick={() => setTab("review")} role="tab" type="button">
          <Eye size={16} /> Track Review <span>{assessment.tracks_needing_review}</span>
        </button>
        <button className={tab === "history" ? "active" : ""} onClick={() => setTab("history")} role="tab" type="button">
          <History size={16} /> Corrections <span>{data.corrections.filter((item) => !item.undone).length}</span>
        </button>
      </div>

      {message ? <div className="quality-message">{message}</div> : null}

      {tab === "overview" ? (
        <div className="quality-overview">
          <div className="quality-metric-grid">
            <div><span>Identity confidence</span><strong>{percent(assessment.average_identity_confidence)}</strong><small>Current run health</small></div>
            <div><span>Suspected switches</span><strong>{assessment.suspected_id_switches}</strong><small>Heuristic review flags</small></div>
            <div><span>Fragmented tracks</span><strong>{assessment.fragmented_tracks}</strong><small>Current run health</small></div>
            <div><span>Needs review</span><strong>{assessment.tracks_needing_review}</strong><small>Tracks above risk threshold</small></div>
            <div className={benchmarkReady ? "measured" : "pending"}><span>IDF1</span><strong>{metricPercent(assessment.idf1)}</strong><small>Ground-truth benchmark</small></div>
            <div className={benchmarkReady ? "measured" : "pending"}><span>HOTA</span><strong>{metricPercent(assessment.hota)}</strong><small>Ground-truth benchmark</small></div>
            <div className={benchmarkReady ? "measured" : "pending"}><span>Exact ID switches</span><strong>{assessment.id_switches ?? "-"}</strong><small>Ground-truth benchmark</small></div>
            <div className={benchmarkReady ? "measured" : "pending"}><span>Fragmentation</span><strong>{assessment.fragmentation ?? "-"}</strong><small>Ground-truth benchmark</small></div>
          </div>

          <div className="quality-runtime-row">
            <span><strong>{assessment.tracker_engine || "-"}</strong> tracker</span>
            <span className={assessment.reid_enabled ? "runtime-on" : "runtime-off"}>
              {assessment.reid_enabled ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
              Re-ID {assessment.reid_enabled ? "active" : "inactive"}
            </span>
            <span>{assessment.reid_model || "No Re-ID model reported"}</span>
          </div>

          <div className={`release-gate-panel ${assessment.release_gate_status}`}>
            <div className="release-gate-heading">
              <div>
                <span className="eyebrow">Release decision</span>
                <h3>{assessment.release_gate_status === "passed" ? "Tracking release passed" : "Tracking release blocked"}</h3>
              </div>
              <span className={`quality-state ${assessment.release_gate_status}`}>
                {assessment.release_gate_status === "passed" ? <ShieldCheck size={15} /> : <AlertTriangle size={15} />}
                {titleCase(assessment.release_gate_status || "not_ready")}
              </span>
            </div>
            {assessment.release_gate?.conditions?.length ? (
              <div className="release-condition-grid">
                {assessment.release_gate.conditions.map((condition) => (
                  <div className={condition.passed ? "passed" : "blocked"} key={condition.code}>
                    {condition.passed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
                    <span><strong>{condition.label}</strong><small>{condition.missing?.length ? `Missing: ${condition.missing.join(", ")}` : `Actual: ${String(condition.actual ?? "not measured")} · Required: ${String(condition.required ?? "-")}`}</small></span>
                  </div>
                ))}
              </div>
            ) : <p>Evaluate verified clips to activate the release decision.</p>}
          </div>

          <div className="release-plan-builder">
            <div><span className="eyebrow">Full-video coverage</span><strong>500-1000 frame clip plan</strong></div>
            <label><LabelWithHelp help="Number of source frames in each validation clip; production validation uses 500 to 1000.">Clip size</LabelWithHelp><input className="input" max="1000" min="500" onChange={(event) => setReleaseClipSize(Number(event.target.value))} step="50" type="number" value={releaseClipSize} /></label>
            <label><LabelWithHelp help="Frames shared by consecutive clips so identities crossing clip boundaries are still checked.">Overlap</LabelWithHelp><input className="input" max="250" min="0" onChange={(event) => setReleaseOverlap(Number(event.target.value))} step="25" type="number" value={releaseOverlap} /></label>
            <ActionWithHelp help="Search run observations for crowding, crossings, re-entry and cross-team identity risk."><button className="button" disabled={busy} onClick={() => void suggestCriticalRanges()} type="button"><AlertTriangle size={16} /> Find critical clips</button></ActionWithHelp>
            <ActionWithHelp help="Split the full processed interval into repeatable validation clips with the selected size and overlap."><button className="button" disabled={busy || releaseOverlap >= releaseClipSize} onClick={() => void buildReleasePlan()} type="button"><FileCheck2 size={16} /> Build plan</button></ActionWithHelp>
            {releasePlan ? <span className="release-plan-result"><strong>{releasePlan.clips_count}</strong> clips · {releasePlan.source_frames} frames · {releasePlan.fps} FPS</span> : null}
            {releasePlan ? (
              <div className="release-plan-clips">
                {releasePlan.clips.map((clip) => (
                  <div className={clip.critical ? "critical" : ""} key={clip.index}>
                    <span><strong>Clip {clip.index + 1}</strong><small>F{clip.start_frame}-F{clip.end_frame} · {clip.frame_count} frames · {clip.scenarios.map(titleCase).join(", ")}</small></span>
                    <button className="button" disabled={busy} onClick={() => void queueReleaseClip(clip)} type="button">
                      {releaseRunningIndex === clip.index ? <RefreshCw className="spin" size={15} /> : <CheckCircle2 size={15} />} Run clip
                    </button>
                  </div>
                ))}
              </div>
            ) : null}
          </div>

          {criticalRanges?.ranges.length ? (
            <div className="critical-range-review">
              <div className="critical-range-heading">
                <div><span className="eyebrow">Automated triage</span><strong>Critical identity windows</strong></div>
                <span>{criticalRanges.ranges.length} windows · {criticalRanges.events_detected} signals</span>
              </div>
              <div className="critical-range-grid">
                {criticalRanges.ranges.map((range) => (
                  <button key={`${range.start_frame}-${range.end_frame}`} onClick={() => selectCriticalRange(range)} type="button">
                    <span><strong>F{range.start_frame}-F{range.end_frame}</strong><small>Peak F{range.peak_frame} · Tracks {range.track_ids.join(", ")}</small></span>
                    <span className={range.severity >= 0.9 ? "high" : "medium"}>{range.scenarios.map(titleCase).join(", ")}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <div className="quality-ground-truth-builder">
            <label><LabelWithHelp help="First local frame included in this annotation clip.">Clip start</LabelWithHelp><input className="input" min="0" onChange={(event) => setGroundTruthStart(Number(event.target.value))} type="number" value={groundTruthStart} /></label>
            <label><LabelWithHelp help="Last local frame included in this annotation clip.">Clip end</LabelWithHelp><input className="input" max={lastAvailableFrame} min={groundTruthStart} onChange={(event) => setGroundTruthEnd(Number(event.target.value))} type="number" value={groundTruthEnd} /></label>
            <label><LabelWithHelp help="Annotation sampling interval. A value of 1 includes every frame; larger values reduce manual work.">Sample every</LabelWithHelp><input className="input" max="120" min="1" onChange={(event) => setGroundTruthStep(Number(event.target.value))} type="number" value={groundTruthStep} /></label>
            <label><LabelWithHelp help="Describes the identity challenge represented by this clip for quality reporting.">Scenario</LabelWithHelp><select className="select" onChange={(event) => setGroundTruthScenario(event.target.value)} value={groundTruthScenario}><option value="crossing">Crossing</option><option value="crowding">Crowding</option><option value="reentry">Long re-entry</option><option value="baseline">Baseline</option></select></label>
            <label><LabelWithHelp help="Records whether this sample comes from a wide tactical view or a close/moving camera.">Camera</LabelWithHelp><select className="select" onChange={(event) => setGroundTruthCamera(event.target.value)} value={groundTruthCamera}><option value="tactical">Tactical</option><option value="close_or_moving">Close / moving</option></select></label>
            <label className="quality-critical-check"><input checked={groundTruthCritical} onChange={(event) => setGroundTruthCritical(event.target.checked)} type="checkbox" /><LabelWithHelp help="Marks this clip as mandatory for the release gate because it contains a difficult case.">Critical clip</LabelWithHelp></label>
            <ActionWithHelp help="Create an editable identity draft for people, teams and participant roles in the selected frames."><button className="button" disabled={busy || groundTruthEnd < groundTruthStart} onClick={() => void openTrackingGroundTruthDraft()} type="button"><Eye size={16} /> Identity editor</button></ActionWithHelp>
            <ActionWithHelp help="Create an editable ball draft with visibility, position, airborne state and estimated height."><button className="button" disabled={busy || groundTruthEnd < groundTruthStart} onClick={() => void openBallGroundTruthDraft()} type="button"><CircleDot size={16} /> Ball editor</button></ActionWithHelp>
          </div>

          <div className="quality-benchmark-band">
            <label className="button" htmlFor={`ground-truth-${runId}`}>
              <Upload size={16} /> {groundTruthName || "Ground truth JSON"}
            </label>
            <input accept="application/json,.json" id={`ground-truth-${runId}`} onChange={(event) => void selectGroundTruth(event)} type="file" />
            <button className="button" disabled={busy} onClick={() => void loadSavedAnnotations("tracking")} type="button"><FileCheck2 size={16} /> Saved identities</button>
            <button className="button" disabled={busy} onClick={() => void loadSavedAnnotations("ball")} type="button"><CircleDot size={16} /> Saved ball</button>
          </div>

          <PaginatedTable
            className="quality-table-wrap quality-table"
            headers={["Track", "Team", "Role", "Identity", "Re-ID", "Motion", "Fragments", "Risk", "Status", "Review"]}
            rows={data.tracks}
            style={{ marginTop: 12 }}
            renderRow={(track) => (
              <tr key={track.track_id}>
                <td><strong>#{track.track_id}</strong></td>
                <td>{track.team ? `Team ${track.team}` : "Unknown"}</td>
                <td><strong>{titleCase(track.role_name)}</strong><small className="table-subline">{percent(track.role_confidence)}{track.role_locked ? " · locked" : " · learning"}</small></td>
                <td><div className="quality-score"><QualityBar value={track.identity_confidence} /><span>{percent(track.identity_confidence)}</span></div></td>
                <td>{percent(track.reid_confidence)}</td><td>{percent(track.motion_consistency)}</td>
                <td>{track.fragment_count}</td><td><RiskBadge risk={track.switch_risk} /></td>
                <td>{titleCase(track.status)}</td>
                <td><button className="button icon-button" onClick={() => { setSelectedTrackId(track.track_id); setTab("review"); }} title={`Review track ${track.track_id}`} type="button"><Eye size={15} /></button></td>
              </tr>
            )}
          />
        </div>
      ) : null}

      {tab === "annotation" ? (
        <div className="quality-annotation-tab">
          <div className="annotation-mode-bar">
            <div className="segmented-control">
              <button className={annotationKind === "tracking" ? "active" : ""} onClick={() => setAnnotationKind("tracking")} type="button"><Eye size={16} /> Identities</button>
              <button className={annotationKind === "ball" ? "active" : ""} onClick={() => setAnnotationKind("ball")} type="button"><CircleDot size={16} /> Ball</button>
            </div>
            {annotationKind === "tracking" ? (
              <label className="annotation-iou"><span>IoU threshold</span><input className="input" max="0.95" min="0.05" onChange={(event) => setIouThreshold(Number(event.target.value))} step="0.05" type="number" value={iouThreshold} /></label>
            ) : null}
            <button className="button" disabled={busy} onClick={() => void loadSavedAnnotations(annotationKind)} type="button"><FileCheck2 size={16} /> Load saved</button>
          </div>
          {annotationKind === "tracking" && groundTruth ? (
            <GroundTruthAnnotationEditor
              busy={busy}
              document={groundTruth}
              fps={fps}
              mode="tracking"
              onChange={(document) => setGroundTruth(document as TrackingGroundTruthDocument)}
              onEvaluate={async (document) => {
                const tracking = document as TrackingGroundTruthDocument;
                setGroundTruth(tracking);
                await benchmark(tracking);
              }}
              onSave={async (document) => saveTrackingAnnotations(document as TrackingGroundTruthDocument)}
              videoFrameOffset={data.source_start_frame || 0}
              videoSrc={api.objectUrl(data.annotation_video_object || videoObject)}
            />
          ) : annotationKind === "ball" && ballGroundTruth ? (
            <GroundTruthAnnotationEditor
              busy={busy}
              document={ballGroundTruth}
              fps={fps}
              metrics={ballGroundTruthMetrics}
              mode="ball"
              onChange={(document) => setBallGroundTruth(document as BallGroundTruthDocument)}
              onEvaluate={async (document) => evaluateBallAnnotations(document as BallGroundTruthDocument)}
              onSave={async (document) => saveBallAnnotations(document as BallGroundTruthDocument)}
              videoFrameOffset={data.source_start_frame || 0}
              videoSrc={api.objectUrl(data.annotation_video_object || videoObject)}
            />
          ) : (
            <div className="annotation-start-state">
              {annotationKind === "tracking" ? <Eye size={24} /> : <CircleDot size={24} />}
              <strong>{annotationKind === "tracking" ? "Identity annotation set" : "Ball annotation set"}</strong>
              <div>
                <button className="button primary" disabled={busy} onClick={() => void (annotationKind === "tracking" ? openTrackingGroundTruthDraft() : openBallGroundTruthDraft())} type="button"><Plus size={16} /> Create draft</button>
                <button className="button" disabled={busy} onClick={() => void loadSavedAnnotations(annotationKind)} type="button"><FileCheck2 size={16} /> Load saved</button>
              </div>
            </div>
          )}
        </div>
      ) : null}

      {tab === "review" ? (
        <div className="track-review">
          <div className="track-review-filters">
            <label className="review-filter"><LabelWithHelp help="Filter tracks by estimated identity-switch risk.">Risk</LabelWithHelp><select className="select" onChange={(event) => setRiskFilter(event.target.value)} value={riskFilter}>
              <option value="all">All risk levels</option><option value="high">High risk</option><option value="medium">Medium risk</option><option value="low">Low risk</option>
            </select></label>
            <label className="review-filter"><LabelWithHelp help="Show pending or already corrected review states.">State</LabelWithHelp><select className="select" onChange={(event) => setStatusFilter(event.target.value)} value={statusFilter}>
              <option value="all">All review states</option><option value="pending">Pending</option><option value="approved">Approved</option><option value="rejected">Rejected</option><option value="merged">Merged</option><option value="split">Split</option>
            </select></label>
            <label className="review-filter"><LabelWithHelp help="Limit review to one team or identities whose team is still unknown.">Team</LabelWithHelp><select className="select" onChange={(event) => setTeamFilter(event.target.value)} value={teamFilter}>
              <option value="all">Both teams</option><option value="1">Team 1</option><option value="2">Team 2</option><option value="unknown">Unknown team</option>
            </select></label>
            <label className="review-filter"><LabelWithHelp help="Limit review to players, goalkeepers, officials, or people outside the pitch.">Role</LabelWithHelp><select className="select" onChange={(event) => setRoleFilter(event.target.value)} value={roleFilter}>
              <option value="all">All roles</option><option value="player">Players</option><option value="goalkeeper">Goalkeepers</option><option value="referee">Referees</option><option value="assistant_referee">Assistant referees</option><option value="staff_outside_pitch">Staff / outside pitch</option>
            </select></label>
            <span>{filteredTracks.length} tracks</span>
          </div>

          <div className="track-review-workspace">
            <aside className="track-review-list">
              {filteredTracks.map((track) => (
                <button className={track.track_id === selectedTrackId ? "active" : ""} key={track.track_id} onClick={() => setSelectedTrackId(track.track_id)} type="button">
                  <span><strong>Track {track.track_id}</strong><small>{titleCase(track.role_name)} · {track.team ? `Team ${track.team}` : "No team"}</small></span>
                  <span><RiskBadge risk={track.switch_risk} /><small>{percent(track.identity_confidence)}</small></span>
                </button>
              ))}
            </aside>

            <div className="track-review-video">
              <video controls preload="metadata" ref={videoRef} src={api.objectUrl(videoObject)} />
              {selectedTrack ? (
                <div className="track-timeline">
                  <button onClick={() => seekToFrame(selectedTrack.first_frame)} type="button">Start {selectedTrack.first_frame ?? "-"}</button>
                  {selectedTrack.observations.slice(0, 12).map((observation) => (
                    <button key={observation.frame} onClick={() => seekToFrame(observation.frame)} type="button">{(observation.frame / Math.max(fps, 0.001)).toFixed(1)}s</button>
                  ))}
                  <button onClick={() => seekToFrame(selectedTrack.last_frame)} type="button">End {selectedTrack.last_frame ?? "-"}</button>
                </div>
              ) : null}
            </div>

            <aside className="track-review-inspector">
              {selectedTrack ? (
                <>
                  <div className="review-track-title">
                    <div><span className="eyebrow">Selected identity</span><h3>Track {selectedTrack.track_id}</h3></div>
                    <RiskBadge risk={selectedTrack.switch_risk} />
                  </div>
                  <div className="review-confidence-grid">
                    <div><span>Identity</span><strong>{percent(selectedTrack.identity_confidence)}</strong><QualityBar value={selectedTrack.identity_confidence} /></div>
                    <div><span>Re-ID</span><strong>{percent(selectedTrack.reid_confidence)}</strong><QualityBar value={selectedTrack.reid_confidence} /></div>
                    <div><span>Motion</span><strong>{percent(selectedTrack.motion_consistency)}</strong><QualityBar value={selectedTrack.motion_consistency} /></div>
                    <div><span>Team</span><strong>{percent(selectedTrack.team_consistency)}</strong><QualityBar value={selectedTrack.team_consistency} /></div>
                    <div><span>Role</span><strong>{percent(selectedTrack.role_confidence)}</strong><QualityBar value={selectedTrack.role_confidence} /></div>
                  </div>
                  <div className="role-assurance-row"><strong>{titleCase(selectedTrack.role_name)}</strong><span>{selectedTrack.role_locked ? "Locked by temporal evidence" : "Still learning"}</span></div>
                  <div className="review-issues">
                    {selectedTrack.issue_codes.length ? selectedTrack.issue_codes.map((issue) => <span key={issue}>{titleCase(issue)}</span>) : <span className="clear">No quality flags</span>}
                  </div>
                  <div className="review-crops">
                    {selectedTrack.crop_objects.map((crop) => (
                      <button key={crop.object_name} onClick={() => seekToFrame(crop.frame)} title={`Frame ${crop.frame}`} type="button">
                        <img alt={`Track ${selectedTrack.track_id} at frame ${crop.frame}`} src={api.objectUrl(crop.object_name)} />
                        <span>F{crop.frame}</span>
                      </button>
                    ))}
                    {!selectedTrack.crop_objects.length ? <div className="review-empty">No crops in this legacy run.</div> : null}
                  </div>
                  <div className="review-actions-primary">
                    <button className="button primary" disabled={busy} onClick={() => void applyCorrection("approve")} type="button"><CheckCircle2 size={16} /> Approve</button>
                    <button className="button danger" disabled={busy} onClick={() => void applyCorrection("reject")} type="button"><XCircle size={16} /> Reject</button>
                  </div>
                  <div className="review-control">
                    <span>Team correction</span>
                    <div className="segmented-control">
                      <button className={selectedTrack.team === 1 ? "active" : ""} disabled={busy} onClick={() => void applyCorrection("change_team", { assigned_team_number: 1 })} type="button">Team 1</button>
                      <button className={selectedTrack.team === 2 ? "active" : ""} disabled={busy} onClick={() => void applyCorrection("change_team", { assigned_team_number: 2 })} type="button">Team 2</button>
                    </div>
                  </div>
                  <div className="review-control">
                    <label htmlFor={`role-${runId}`}><LabelWithHelp help="Correct and lock the participant role used by tracking, analytics and reports.">Participant role</LabelWithHelp></label>
                    <div className="control-row"><select className="select" id={`role-${runId}`} value={participantRole} onChange={(event) => setParticipantRole(event.target.value as TrackReviewItem["role_name"])}><option value="player">Player</option><option value="goalkeeper">Goalkeeper</option><option value="referee">Referee</option><option value="assistant_referee">Assistant referee</option><option value="staff_outside_pitch">Staff / outside pitch</option></select><button className="button icon-button" disabled={busy || participantRole === selectedTrack.role_name} onClick={() => void applyCorrection("change_role", { assigned_role_name: participantRole })} title="Confirm participant role" type="button"><UserRoundCog size={16} /></button></div>
                    <small>{selectedTrack.role_evidence.length ? selectedTrack.role_evidence.map(titleCase).join(" · ") : "No role evidence recorded"}</small>
                  </div>
                  <div className="review-control">
                    <label htmlFor={`player-${runId}`}><LabelWithHelp help="Link this canonical track to a roster player.">Player identity</LabelWithHelp></label>
                    <div className="control-row"><select className="select" id={`player-${runId}`} onChange={(event) => setPlayerId(Number(event.target.value) || null)} value={playerId || ""}><option value="">Select player</option>{data.players.map((player) => <option key={player.id} value={player.id}>{player.jersey_number != null ? `#${player.jersey_number} ` : ""}{player.name}</option>)}</select><button className="button icon-button" disabled={!playerId || busy} onClick={() => void applyCorrection("assign_player", { assigned_player_id: playerId })} title="Assign player" type="button"><UserCheck size={16} /></button></div>
                  </div>
                  <div className="review-control">
                    <label htmlFor={`merge-${runId}`}><LabelWithHelp help="Combine this track with another track when both represent the same person.">Merge into</LabelWithHelp></label>
                    <div className="control-row"><select className="select" id={`merge-${runId}`} onChange={(event) => setMergeTarget(Number(event.target.value))} value={mergeTarget || ""}>{data.tracks.filter((track) => track.track_id !== selectedTrack.track_id).map((track) => <option key={track.track_id} value={track.track_id}>Track {track.track_id}</option>)}</select><button className="button icon-button" disabled={!mergeTarget || busy} onClick={() => void applyCorrection("merge", { target_track_id: mergeTarget })} title="Merge track" type="button"><GitMerge size={16} /></button></div>
                  </div>
                  <div className="review-control">
                    <label htmlFor={`split-${runId}`}><LabelWithHelp help="Create a new identity from this frame when the current track switched to a different person.">Split at frame</LabelWithHelp></label>
                    <div className="control-row"><input className="input" id={`split-${runId}`} max={selectedTrack.last_frame ?? undefined} min={(selectedTrack.first_frame ?? 0) + 1} onChange={(event) => setSplitFrame(Number(event.target.value))} type="number" value={splitFrame ?? ""} /><button className="button icon-button" disabled={!splitFrame || busy} onClick={() => void applyCorrection("split", { split_frame: splitFrame })} title="Split track" type="button"><Scissors size={16} /></button></div>
                  </div>
                  <label className="review-note"><LabelWithHelp help="Optional audit note saved with the next correction.">Review note</LabelWithHelp><textarea className="textarea" onChange={(event) => setNote(event.target.value)} rows={2} value={note} /></label>
                </>
              ) : <div className="review-empty">No track matches the selected filters.</div>}
            </aside>
          </div>
        </div>
      ) : null}

      {tab === "history" ? (
        <div className="quality-history">
          {!data.corrections.length ? <div className="review-empty">No corrections saved for this run.</div> : (
            <PaginatedTable
              headers={["Action", "Source", "Target / value", "Note", "Created", "State", "Undo"]}
              rows={data.corrections}
              renderRow={(correction) => (
                <tr key={correction.id}>
                  <td>{titleCase(correction.action)}</td>
                  <td>{correction.source_track_id != null ? `Track ${correction.source_track_id}` : "-"}</td>
                  <td>{correction.target_track_id != null ? `Track ${correction.target_track_id}` : correction.split_frame != null ? `Frame ${correction.split_frame}` : correction.assigned_role_name ? titleCase(correction.assigned_role_name) : correction.assigned_team_number != null ? `Team ${correction.assigned_team_number}` : correction.assigned_player_id != null ? `Player ${correction.assigned_player_id}` : "-"}</td>
                  <td>{correction.note || "-"}</td><td>{correction.created_at ? new Date(correction.created_at).toLocaleString() : "-"}</td>
                  <td>{correction.undone ? "Undone" : "Active"}</td>
                  <td><button className="button icon-button" disabled={busy || correction.undone} onClick={() => void undo(correction.id)} title="Undo correction" type="button"><Undo2 size={16} /></button></td>
                </tr>
              )}
            />
          )}
        </div>
      ) : null}
    </section>
  );
}
