import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  FileCheck2,
  GitMerge,
  History,
  RefreshCw,
  RotateCcw,
  Scissors,
  ShieldCheck,
  Undo2,
  Upload,
  UserCheck,
  XCircle
} from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type {
  TrackReviewItem,
  TrackingQualityResponse
} from "./types";

type QualityTab = "overview" | "review" | "history";

type TrackingQualityPanelProps = {
  matchId: number;
  runId: number;
  videoObject: string;
  fps: number;
  onLayersChanged?: () => void;
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
  onLayersChanged
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
  const [mergeTarget, setMergeTarget] = useState<number | null>(null);
  const [splitFrame, setSplitFrame] = useState<number | null>(null);
  const [playerId, setPlayerId] = useState<number | null>(null);
  const [note, setNote] = useState("");
  const [groundTruth, setGroundTruth] = useState<Record<string, unknown> | null>(null);
  const [groundTruthName, setGroundTruthName] = useState<string | null>(null);
  const [iouThreshold, setIouThreshold] = useState(0.5);
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
      return true;
    });
  }, [data, riskFilter, statusFilter, teamFilter]);

  const selectedTrack = data?.tracks.find((track) => track.track_id === selectedTrackId) || null;

  useEffect(() => {
    if (!selectedTrack) return;
    setSplitFrame(selectedTrack.first_frame ?? null);
    setPlayerId(selectedTrack.assigned_player_id ?? null);
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
      setGroundTruth(parsed);
      setGroundTruthName(file.name);
      setMessage("Ground truth loaded and ready to evaluate.");
    } catch {
      setGroundTruth(null);
      setGroundTruthName(null);
      setMessage("Ground truth must be a valid JSON file.");
    }
  }

  async function benchmark() {
    if (!groundTruth) return;
    setBusy(true);
    setMessage("Measuring IDF1, HOTA, switches, and fragmentation...");
    try {
      const response = await api.benchmarkTrackingQuality(
        matchId,
        runId,
        groundTruth,
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
          <h2 className="section-title">Tracking Quality Gate</h2>
        </div>
        <div className="quality-heading-actions">
          <span className={`quality-state ${assessment.status}`}>
            {assessment.status === "approved" ? <ShieldCheck size={15} /> : <AlertTriangle size={15} />}
            {titleCase(assessment.status)}
          </span>
          <button className="button icon-button" disabled={busy} onClick={() => void load()} title="Refresh quality" type="button">
            <RefreshCw size={16} />
          </button>
          <button className="button" disabled={busy} onClick={() => void recalculate()} type="button">
            <RotateCcw size={16} /> Recalculate
          </button>
        </div>
      </div>

      <div className="quality-tabs" role="tablist">
        <button className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")} role="tab" type="button">
          <FileCheck2 size={16} /> Quality Overview
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

          <div className="quality-benchmark-band">
            <label className="button" htmlFor={`ground-truth-${runId}`}>
              <Upload size={16} /> {groundTruthName || "Ground truth JSON"}
            </label>
            <input accept="application/json,.json" id={`ground-truth-${runId}`} onChange={(event) => void selectGroundTruth(event)} type="file" />
            <label>
              <span>IoU threshold</span>
              <input className="input" max="0.95" min="0.05" onChange={(event) => setIouThreshold(Number(event.target.value))} step="0.05" type="number" value={iouThreshold} />
            </label>
            <button className="button primary" disabled={!groundTruth || busy || !assessment.predictions_object} onClick={() => void benchmark()} type="button">
              <ShieldCheck size={16} /> Evaluate
            </button>
          </div>

          <div className="quality-table-wrap">
            <table className="table quality-table">
              <thead><tr><th>Track</th><th>Team</th><th>Identity</th><th>Re-ID</th><th>Motion</th><th>Fragments</th><th>Risk</th><th>Status</th><th>Review</th></tr></thead>
              <tbody>
                {data.tracks.map((track) => (
                  <tr key={track.track_id}>
                    <td><strong>#{track.track_id}</strong></td>
                    <td>{track.team ? `Team ${track.team}` : "Unknown"}</td>
                    <td><div className="quality-score"><QualityBar value={track.identity_confidence} /><span>{percent(track.identity_confidence)}</span></div></td>
                    <td>{percent(track.reid_confidence)}</td>
                    <td>{percent(track.motion_consistency)}</td>
                    <td>{track.fragment_count}</td>
                    <td><RiskBadge risk={track.switch_risk} /></td>
                    <td>{titleCase(track.status)}</td>
                    <td><button className="button icon-button" onClick={() => { setSelectedTrackId(track.track_id); setTab("review"); }} title={`Review track ${track.track_id}`} type="button"><Eye size={15} /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {tab === "review" ? (
        <div className="track-review">
          <div className="track-review-filters">
            <select className="select" onChange={(event) => setRiskFilter(event.target.value)} value={riskFilter}>
              <option value="all">All risk levels</option><option value="high">High risk</option><option value="medium">Medium risk</option><option value="low">Low risk</option>
            </select>
            <select className="select" onChange={(event) => setStatusFilter(event.target.value)} value={statusFilter}>
              <option value="all">All review states</option><option value="pending">Pending</option><option value="approved">Approved</option><option value="rejected">Rejected</option><option value="merged">Merged</option><option value="split">Split</option>
            </select>
            <select className="select" onChange={(event) => setTeamFilter(event.target.value)} value={teamFilter}>
              <option value="all">Both teams</option><option value="1">Team 1</option><option value="2">Team 2</option><option value="unknown">Unknown team</option>
            </select>
            <span>{filteredTracks.length} tracks</span>
          </div>

          <div className="track-review-workspace">
            <aside className="track-review-list">
              {filteredTracks.map((track) => (
                <button className={track.track_id === selectedTrackId ? "active" : ""} key={track.track_id} onClick={() => setSelectedTrackId(track.track_id)} type="button">
                  <span><strong>Track {track.track_id}</strong><small>{track.team ? `Team ${track.team}` : "Unknown team"}</small></span>
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
                  </div>
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
                    <label htmlFor={`player-${runId}`}>Player identity</label>
                    <div className="control-row"><select className="select" id={`player-${runId}`} onChange={(event) => setPlayerId(Number(event.target.value) || null)} value={playerId || ""}><option value="">Select player</option>{data.players.map((player) => <option key={player.id} value={player.id}>{player.jersey_number != null ? `#${player.jersey_number} ` : ""}{player.name}</option>)}</select><button className="button icon-button" disabled={!playerId || busy} onClick={() => void applyCorrection("assign_player", { assigned_player_id: playerId })} title="Assign player" type="button"><UserCheck size={16} /></button></div>
                  </div>
                  <div className="review-control">
                    <label htmlFor={`merge-${runId}`}>Merge into</label>
                    <div className="control-row"><select className="select" id={`merge-${runId}`} onChange={(event) => setMergeTarget(Number(event.target.value))} value={mergeTarget || ""}>{data.tracks.filter((track) => track.track_id !== selectedTrack.track_id).map((track) => <option key={track.track_id} value={track.track_id}>Track {track.track_id}</option>)}</select><button className="button icon-button" disabled={!mergeTarget || busy} onClick={() => void applyCorrection("merge", { target_track_id: mergeTarget })} title="Merge track" type="button"><GitMerge size={16} /></button></div>
                  </div>
                  <div className="review-control">
                    <label htmlFor={`split-${runId}`}>Split at frame</label>
                    <div className="control-row"><input className="input" id={`split-${runId}`} max={selectedTrack.last_frame ?? undefined} min={(selectedTrack.first_frame ?? 0) + 1} onChange={(event) => setSplitFrame(Number(event.target.value))} type="number" value={splitFrame ?? ""} /><button className="button icon-button" disabled={!splitFrame || busy} onClick={() => void applyCorrection("split", { split_frame: splitFrame })} title="Split track" type="button"><Scissors size={16} /></button></div>
                  </div>
                  <label className="review-note"><span>Review note</span><textarea className="textarea" onChange={(event) => setNote(event.target.value)} rows={2} value={note} /></label>
                </>
              ) : <div className="review-empty">No track matches the selected filters.</div>}
            </aside>
          </div>
        </div>
      ) : null}

      {tab === "history" ? (
        <div className="quality-history">
          {!data.corrections.length ? <div className="review-empty">No corrections saved for this run.</div> : (
            <table className="table">
              <thead><tr><th>Action</th><th>Source</th><th>Target / value</th><th>Note</th><th>Created</th><th>State</th><th>Undo</th></tr></thead>
              <tbody>{data.corrections.map((correction) => (
                <tr key={correction.id}>
                  <td>{titleCase(correction.action)}</td>
                  <td>{correction.source_track_id != null ? `Track ${correction.source_track_id}` : "-"}</td>
                  <td>{correction.target_track_id != null ? `Track ${correction.target_track_id}` : correction.split_frame != null ? `Frame ${correction.split_frame}` : correction.assigned_team_number != null ? `Team ${correction.assigned_team_number}` : correction.assigned_player_id != null ? `Player ${correction.assigned_player_id}` : "-"}</td>
                  <td>{correction.note || "-"}</td>
                  <td>{correction.created_at ? new Date(correction.created_at).toLocaleString() : "-"}</td>
                  <td>{correction.undone ? "Undone" : "Active"}</td>
                  <td><button className="button icon-button" disabled={busy || correction.undone} onClick={() => void undo(correction.id)} title="Undo correction" type="button"><Undo2 size={16} /></button></td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
      ) : null}
    </section>
  );
}
