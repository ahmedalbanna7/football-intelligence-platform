import {
  Activity,
  BarChart3,
  Bot,
  CalendarDays,
  FileText,
  Gauge,
  Lightbulb,
  RefreshCw,
  Settings,
  Shield,
  Upload,
  Users,
  Video
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type {
  IdentityAssignment,
  MatchSummary,
  Player,
  PrimaryTeamProfile,
  ProcessingSummary,
  FirstAnalysisSummary,
  MatchAnalysisPlusRun,
  ReportResponse,
  RouteKey,
  RosterPlayer,
  Team,
  TrackSummary
} from "./types";

const zones = [
  "GK",
  "LB",
  "LCB",
  "CB",
  "RCB",
  "RB",
  "LWB",
  "RWB",
  "DM",
  "LCM",
  "CM",
  "RCM",
  "AM",
  "LW",
  "RW",
  "ST",
  "LST",
  "RST"
];

const preferredSides = ["left", "center", "right", "either", "unknown"];

const positionLabels = [
  "Goalkeeper",
  "Fullback",
  "Wingback",
  "Center Back",
  "Defensive Midfielder",
  "Central Midfielder",
  "Attacking Midfielder",
  "Winger",
  "Second Striker",
  "Striker"
];

const routes: Array<{
  key: RouteKey;
  label: string;
  icon: ReactNode;
  subtitle: string;
}> = [
  { key: "dashboard", label: "Dashboard", icon: <Gauge size={18} />, subtitle: "Operational overview" },
  { key: "settings", label: "Settings", icon: <Settings size={18} />, subtitle: "System defaults" },
  { key: "my-team", label: "My Team", icon: <Shield size={18} />, subtitle: "Primary club identity" },
  { key: "teams", label: "Teams", icon: <Users size={18} />, subtitle: "Opponent history and rosters" },
  { key: "matches", label: "Matches", icon: <Video size={18} />, subtitle: "Upload and prepare games" },
  { key: "analysis", label: "Analysis", icon: <Activity size={18} />, subtitle: "Pipeline outputs" },
  { key: "first-analysis", label: "First Analysis", icon: <BarChart3 size={18} />, subtitle: "Annotated football_analysis-main video" },
  { key: "match-analysis-plus", label: "Match Analysis +", icon: <BarChart3 size={18} />, subtitle: "Full player, ball, team, and pitch analysis" },
  { key: "reports", label: "Reports", icon: <FileText size={18} />, subtitle: "Team and player reports" },
  { key: "agent", label: "Agent", icon: <Bot size={18} />, subtitle: "Coach assistant" },
  { key: "recommendations", label: "Recommendations", icon: <Lightbulb size={18} />, subtitle: "Match and season ideas" }
];

function useAsyncData<T>(loader: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    loader()
      .then((payload) => {
        if (active) setData(payload);
      })
      .catch((err: Error) => {
        if (active) setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [...deps, nonce]);

  return { data, loading, error, refresh: () => setNonce((value) => value + 1) };
}

function statusBadge(status?: string | null) {
  if (!status) return <span className="badge warn">unknown</span>;
  const tone = status.includes("fail") || status.includes("error") ? "error" : status === "processed" ? "" : "warn";
  return <span className={`badge ${tone}`}>{status}</span>;
}

function StatCard({ title, value, label }: { title: string; value: ReactNode; label?: string }) {
  return (
    <div className="card">
      <h3 className="card-title">{title}</h3>
      <p className="metric">{value}</p>
      {label ? <p className="metric-label">{label}</p> : null}
    </div>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

function AppShell({
  route,
  setRoute,
  children
}: {
  route: RouteKey;
  setRoute: (route: RouteKey) => void;
  children: ReactNode;
}) {
  const active = routes.find((item) => item.key === route) || routes[0];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">SI</div>
          <div>
            <div className="brand-title">Sports Intelligence</div>
            <div className="brand-subtitle">Football analysis platform</div>
          </div>
        </div>
        <nav className="nav">
          {routes.map((item) => (
            <button
              className={`nav-button ${route === item.key ? "active" : ""}`}
              key={item.key}
              onClick={() => setRoute(item.key)}
              type="button"
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          Sprint 1 UI foundation. Built around the current RabbitMQ, YOLO, team, lineup,
          identity, reports, and future live-processing surface.
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div>
            <h1 className="page-title">{active.label}</h1>
            <p className="page-subtitle">{active.subtitle}</p>
          </div>
          <div className="toolbar">
            <span className="badge">API {api.baseUrl}</span>
          </div>
        </header>
        <div className="content">{children}</div>
      </main>
    </div>
  );
}

function DashboardPage({ setRoute }: { setRoute: (route: RouteKey) => void }) {
  const matches = useAsyncData(() => api.listMatches(8), []);
  const latest = matches.data?.items?.[0];
  const summary = latest?.latest_processing_job?.result_summary;

  return (
    <>
      <section className="grid four">
        <StatCard title="Matches" value={matches.data?.items?.length ?? "-"} label="latest loaded" />
        <StatCard title="Latest status" value={latest ? statusBadge(latest.status) : "-"} label={latest?.title} />
        <StatCard title="YOLO detections" value={summary?.detections?.detections_count ?? "-"} label={summary?.detections?.engine || "waiting"} />
        <StatCard title="Identity resolved" value={summary?.tactical_identity?.resolved_count ?? "-"} label="from tactical layer" />
      </section>

      <section className="split">
        <div className="section">
          <div className="section-header">
            <h2 className="section-title">Recent Matches</h2>
            <button className="button" onClick={matches.refresh} type="button">
              <RefreshCw size={16} /> Refresh
            </button>
          </div>
          <MatchesTable matches={matches.data?.items || []} onOpen={() => setRoute("analysis")} />
        </div>
        <div className="section">
          <h2 className="section-title">Next Actions</h2>
          <div className="card">
            <div className="grid">
              <button className="button primary" onClick={() => setRoute("matches")} type="button">
                <Upload size={16} /> Upload match
              </button>
              <button className="button" onClick={() => setRoute("my-team")} type="button">
                <Shield size={16} /> Configure primary team
              </button>
              <button className="button" onClick={() => setRoute("teams")} type="button">
                <Users size={16} /> Build tactical roster
              </button>
            </div>
          </div>
          <RecommendationsPreview match={latest} />
        </div>
      </section>
    </>
  );
}

function MatchesTable({
  matches,
  onOpen,
  onDelete
}: {
  matches: MatchSummary[];
  onOpen?: (id: number) => void;
  onDelete?: (id: number) => void;
}) {
  if (!matches.length) return <Empty>No matches yet. Upload a video to start the pipeline.</Empty>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Title</th>
            <th>Status</th>
            <th>Type</th>
            <th>Scope</th>
              <th>Analysis</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {matches.map((match) => (
            <tr key={match.id}>
              <td>{match.id}</td>
              <td>{match.title}</td>
              <td>{statusBadge(match.status)}</td>
              <td>{match.match_context?.match_type || "-"}</td>
              <td>{match.match_context?.analysis_scope || "-"}</td>
              <td>
                {match.latest_match_analysis_run
                  ? `M+ ${match.latest_match_analysis_run.status}`
                  : match.latest_processing_job?.result_summary?.detections?.engine || "-"}
              </td>
              <td>
                <button className="button" onClick={() => onOpen?.(match.id)} type="button">
                  Open
                </button>
                {onDelete ? (
                  <button className="button danger" onClick={() => onDelete(match.id)} type="button">
                    Delete
                  </button>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SettingsPage() {
  const yolo = useAsyncData(() => api.getYoloStatus(), []);

  return (
    <section className="grid two">
      <div className="card">
        <h2 className="section-title">Analysis Defaults</h2>
        <div className="form-grid" style={{ marginTop: 14 }}>
          <label className="field">
            <span className="label">YOLO mode</span>
            <select className="select" defaultValue="auto">
              <option value="auto">auto</option>
              <option value="real">real</option>
              <option value="stub">stub</option>
            </select>
          </label>
          <label className="field">
            <span className="label">Default model</span>
            <input className="input" defaultValue="yolo11n.pt" />
          </label>
          <label className="field">
            <span className="label">Confidence</span>
            <input className="input" defaultValue="0.25" />
          </label>
          <label className="field">
            <span className="label">Image size</span>
            <input className="input" defaultValue="640" />
          </label>
        </div>
        <p className="muted small">These controls are prepared for a future settings API. Current values are backend env driven.</p>
      </div>
      <div className="card">
        <div className="section-header">
          <h2 className="section-title">YOLO Runtime</h2>
          <button className="button" onClick={yolo.refresh} type="button">
            <RefreshCw size={16} /> Refresh
          </button>
        </div>
        <div className="grid" style={{ marginTop: 14 }}>
          {yolo.error ? <div className="empty">{yolo.error}</div> : null}
          <div className="toolbar">
            <span className={`badge ${yolo.data?.status === "ok" ? "" : "error"}`}>
              {yolo.data?.status || "loading"}
            </span>
            <span className="badge">{yolo.data?.engine || "engine"}</span>
            <span className="badge">{yolo.data?.device || "device"}</span>
          </div>
          <div className="table-wrap">
            <table className="table">
              <tbody>
                <tr><th>Mode</th><td>{yolo.data?.mode || "-"}</td></tr>
                <tr><th>Model</th><td>{yolo.data?.model || "-"}</td></tr>
                <tr><th>Model file</th><td>{String(yolo.data?.model_file_exists ?? "-")}</td></tr>
                <tr><th>Loaded</th><td>{String(yolo.data?.model_loaded ?? "-")}</td></tr>
                <tr><th>Classes</th><td>{yolo.data?.allowed_class_names?.join(", ") || "-"}</td></tr>
                <tr><th>Confidence</th><td>{yolo.data?.confidence ?? "-"}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="card">
        <h2 className="section-title">Service Map</h2>
        <div className="grid" style={{ marginTop: 14 }}>
          {["Backend API :8000", "Frontend :5173", "RabbitMQ :5672 / :15672", "MinIO :9000 / :9001", "PostgreSQL", "Video Worker"].map((item) => (
            <div className="toolbar" key={item}>
              <span className="badge">{item}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function MyTeamPage() {
  const profile = useAsyncData(() => api.getPrimaryTeam(), []);
  const roster = useAsyncData(() => api.listPrimaryTeamPlayers(), []);
  const [message, setMessage] = useState<string | null>(null);

  async function saveProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage(null);
    const form = new FormData(event.currentTarget);
    await api.savePrimaryTeam(form);
    setMessage("Primary team saved.");
    profile.refresh();
  }

  async function savePlayer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    await api.savePrimaryTeamPlayer({
      player_name: String(form.get("player_name") || ""),
      shirt_number: Number(form.get("shirt_number")),
      primary_zone: String(form.get("primary_zone") || ""),
      secondary_zones: form.getAll("secondary_zones").map(String).filter(Boolean),
      position_label: String(form.get("position_label") || ""),
      preferred_side: String(form.get("preferred_side") || "unknown"),
      notes: String(form.get("notes") || "")
    });
    setMessage("Player saved.");
    event.currentTarget.reset();
    roster.refresh();
  }

  async function deletePlayer(entryId: number) {
    await api.deletePrimaryTeamPlayer(entryId);
    setMessage("Player deleted.");
    roster.refresh();
  }

  return (
    <section className="grid">
      <div className="grid two">
        <div className="card">
          <h2 className="section-title">Primary Team Profile</h2>
          <form className="form-grid" onSubmit={saveProfile} style={{ marginTop: 14 }}>
            <label className="field full">
              <span className="label">Team name</span>
              <input className="input" name="team_name" defaultValue={profile.data?.team_name || ""} required />
            </label>
            <label className="field">
              <span className="label">Primary kit image</span>
              <input className="input" name="primary_kit_image" type="file" accept="image/*" />
            </label>
            <label className="field">
              <span className="label">Alternate kit image</span>
              <input className="input" name="alternate_kit_image" type="file" accept="image/*" />
            </label>
            <div className="field full">
              <button className="button primary" type="submit">
                <Shield size={16} /> Save profile
              </button>
            </div>
          </form>
          {message ? <p className="badge">{message}</p> : null}
        </div>
        <div className="card">
          <h2 className="section-title">Current Kit References</h2>
          <div className="grid" style={{ marginTop: 14 }}>
            <div>
              <div className="label">Primary</div>
              <p className="muted">{profile.data?.primary_kit_image_object_name || "Not uploaded"}</p>
            </div>
            <div>
              <div className="label">Alternate</div>
              <p className="muted">{profile.data?.alternate_kit_image_object_name || "Not uploaded"}</p>
            </div>
          </div>
        </div>
      </div>

      <section className="split">
        <div className="section">
          <div className="section-header">
            <h2 className="section-title">My Players</h2>
            <button className="button" onClick={roster.refresh} type="button">
              <RefreshCw size={16} /> Refresh
            </button>
          </div>
          <PrimaryRosterTable players={roster.data?.items || []} onDelete={deletePlayer} />
        </div>
        <div className="card">
          <h2 className="section-title">Add My Player</h2>
          <form className="form-grid" onSubmit={savePlayer} style={{ marginTop: 14 }}>
            <label className="field full">
              <span className="label">Name</span>
              <input className="input" name="player_name" required />
            </label>
            <label className="field">
              <span className="label">Number</span>
              <input className="input" name="shirt_number" min="1" type="number" required />
            </label>
            <label className="field">
              <span className="label">Primary zone</span>
              <select className="select" name="primary_zone" defaultValue="CM" required>
                {zones.map((zone) => <option key={zone}>{zone}</option>)}
              </select>
            </label>
            <label className="field">
              <span className="label">Secondary zones</span>
              <select className="select" name="secondary_zones" multiple>
                {zones.map((zone) => <option key={zone}>{zone}</option>)}
              </select>
            </label>
            <label className="field">
              <span className="label">Preferred side</span>
              <select className="select" name="preferred_side" defaultValue="unknown">
                {preferredSides.map((side) => <option key={side}>{side}</option>)}
              </select>
            </label>
            <label className="field">
              <span className="label">Position label</span>
              <select className="select" name="position_label" defaultValue="Central Midfielder">
                {positionLabels.map((label) => <option key={label}>{label}</option>)}
              </select>
            </label>
            <label className="field full">
              <span className="label">Notes</span>
              <textarea className="textarea" name="notes" />
            </label>
            <button className="button primary" type="submit">Save player</button>
          </form>
        </div>
      </section>
    </section>
  );
}

function PrimaryRosterTable({ players, onDelete }: { players: RosterPlayer[]; onDelete: (id: number) => void }) {
  if (!players.length) return <Empty>No primary team players yet.</Empty>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>#</th>
            <th>Name</th>
            <th>Zone</th>
            <th>Secondary</th>
            <th>Side</th>
            <th>Label</th>
            <th>Notes</th>
            <th>Context</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {players.map((player) => (
            <tr key={player.id}>
              <td>{player.shirt_number}</td>
              <td>{player.player_name}</td>
              <td>{player.primary_zone || player.position || "-"}</td>
              <td>{player.secondary_zones?.join(", ") || "-"}</td>
              <td>{player.preferred_side || "-"}</td>
              <td>{player.position_label || "-"}</td>
              <td>{player.notes || "-"}</td>
              <td>{player.team_context}</td>
              <td>
                <button className="button danger" onClick={() => onDelete(player.id)} type="button">
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TeamsPage() {
  const teams = useAsyncData(() => api.listTeams("opponent"), []);
  const [selectedTeamId, setSelectedTeamId] = useState<number | null>(null);
  const selected = selectedTeamId || teams.data?.items?.[0]?.id || null;
  const players = useAsyncData(() => (selected ? api.listTeamPlayers(selected) : Promise.resolve({ team: { id: 0, name: "" }, players: [] })), [selected]);
  const teamMatches = useAsyncData(() => (selected ? api.listTeamMatches(selected) : Promise.resolve({ team: { id: 0, name: "" }, matches: [] })), [selected]);
  const [message, setMessage] = useState<string | null>(null);

  async function createTeam(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const team = await api.createTeam({
      name: String(form.get("name")),
      team_type: "opponent",
      notes: String(form.get("notes") || "")
    });
    setSelectedTeamId(team.id);
    event.currentTarget.reset();
    teams.refresh();
  }

  async function saveTeamProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected) return;
    const form = new FormData(event.currentTarget);
    form.set("team_type", "opponent");
    await api.saveTeamProfile(selected, form);
    setMessage("Opponent profile saved.");
    teams.refresh();
    teamMatches.refresh();
  }

  async function createPlayer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected) return;
    const form = new FormData(event.currentTarget);
    const jerseyNumber = String(form.get("jersey_number") || "").trim();
    await api.createTeamPlayer(selected, {
      name: String(form.get("name")),
      jersey_number: jerseyNumber ? Number(jerseyNumber) : null,
      primary_zone: String(form.get("primary_zone")),
      secondary_zones: form.getAll("secondary_zones").map(String).filter(Boolean),
      position_label: String(form.get("position_label") || ""),
      preferred_side: String(form.get("preferred_side") || "unknown"),
      notes: String(form.get("notes") || "")
    });
    setMessage("Player added.");
    event.currentTarget.reset();
    players.refresh();
  }

  async function deleteTeam() {
    if (!selected) return;
    await api.deleteTeam(selected);
    setMessage("Team deleted.");
    setSelectedTeamId(null);
    teams.refresh();
  }

  async function deletePlayer(playerId: number) {
    if (!selected) return;
    await api.deleteTeamPlayer(selected, playerId);
    setMessage("Player deleted.");
    players.refresh();
  }

  const selectedTeam = teams.data?.items.find((team) => team.id === selected) || players.data?.team;

  return (
    <section className="grid">
      <div className="section">
        <div className="card">
          <h2 className="section-title">Opponent Teams History</h2>
          <form className="toolbar" onSubmit={createTeam} style={{ marginTop: 14 }}>
            <input className="input" name="name" placeholder="Team name" required style={{ maxWidth: 360 }} />
            <input className="input" name="notes" placeholder="Notes" style={{ maxWidth: 360 }} />
            <button className="button primary" type="submit">Create opponent</button>
          </form>
          <div className="toolbar" style={{ marginTop: 14 }}>
            {(teams.data?.items || []).map((team) => (
              <button
                className={`button ${selected === team.id ? "primary" : ""}`}
                key={team.id}
                onClick={() => setSelectedTeamId(team.id)}
                type="button"
              >
                {team.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {selected ? (
        <section className="split">
          <div className="section">
            <div className="card">
              <h2 className="section-title">{selectedTeam?.name || "Opponent"} Profile</h2>
              <div className="toolbar" style={{ marginTop: 10 }}>
                <button className="button danger" onClick={deleteTeam} type="button">
                  Delete team
                </button>
              </div>
              <form className="form-grid" onSubmit={saveTeamProfile} style={{ marginTop: 14 }}>
                <label className="field full">
                  <span className="label">Name</span>
                  <input className="input" name="name" defaultValue={selectedTeam?.name || ""} />
                </label>
                <label className="field">
                  <span className="label">Primary kit</span>
                  <input className="input" name="primary_kit_image" type="file" accept="image/*" />
                </label>
                <label className="field">
                  <span className="label">Alternate kit</span>
                  <input className="input" name="alternate_kit_image" type="file" accept="image/*" />
                </label>
                <label className="field full">
                  <span className="label">Notes</span>
                  <textarea className="textarea" name="notes" defaultValue={selectedTeam?.notes || ""} />
                </label>
                <button className="button primary" type="submit">Save opponent</button>
              </form>
              <div className="grid" style={{ marginTop: 14 }}>
                <p className="muted small">Primary kit: {selectedTeam?.primary_kit_image_object_name || "Not uploaded"}</p>
                <p className="muted small">Alternate kit: {selectedTeam?.alternate_kit_image_object_name || "Not uploaded"}</p>
              </div>
              {message ? <p className="badge">{message}</p> : null}
            </div>
            <PlayerTable players={players.data?.players || []} onDelete={deletePlayer} />
            <OpponentMatchesTable matches={teamMatches.data?.matches || []} />
          </div>
          <div className="card">
            <h2 className="section-title">Add Opponent Player</h2>
            <form className="form-grid" onSubmit={createPlayer} style={{ marginTop: 14 }}>
              <label className="field full">
                <span className="label">Name</span>
                <input className="input" name="name" required />
              </label>
              <label className="field">
                <span className="label">Number</span>
                <input className="input" name="jersey_number" type="number" />
              </label>
              <label className="field">
                <span className="label">Primary zone</span>
                <select className="select" name="primary_zone" defaultValue="CM">
                  {zones.map((zone) => <option key={zone}>{zone}</option>)}
                </select>
              </label>
              <label className="field">
                <span className="label">Secondary zones</span>
                <select className="select" name="secondary_zones" multiple>
                  {zones.map((zone) => <option key={zone}>{zone}</option>)}
                </select>
              </label>
              <label className="field">
                <span className="label">Preferred side</span>
                <select className="select" name="preferred_side" defaultValue="unknown">
                  {preferredSides.map((side) => <option key={side}>{side}</option>)}
                </select>
              </label>
              <label className="field full">
                <span className="label">Position label</span>
                <select className="select" name="position_label" defaultValue="Central Midfielder">
                  {positionLabels.map((label) => <option key={label}>{label}</option>)}
                </select>
              </label>
              <label className="field full">
                <span className="label">Notes</span>
                <textarea className="textarea" name="notes" />
              </label>
              <button className="button primary" type="submit">Add player</button>
            </form>
          </div>
        </section>
      ) : (
        <Empty>Create an opponent team or upload a match against a new opponent.</Empty>
      )}
    </section>
  );
}

function OpponentMatchesTable({
  matches
}: {
  matches: Array<{
    id: number;
    title: string;
    status: string;
    match_type?: string;
    analysis_scope?: string;
    created_at?: string;
  }>;
}) {
  if (!matches.length) return <Empty>No matches linked to this opponent yet.</Empty>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Match</th>
            <th>Status</th>
            <th>Type</th>
            <th>Scope</th>
          </tr>
        </thead>
        <tbody>
          {matches.map((match) => (
            <tr key={match.id}>
              <td>{match.id}</td>
              <td>{match.title}</td>
              <td>{statusBadge(match.status)}</td>
              <td>{match.match_type || "-"}</td>
              <td>{match.analysis_scope || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PlayerTable({ players, onDelete }: { players: Player[]; onDelete: (id: number) => void }) {
  if (!players.length) return <Empty>No players yet.</Empty>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>#</th>
            <th>Name</th>
            <th>Zone</th>
            <th>Secondary</th>
            <th>Side</th>
            <th>Label</th>
            <th>Notes</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {players.map((player) => (
            <tr key={player.id}>
              <td>{player.jersey_number}</td>
              <td>{player.name}</td>
              <td>{player.primary_zone}</td>
              <td>{player.secondary_zones?.join(", ")}</td>
              <td>{player.preferred_side}</td>
              <td>{player.position_label}</td>
              <td>{player.notes || "-"}</td>
              <td>
                <button className="button danger" onClick={() => onDelete(player.id)} type="button">
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MatchesPage() {
  const opponents = useAsyncData(() => api.listTeams("opponent"), []);
  const matches = useAsyncData(() => api.listMatches(15), []);
  const [message, setMessage] = useState<string | null>(null);
  const [matchCategory, setMatchCategory] = useState("competitive");
  const [matchType, setMatchType] = useState("my_team_vs_opponent");
  const [firstTeamMode, setFirstTeamMode] = useState("saved");
  const [anotherTeamMode, setAnotherTeamMode] = useState("saved");
  const isMyTeamMatch = matchType === "my_team_vs_opponent";
  const isOpponentVsOpponent = matchType === "opponent_vs_opponent";
  const isInternal = matchCategory === "internal_scrimmage" || matchCategory === "academy_match";
  const anotherLabel = isInternal ? "Another side" : isOpponentVsOpponent ? "Second team" : "Other team";

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formElement = event.currentTarget;
    setMessage("Uploading...");
    const form = new FormData(formElement);
    form.set("match_category", matchCategory);
    form.set("match_type", isInternal ? matchCategory : matchType);
    if (isInternal || isMyTeamMatch) {
      form.delete("opponent_team_id");
      form.delete("opponent_team_name");
      form.delete("opponent_kit_image");
      form.delete("opponent_alternate_kit_image");
      form.delete("opponent_players_json");
    }
    const opponentTeamId = form.get("opponent_team_id");
    if (!opponentTeamId || firstTeamMode !== "saved") form.delete("opponent_team_id");
    if (firstTeamMode !== "new") {
      form.delete("opponent_team_name");
      form.delete("opponent_kit_image");
      form.delete("opponent_alternate_kit_image");
      form.delete("opponent_players_json");
    }
    const anotherTeamId = form.get("another_team_id");
    if (!anotherTeamId || anotherTeamMode !== "saved") form.delete("another_team_id");
    if (anotherTeamMode !== "new") {
      form.delete("another_team_name");
      form.delete("another_kit_image");
      form.delete("another_alternate_kit_image");
      form.delete("another_players_json");
    }
    if (!form.get("opponent_players_json")) form.delete("opponent_players_json");
    if (!form.get("another_players_json")) form.delete("another_players_json");
    const response = await api.uploadMatch(form);
    setMessage(`Match ${response.match_id} queued.`);
    formElement.reset();
    matches.refresh();
  }

  async function deleteMatch(id: number) {
    await api.deleteMatch(id);
    setMessage(`Match ${id} deleted.`);
    matches.refresh();
  }

  return (
    <section className="split">
      <div className="card">
        <h2 className="section-title">Upload Match</h2>
        <form className="form-grid" onSubmit={upload} style={{ marginTop: 14 }}>
          <label className="field full">
            <span className="label">Video file</span>
            <input className="input" type="file" name="file" accept="video/*" required />
          </label>
          <label className="field">
            <span className="label">Event type</span>
            <select className="select" name="match_category" value={matchCategory} onChange={(event) => {
              setMatchCategory(event.target.value);
              if (event.target.value === "internal_scrimmage" || event.target.value === "academy_match") {
                setMatchType(event.target.value);
              } else {
                setMatchType("my_team_vs_opponent");
              }
            }}>
              <option value="competitive">competitive match</option>
              <option value="friendly">friendly match</option>
              <option value="internal_scrimmage">internal scrimmage</option>
              <option value="academy_match">academy match</option>
            </select>
          </label>
          {!isInternal ? (
            <label className="field">
              <span className="label">Match type</span>
              <select className="select" name="match_type" value={matchType} onChange={(event) => setMatchType(event.target.value)}>
                <option value="my_team_vs_opponent">my team vs opponent</option>
                <option value="opponent_vs_opponent">opponent vs opponent</option>
              </select>
            </label>
          ) : null}
          <label className="field">
            <span className="label">Analysis scope</span>
            <select className="select" name="analysis_scope" defaultValue="both">
              <option value="my_team">my team</option>
              <option value="another_team">{isInternal ? "another side" : "another team"}</option>
              <option value="both">both sides</option>
              <option value="none">no player/team analysis</option>
            </select>
          </label>
          <label className="field">
            <span className="label">{isOpponentVsOpponent ? "First team formation" : "My team formation"}</span>
            <input className="input" name="formation" placeholder="4-3-3" />
          </label>
          {isOpponentVsOpponent ? (
            <>
              <label className="field">
                <span className="label">First team source</span>
                <select className="select" value={firstTeamMode} onChange={(event) => setFirstTeamMode(event.target.value)}>
                  <option value="saved">saved team</option>
                  <option value="new">new team</option>
                </select>
              </label>
              {firstTeamMode === "saved" ? (
                <label className="field">
                  <span className="label">First saved team</span>
                  <select className="select" name="opponent_team_id" defaultValue="">
                    <option value="">Select team</option>
                    {(opponents.data?.items || []).map((team) => <option value={team.id} key={team.id}>{team.name}</option>)}
                  </select>
                </label>
              ) : (
                <>
                  <label className="field">
                    <span className="label">First team name</span>
                    <input className="input" name="opponent_team_name" />
                  </label>
                  <label className="field">
                    <span className="label">First kit image</span>
                    <input className="input" name="opponent_kit_image" type="file" accept="image/*" />
                  </label>
                  <label className="field full">
                    <span className="label">First players JSON optional</span>
                    <textarea className="textarea" name="opponent_players_json" placeholder='[{"name":"Player name","jersey_number":10}]' />
                  </label>
                </>
              )}
            </>
          ) : null}
          <label className="field">
            <span className="label">{anotherLabel} formation</span>
            <input className="input" name="another_formation" placeholder="4-2-3-1" />
          </label>
          <label className="field">
            <span className="label">{anotherLabel} source</span>
            <select className="select" value={anotherTeamMode} onChange={(event) => setAnotherTeamMode(event.target.value)}>
              <option value="saved">saved team</option>
              <option value="new">new team</option>
            </select>
          </label>
          {anotherTeamMode === "saved" ? (
            <label className="field">
              <span className="label">Saved {anotherLabel.toLowerCase()}</span>
              <select className="select" name="another_team_id" defaultValue="">
                <option value="">Select team</option>
                {(opponents.data?.items || []).map((team) => <option value={team.id} key={team.id}>{team.name}</option>)}
              </select>
            </label>
          ) : (
            <label className="field">
              <span className="label">New {anotherLabel.toLowerCase()} name</span>
              <input className="input" name="another_team_name" />
            </label>
          )}
          <label className="field">
            <span className="label">Kit source my team</span>
            <select className="select" name="primary_team_kit_source" defaultValue="auto">
              <option value="auto">auto</option>
              <option value="primary">primary</option>
              <option value="alternate">alternate</option>
              <option value="unknown">unknown</option>
            </select>
          </label>
          <label className="field">
            <span className="label">Kit source {anotherLabel.toLowerCase()}</span>
            <select className="select" name="another_team_kit_source" defaultValue="auto">
              <option value="auto">auto</option>
              <option value="primary">primary</option>
              <option value="alternate">alternate</option>
              <option value="unknown">unknown</option>
            </select>
          </label>
          {anotherTeamMode === "new" ? (
            <>
              <label className="field">
                <span className="label">{anotherLabel} kit image</span>
                <input className="input" name="another_kit_image" type="file" accept="image/*" />
              </label>
              <label className="field">
                <span className="label">{anotherLabel} alternate kit</span>
                <input className="input" name="another_alternate_kit_image" type="file" accept="image/*" />
              </label>
              <label className="field full">
                <span className="label">{anotherLabel} players JSON optional</span>
                <textarea
                  className="textarea"
                  name="another_players_json"
                  placeholder='[{"name":"Player name","jersey_number":10,"primary_zone":"ST"}]'
                />
              </label>
            </>
          ) : null}
          {false ? (
            <select className="select" name="opponent_team_id" defaultValue="">
              {(opponents.data?.items || []).map((team) => <option value={team.id} key={team.id}>{team.name}</option>)}
            </select>
          ) : null}
          <div className="field full">
            <button className="button primary" type="submit">
              <Upload size={16} /> Upload and queue
            </button>
          </div>
        </form>
        {message ? <p className="badge">{message}</p> : null}
      </div>
      <div className="section">
        <h2 className="section-title">Matches</h2>
        <MatchesTable matches={matches.data?.items || []} onDelete={deleteMatch} />
      </div>
    </section>
  );
}

function AnalysisPage() {
  const matches = useAsyncData(() => api.listMatches(20), []);
  const [matchId, setMatchId] = useState<number | null>(null);
  const activeId = matchId || matches.data?.items?.[0]?.id || null;
  const processing = useAsyncData(() => (activeId ? api.getProcessing(activeId) : Promise.resolve(null)), [activeId]);
  const identity = useAsyncData(() => (activeId ? api.getIdentityAssignments(activeId) : Promise.resolve(null)), [activeId]);
  const tracks = useAsyncData(() => (activeId ? api.getTracks(activeId) : Promise.resolve(null)), [activeId]);
  const summary = processing.data?.job?.result_summary;

  return (
    <section className="grid">
      <div className="card">
        <div className="toolbar">
          <select className="select" value={activeId || ""} onChange={(event) => setMatchId(Number(event.target.value))} style={{ maxWidth: 420 }}>
            {(matches.data?.items || []).map((match) => <option key={match.id} value={match.id}>#{match.id} {match.title}</option>)}
          </select>
          <button className="button" onClick={() => { processing.refresh(); identity.refresh(); tracks.refresh(); }} type="button">
            <RefreshCw size={16} /> Refresh
          </button>
        </div>
      </div>
      <ProcessingCards summary={summary} />
      <TrackReviewPanel
        matchId={activeId}
        tracks={tracks.data?.tracks || []}
        onSaved={() => {
          tracks.refresh();
          identity.refresh();
        }}
      />
      <TacticalIdentityPanel assignments={identity.data?.assignments || []} />
    </section>
  );
}

function ProcessingCards({ summary }: { summary?: ProcessingSummary | null }) {
  const rawClassCounts = summary?.detections?.raw_class_counts || {};
  const personObservations = rawClassCounts.person ?? summary?.detections?.class_counts?.player ?? 0;
  const ballObservations = rawClassCounts["sports ball"] ?? summary?.detections?.class_counts?.ball ?? 0;
  const yoloReady = summary?.detections?.engine === "ultralytics_yolo";
  const trackingReady = summary?.tracking?.engine === "ultralytics_bytetrack";
  const artifactsReady = summary?.artifacts?.status === "ok";
  const cropsReady = Boolean(summary?.crops?.crops_count);
  const identityReady = Boolean(summary?.tactical_identity?.resolved_count);

  return (
    <>
      <section className="grid four">
        <StatCard title="Detector" value={summary?.detections?.engine || "-"} label={summary?.detections?.model || ""} />
        <StatCard title="Detection Observations" value={summary?.detections?.detections_count ?? "-"} label={`${summary?.detections?.frames_processed ?? "-"} sampled frames`} />
        <StatCard title="Tracklets" value={summary?.tracking?.tracks_count ?? "-"} label={summary?.tracking?.mode || summary?.tracking?.engine || "tracker pending"} />
        <StatCard title="Track Events" value={summary?.events?.events_count ?? "-"} label={`${summary?.events?.tracks_received ?? "-"} tracks received`} />
      </section>
      <section className="grid two">
        <div className="card">
          <h2 className="section-title">YOLO Observation Counts</h2>
          <div className="toolbar" style={{ marginTop: 14 }}>
            <span className="badge">person observations: {personObservations}</span>
            <span className="badge">ball observations: {ballObservations}</span>
          </div>
          <p className="muted small">
            This is not player count. It means YOLO saw people across sampled frames. Avg confidence: {summary?.detections?.confidence?.avg ?? "-"} · Elapsed: {summary?.detections?.elapsed_ms ?? "-"} ms
          </p>
        </div>
        <div className="card">
          <h2 className="section-title">Raw Model Classes</h2>
          <div className="toolbar" style={{ marginTop: 14 }}>
            {Object.keys(rawClassCounts).length
              ? Object.entries(rawClassCounts).map(([name, count]) => <span className="badge" key={name}>{name}: {count}</span>)
              : <span className="muted">No raw class counts yet</span>}
          </div>
          <p className="muted small">
            Device: {summary?.detections?.device || "-"} · Skipped frames: {summary?.detections?.frames_skipped ?? "-"} · Tracking input: {summary?.tracking?.detections_received ?? "-"} observations
          </p>
        </div>
      </section>
      <section className="grid two">
        <div className="card">
          <h2 className="section-title">YOLO Integration Status</h2>
          <div className="check-list">
            <span>{yoloReady ? "Done" : "Pending"}: real YOLO detections</span>
            <span>{trackingReady ? "Done" : "Partial"}: ByteTrack tracking</span>
            <span>{artifactsReady ? "Done" : "Pending"}: detections/tracks JSONL artifacts</span>
            <span>{cropsReady ? "Done" : "Pending"}: player and jersey crops</span>
            <span>{identityReady ? "Done" : "Pending"}: resolved player identities</span>
          </div>
        </div>
        <div className="card">
          <h2 className="section-title">Stored Artifacts</h2>
          {summary?.artifacts?.paths ? (
            <div className="artifact-list">
              {Object.entries(summary.artifacts.paths).map(([name, path]) => (
                <p className="muted small" key={name}>{name}: {path}</p>
              ))}
            </div>
          ) : (
            <p className="muted small">Artifacts will appear after the next processing run with the completed YOLO contract.</p>
          )}
          <p className="muted small">
            Crops: {summary?.crops?.crops_count ?? 0} player · {summary?.crops?.jersey_crops_count ?? 0} jersey
          </p>
        </div>
      </section>
    </>
  );
}

function TrackReviewPanel({
  matchId,
  tracks,
  onSaved
}: {
  matchId: number | null;
  tracks: TrackSummary[];
  onSaved: () => void;
}) {
  const [message, setMessage] = useState<string | null>(null);
  const visibleTracks = tracks.filter((track) => track.class_name === "player").slice(0, 80);

  async function assignTrack(event: FormEvent<HTMLFormElement>, track: TrackSummary) {
    event.preventDefault();
    if (!matchId) return;
    const form = new FormData(event.currentTarget);
    const shirtNumber = String(form.get("shirt_number") || "").trim();
    await api.saveTrackAssignment(matchId, {
      track_id: track.track_id,
      player_name: String(form.get("player_name") || ""),
      team_context: String(form.get("team_context") || track.team_context || "unknown"),
      shirt_number: shirtNumber ? Number(shirtNumber) : null,
      position: String(form.get("position") || "")
    });
    setMessage(`Track ${track.track_id} assigned.`);
    onSaved();
  }

  async function autoAssign() {
    if (!matchId) return;
    await api.autoAssignTracks(matchId);
    setMessage("Auto assignment finished.");
    onSaved();
  }

  if (!matchId) return null;

  return (
    <div className="card">
      <div className="section-header">
        <div>
          <h2 className="section-title">Track Review</h2>
          <p className="muted small">Review random crop samples and confirm only OCR/roster matches you trust.</p>
        </div>
        <button className="button" onClick={autoAssign} type="button">
          <RefreshCw size={16} /> Auto assign by number
        </button>
      </div>
      {message ? <p className="badge">{message}</p> : null}
      {!visibleTracks.length ? (
        <Empty>No player tracks available. Reprocess a match after YOLO/ByteTrack completes.</Empty>
      ) : (
        <div className="track-grid">
          {visibleTracks.map((track) => {
            const sample = track.crop_samples?.find((item) => item.crop_path) || track.crop_samples?.[0];
            const imagePath = sample?.crop_path || sample?.jersey_crop_path;
            return (
              <div className="track-card" key={track.track_id}>
                <div className="track-media">
                  {imagePath ? (
                    <img src={api.objectUrl(imagePath)} alt={`Track ${track.track_id}`} />
                  ) : (
                    <div className="empty">No crop</div>
                  )}
                </div>
                <div className="track-meta">
                  <div className="toolbar">
                    <span className="badge">Track {track.track_id}</span>
                    <span className="badge">{track.team_context || "unknown"}</span>
                    <span className="badge">OCR #{track.recognized_shirt_number ?? "not verified"}</span>
                  </div>
                  <p className="muted small">
                    {track.frames_count ?? 0} frames · team confidence {track.team_assignment_confidence ?? "-"} · OCR {track.shirt_number_confidence ?? "-"} · {track.shirt_number_source || "no_ocr"}
                  </p>
                  <form className="form-grid compact" onSubmit={(event) => assignTrack(event, track)}>
                    <label className="field full">
                      <span className="label">Player name</span>
                      <input className="input" name="player_name" defaultValue={track.assignment?.player_name || ""} required />
                    </label>
                    <label className="field">
                      <span className="label">Team</span>
                      <select className="select" name="team_context" defaultValue={track.assignment?.team_context || track.team_context || "unknown"}>
                        <option value="primary_team">primary_team</option>
                        <option value="opponent_team">opponent_team</option>
                        <option value="club_internal">club_internal</option>
                      </select>
                    </label>
                    <label className="field">
                      <span className="label">Number</span>
                      <input className="input" name="shirt_number" type="number" defaultValue={track.assignment?.shirt_number ?? track.recognized_shirt_number ?? ""} />
                    </label>
                    <label className="field full">
                      <span className="label">Position</span>
                      <input className="input" name="position" defaultValue={track.assignment?.position || ""} />
                    </label>
                    <button className="button primary" type="submit">Save identity</button>
                  </form>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function TacticalIdentityPanel({ assignments }: { assignments: IdentityAssignment[] }) {
  const visibleAssignments = assignments.slice(0, 80);

  return (
    <div className="card">
      <h2 className="section-title">Tactical Identity Tracklets</h2>
      {!assignments.length ? (
        <Empty>No identity assignments yet. Add lineup before processing or reprocess this match later.</Empty>
      ) : (
        <div className="table-wrap" style={{ marginTop: 14 }}>
          <p className="muted small">
            Showing {visibleAssignments.length} of {assignments.length} tracklet assignments. These are not final player identities until real tracking and player re-identification are connected.
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Track</th>
                <th>Player</th>
                <th>Zone</th>
                <th>Confidence</th>
                <th>Top reasons</th>
              </tr>
            </thead>
            <tbody>
              {visibleAssignments.map((item) => (
                <tr key={item.track_id}>
                  <td>{item.track_id}</td>
                  <td>{item.resolved_player?.name || item.resolved_player_id || "-"}</td>
                  <td>{item.zone || item.resolved_player?.zone || "-"}</td>
                  <td>{item.confidence ?? "-"}</td>
                  <td>
                    {item.candidates?.[0]?.reasons
                      ? Object.entries(item.candidates[0].reasons)
                          .map(([key, value]) => `${key}: ${value}`)
                          .join(", ")
                      : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function FirstAnalysisPage() {
  const matches = useAsyncData(() => api.listMatches(20), []);
  const [matchId, setMatchId] = useState<number | null>(null);
  const [maxFrames, setMaxFrames] = useState(450);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [result, setResult] = useState<FirstAnalysisSummary | null>(null);
  const activeId = matchId || matches.data?.items?.[0]?.id || null;
  const saved = useAsyncData(
    () => (activeId ? api.getFirstAnalysis(activeId) : Promise.resolve({ exists: false, summary: null })),
    [activeId]
  );
  const summary = result || saved.data?.summary || null;

  async function runAnalysis() {
    if (!activeId) return;
    setRunning(true);
    setMessage("Running first analysis...");
    try {
      const response = await api.runFirstAnalysis(activeId, maxFrames);
      setResult(response);
      setVideoError(null);
      setMessage("First analysis video generated.");
      saved.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "First analysis failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="grid">
      <div className="card">
        <div className="toolbar">
          <select className="select" value={activeId || ""} onChange={(event) => setMatchId(Number(event.target.value))} style={{ maxWidth: 520 }}>
            {(matches.data?.items || []).map((match) => <option key={match.id} value={match.id}>#{match.id} {match.title}</option>)}
          </select>
          <label className="toolbar">
            <span className="label">Max frames</span>
            <input
              className="input"
              min="0"
              max="200000"
              onChange={(event) => setMaxFrames(Number(event.target.value))}
              style={{ maxWidth: 120 }}
              type="number"
              value={maxFrames}
            />
          </label>
          <button className="button primary" disabled={!activeId || running} onClick={runAnalysis} type="button">
            <BarChart3 size={16} /> {running ? "Running..." : "Run first analysis"}
          </button>
          <button className="button" onClick={() => { saved.refresh(); matches.refresh(); }} type="button">
            <RefreshCw size={16} /> Refresh
          </button>
        </div>
        {message ? <p className="badge">{message}</p> : null}
        <p className="muted small">
          Uses the copied football_analysis-main idea to draw tracks, camera movement, ball control, speed, and distance. Set Max frames to 0 for the full video; 450 frames is about 18 seconds at 25fps.
        </p>
      </div>

      {summary ? (
        <>
          <section className="grid four">
            <StatCard title="Frames" value={summary.frames_processed} label={`max ${summary.max_frames}`} />
            <StatCard title="Tracks" value={summary.player_tracks_count} label="player track ids" />
            <StatCard title="Detections" value={summary.detections_count} label={`${summary.ball_observations} ball observations`} />
            <StatCard title="Elapsed" value={`${summary.elapsed_ms} ms`} label={summary.engine} />
          </section>
          <section className="grid two">
            <div className="card">
              <h2 className="section-title">Output Video</h2>
              <video
                className="analysis-video"
                controls
                onError={() => setVideoError("Browser could not play this generated video. Run first analysis again to regenerate it as WebM.")}
                src={api.objectUrl(summary.output_object)}
              />
              {videoError ? <p className="badge error">{videoError}</p> : null}
              <div className="toolbar" style={{ marginTop: 14 }}>
                <a className="button" href={api.objectUrl(summary.output_object)} target="_blank" rel="noreferrer">
                  Open video
                </a>
              </div>
            </div>
            <div className="card">
              <h2 className="section-title">First Analysis Summary</h2>
              <div className="table-wrap" style={{ marginTop: 14 }}>
                <table className="table">
                  <tbody>
                    <tr><th>Model</th><td>{summary.model}</td></tr>
                    <tr><th>Output codec</th><td>{summary.output_codec || "-"}</td></tr>
                    <tr><th>Resolution</th><td>{summary.resolution.join(" x ")}</td></tr>
                    <tr><th>FPS</th><td>{summary.fps}</td></tr>
                    <tr><th>Team 1 ball control</th><td>{summary.team_ball_control.team_1_percent}%</td></tr>
                    <tr><th>Team 2 ball control</th><td>{summary.team_ball_control.team_2_percent}%</td></tr>
                    <tr><th>Original project</th><td>{summary.original_project_path}</td></tr>
                    <tr><th>Artifact</th><td>{summary.output_object}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </>
      ) : (
        <Empty>Select an uploaded match and run first analysis to generate an annotated video.</Empty>
      )}
    </section>
  );
}

function MatchAnalysisPlusPage() {
  const matches = useAsyncData(() => api.listMatches(30), []);
  const [matchId, setMatchId] = useState<number | null>(null);
  const [maxFrames, setMaxFrames] = useState(450);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const activeId = matchId || matches.data?.items?.[0]?.id || null;
  const runs = useAsyncData(
    () => (activeId ? api.getMatchAnalysisPlus(activeId) : Promise.resolve({ match_id: 0, match_title: "", runs: [], latest: null })),
    [activeId]
  );
  const selectedRun =
    runs.data?.runs.find((item) => item.id === selectedRunId) ||
    runs.data?.latest ||
    null;
  const summary = selectedRun?.summary || null;

  async function runAnalysis() {
    if (!activeId) return;
    setRunning(true);
    setMessage("Running Match Analysis +...");
    try {
      const response = await api.runMatchAnalysisPlus(activeId, {
        mode: "FULL_ANALYSIS",
        max_frames: maxFrames
      });
      setSelectedRunId(response.id);
      setVideoError(null);
      setMessage(`Run #${response.id} saved.`);
      runs.refresh();
      matches.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Match Analysis + failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="grid">
      <div className="card">
        <div className="toolbar">
          <select className="select" value={activeId || ""} onChange={(event) => { setMatchId(Number(event.target.value)); setSelectedRunId(null); }} style={{ maxWidth: 520 }}>
            {(matches.data?.items || []).map((match) => <option key={match.id} value={match.id}>#{match.id} {match.title}</option>)}
          </select>
          <label className="toolbar">
            <span className="label">Max frames</span>
            <input
              className="input"
              min="0"
              max="200000"
              onChange={(event) => setMaxFrames(Number(event.target.value))}
              style={{ maxWidth: 120 }}
              type="number"
              value={maxFrames}
            />
          </label>
          <button className="button primary" disabled={!activeId || running} onClick={runAnalysis} type="button">
            <BarChart3 size={16} /> {running ? "Running..." : "Run analysis"}
          </button>
          <button className="button" onClick={() => { runs.refresh(); matches.refresh(); }} type="button">
            <RefreshCw size={16} /> Refresh
          </button>
        </div>
        {message ? <p className="badge">{message}</p> : null}
        <p className="muted small">Full analysis profile. Results are saved per match.</p>
      </div>

      <section className="grid two">
        <div className="card">
          <h2 className="section-title">Saved Runs</h2>
          {!runs.data?.runs.length ? (
            <Empty>No saved Match Analysis + runs for this match yet.</Empty>
          ) : (
            <div className="table-wrap" style={{ marginTop: 14 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Profile</th>
                    <th>Status</th>
                    <th>Frames</th>
                    <th>Created</th>
                    <th>Open</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.data.runs.map((run) => (
                    <tr key={run.id}>
                      <td>#{run.id}</td>
                      <td>{run.mode === "FULL_ANALYSIS" ? "Full analysis" : "Legacy run"}</td>
                      <td>{statusBadge(run.status)}</td>
                      <td>{run.summary?.frames_processed ?? run.max_frames}</td>
                      <td>{run.created_at ? new Date(run.created_at).toLocaleString() : "-"}</td>
                      <td>
                        <button className="button" onClick={() => { setSelectedRunId(run.id); setVideoError(null); }} type="button">
                          Open
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="card">
          <h2 className="section-title">Run Details</h2>
          {selectedRun ? (
            <div className="table-wrap" style={{ marginTop: 14 }}>
              <table className="table">
                <tbody>
                  <tr><th>Run</th><td>#{selectedRun.id}</td></tr>
                  <tr><th>Status</th><td>{selectedRun.status}</td></tr>
                  <tr><th>Profile</th><td>{selectedRun.mode === "FULL_ANALYSIS" ? "Full analysis" : "Legacy run"}</td></tr>
                  <tr><th>Source</th><td>{selectedRun.source}</td></tr>
                  <tr><th>Worker</th><td>{summary?.worker || "-"}</td></tr>
                  <tr><th>Project</th><td>{summary?.source_project || "apps/match-analysis-worker/sports-main"}</td></tr>
                  <tr><th>Error</th><td>{selectedRun.error_message || "-"}</td></tr>
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>Select or create a run to see details.</Empty>
          )}
        </div>
      </section>

      {summary && selectedRun?.output_object ? (
        <>
          <section className="grid four">
            <StatCard title="Frames" value={summary.frames_processed} label={`max ${summary.max_frames}`} />
            <StatCard title="Tracks" value={summary.tracks_count} label={summary.model_mode || "sports-main worker"} />
            <StatCard title="Detections" value={summary.detections_count} label={Object.entries(summary.class_counts || {}).map(([key, value]) => `${key}: ${value}`).join(" / ") || "class counts"} />
            <StatCard title="Elapsed" value={`${summary.elapsed_ms} ms`} label={summary.output_codec || "codec"} />
          </section>

          <section className="grid two">
            <div className="card">
              <h2 className="section-title">Output Video</h2>
              <video
                className="analysis-video"
                controls
                onError={() => setVideoError("Browser could not play this generated video. Re-run the analysis or check the worker logs.")}
                src={api.objectUrl(selectedRun.output_object)}
              />
              {videoError ? <p className="badge error">{videoError}</p> : null}
              <div className="toolbar" style={{ marginTop: 14 }}>
                <a className="button" href={api.objectUrl(selectedRun.output_object)} target="_blank" rel="noreferrer">
                  Open video
                </a>
                {selectedRun.summary_object ? (
                  <a className="button" href={api.objectUrl(selectedRun.summary_object)} target="_blank" rel="noreferrer">
                    Open JSON
                  </a>
                ) : null}
              </div>
            </div>

            <div className="card">
              <h2 className="section-title">Visible Data</h2>
              <div className="table-wrap" style={{ marginTop: 14 }}>
                <table className="table">
                  <tbody>
                    <tr><th>Model</th><td>{summary.model}</td></tr>
                    <tr><th>Resolution</th><td>{summary.resolution.join(" x ")}</td></tr>
                    <tr><th>FPS</th><td>{summary.fps}</td></tr>
                    <tr><th>Processing FPS</th><td>{summary.processing_fps ?? "-"}</td></tr>
                    <tr><th>Confidence avg</th><td>{summary.confidence?.avg ?? "-"}</td></tr>
                    <tr><th>Fixture detections rejected</th><td>{summary.player_filter?.rejected_field_fixtures ?? 0}</td></tr>
                    <tr><th>Static ball candidates rejected</th><td>{summary.ball_filter?.filtered_static_candidates ?? 0}</td></tr>
                    <tr><th>Team kit references</th><td>{Object.keys(summary.team_classifier?.kit_anchors_bgr || {}).length}</td></tr>
                    <tr><th>Pitch calibration</th><td>{summary.radar?.calibration_mode ?? "Not calibrated"}</td></tr>
                    <tr><th>Radar calibrations</th><td>{summary.radar?.successful_calibrations ?? 0}</td></tr>
                    <tr>
                      <th>Line alignment</th>
                      <td>
                        {summary.radar?.last_line_alignment_score != null
                          ? `${(summary.radar.last_line_alignment_score * 100).toFixed(1)}%`
                          : "-"}
                      </td>
                    </tr>
                    <tr>
                      <th>Camera tracking</th>
                      <td>
                        {summary.radar?.camera_tracking
                          ? `${summary.radar.camera_tracking.successes}/${summary.radar.camera_tracking.attempts}`
                          : "-"}
                      </td>
                    </tr>
                    <tr>
                      <th>Metric pitch</th>
                      <td>
                        {summary.radar?.pitch_template
                          ? `${summary.radar.pitch_template.length_cm / 100} x ${summary.radar.pitch_template.width_cm / 100} m`
                          : "-"}
                      </td>
                    </tr>
                    <tr><th>Heatmap coordinates</th><td>{summary.metric_tracking?.heatmap_ready ? "Ready" : "Not calibrated"}</td></tr>
                    <tr><th>Radar frames rendered</th><td>{summary.radar?.rendered_frames ?? 0}</td></tr>
                    <tr><th>Team 1 control</th><td>{summary.team_ball_control?.team_1_percent ?? 0}%</td></tr>
                    <tr><th>Team 2 control</th><td>{summary.team_ball_control?.team_2_percent ?? 0}%</td></tr>
                    <tr><th>Output</th><td>{summary.output_object}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          <div className="card">
            <h2 className="section-title">Tracks Data</h2>
            <MiniDataTable
              rows={(summary.tracks || []) as Array<Record<string, unknown>>}
              columns={["track_id", "team", "frames", "distance_m", "last_speed_kmh"]}
            />
          </div>
        </>
      ) : null}
    </section>
  );
}

function ReportsPage() {
  const matches = useAsyncData(() => api.listMatches(20), []);
  const [matchId, setMatchId] = useState<number | null>(null);
  const activeId = matchId || matches.data?.items?.[0]?.id || null;
  const report = useAsyncData<ReportResponse | null>(() => (activeId ? api.getReport(activeId) : Promise.resolve(null)), [activeId]);
  const reportData = report.data?.report?.data;
  const counts = reportData?.counts;
  const summary = reportData?.summary as Record<string, unknown> | undefined;
  const charts = reportData?.charts as
    | {
        team_distance?: Array<Record<string, unknown>>;
        player_speed?: Array<Record<string, unknown>>;
      }
    | undefined;
  const teams = reportData?.teams as
    | {
        primary_team?: Array<Record<string, unknown>>;
        opponent_team?: Array<Record<string, unknown>>;
      }
    | undefined;
  const manualAssignments = reportData?.identity?.manual_assignments || [];

  return (
    <section className="grid">
      <div className="card">
        <div className="toolbar">
          <select className="select" value={activeId || ""} onChange={(event) => setMatchId(Number(event.target.value))} style={{ maxWidth: 420 }}>
            {(matches.data?.items || []).map((match) => <option key={match.id} value={match.id}>#{match.id} {match.title}</option>)}
          </select>
          <button className="button" onClick={report.refresh} type="button">
            <RefreshCw size={16} /> Refresh
          </button>
          {activeId ? (
            <a className="button" href={api.reportPdfUrl(activeId)}>
              <FileText size={16} /> PDF
            </a>
          ) : null}
        </div>
      </div>
      <div className="grid four">
        <StatCard title="Detections" value={counts?.detections ?? "-"} />
        <StatCard title="Tracks" value={counts?.tracks ?? "-"} />
        <StatCard title="Events" value={counts?.events ?? "-"} />
        <StatCard title="Identity" value={manualAssignments.length || counts?.identity_resolved || counts?.identity_assignments || "-"} />
      </div>
      <section className="grid two">
        <div className="card">
          <h2 className="section-title">Match Summary</h2>
          <div className="table-wrap" style={{ marginTop: 14 }}>
            <table className="table">
              <tbody>
                <tr><th>Primary team</th><td>{String(summary?.primary_team_name || "-")}</td></tr>
                <tr><th>Opponent</th><td>{String(summary?.opponent_team_name || "-")}</td></tr>
                <tr><th>Type</th><td>{String(summary?.match_type || "-")}</td></tr>
                <tr><th>Scope</th><td>{String(summary?.analysis_scope || "-")}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <h2 className="section-title">Team Distance</h2>
          <MiniDataTable rows={charts?.team_distance || []} columns={["team_context", "players_count", "total_distance"]} />
        </div>
      </section>
      <section className="grid two">
        <div className="card">
          <h2 className="section-title">My Team Players</h2>
          <MiniDataTable rows={teams?.primary_team || []} columns={["track_id", "recognized_shirt_number", "distance", "average_speed", "max_speed"]} />
        </div>
        <div className="card">
          <h2 className="section-title">Opponent Players</h2>
          <MiniDataTable rows={teams?.opponent_team || []} columns={["track_id", "recognized_shirt_number", "distance", "average_speed", "max_speed"]} />
        </div>
      </section>
      <div className="card">
        <h2 className="section-title">Manual Identities</h2>
        <MiniDataTable
          rows={manualAssignments.map((item) => ({
            track_id: item.track_id,
            player: item.resolved_player?.name,
            number: item.resolved_player?.jersey_number,
            team_context: item.team_context,
            confidence: item.confidence
          }))}
          columns={["track_id", "player", "number", "team_context", "confidence"]}
        />
      </div>
    </section>
  );
}

function MiniDataTable({
  rows,
  columns
}: {
  rows: Array<Record<string, unknown>>;
  columns: string[];
}) {
  if (!rows.length) return <Empty>No data yet.</Empty>;
  return (
    <div className="table-wrap" style={{ marginTop: 14 }}>
      <table className="table">
        <thead>
          <tr>
            {columns.map((column) => <th key={column}>{column}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 80).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{String(row[column] ?? "-")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AgentPage() {
  const [messages, setMessages] = useState([
    { role: "agent", text: "Select a match in Analysis or Reports, then ask for a tactical explanation, player plan, or training idea." }
  ]);
  const [text, setText] = useState("");

  function send(event: FormEvent) {
    event.preventDefault();
    if (!text.trim()) return;
    const prompt = text.trim();
    setMessages((items) => [
      ...items,
      { role: "user", text: prompt },
      {
        role: "agent",
        text: "Agent API is prepared as a UI surface. Next backend step: connect this panel to LLM Coach with match_id context and report retrieval."
      }
    ]);
    setText("");
  }

  return (
    <section className="split">
      <div className="card chat-window">
        <h2 className="section-title">Coach Agent</h2>
        <div className="grid" style={{ flex: 1 }}>
          {messages.map((message, index) => (
            <div className={`message ${message.role === "user" ? "user" : ""}`} key={`${message.role}-${index}`}>
              {message.text}
            </div>
          ))}
        </div>
        <form className="toolbar" onSubmit={send}>
          <input className="input" value={text} onChange={(event) => setText(event.target.value)} placeholder="Ask about the last match..." />
          <button className="button primary" type="submit">Send</button>
        </form>
      </div>
      <div className="card">
        <h2 className="section-title">Agent Context Slots</h2>
        <ul className="muted">
          <li>Match report</li>
          <li>Player identity assignments</li>
          <li>Season aggregates</li>
          <li>Training plan generation</li>
          <li>Manual coach corrections</li>
        </ul>
      </div>
    </section>
  );
}

function RecommendationsPage() {
  const matches = useAsyncData(() => api.listMatches(5), []);
  const latest = matches.data?.items?.[0];
  return (
    <section className="grid three">
      <RecommendationCard
        title="Based on Last Match"
        items={[
          latest ? `Review ${latest.title} tactical identity gaps before reprocessing.` : "Upload a match to generate game-specific recommendations.",
          "Compare YOLO detections with final tracks once ByteTrack v1 lands.",
          "Add starting lineup before processing to unlock player-level confidence."
        ]}
      />
      <RecommendationCard
        title="Based on Season"
        items={[
          "Season model is reserved for aggregate match stats.",
          "Track recurring weak zones by formation.",
          "Prepare endpoints for player load and development trends."
        ]}
      />
      <RecommendationCard
        title="Players"
        items={[
          "Missing jersey OCR confidence should trigger manual review.",
          "Players with repeated low zone-match confidence need lineup correction.",
          "Training plans will use report identity assignments."
        ]}
      />
    </section>
  );
}

function RecommendationCard({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="card">
      <h2 className="section-title">{title}</h2>
      <div className="grid" style={{ marginTop: 14 }}>
        {items.map((item) => <div className="empty" key={item}>{item}</div>)}
      </div>
    </div>
  );
}

function RecommendationsPreview({ match }: { match?: MatchSummary }) {
  return (
    <div className="card">
      <h2 className="section-title">Smart Suggestions</h2>
      <div className="grid" style={{ marginTop: 14 }}>
        <div className="empty">
          {match ? `Latest match #${match.id}: check report and identity before tactical review.` : "Upload a match to unlock recommendations."}
        </div>
        <div className="empty">Add lineup before processing to improve player identity confidence.</div>
      </div>
    </div>
  );
}

export function App() {
  const [route, setRoute] = useState<RouteKey>("dashboard");
  const page = useMemo(() => {
    switch (route) {
      case "dashboard":
        return <DashboardPage setRoute={setRoute} />;
      case "settings":
        return <SettingsPage />;
      case "my-team":
        return <MyTeamPage />;
      case "teams":
        return <TeamsPage />;
      case "matches":
        return <MatchesPage />;
      case "analysis":
        return <AnalysisPage />;
      case "first-analysis":
        return <FirstAnalysisPage />;
      case "match-analysis-plus":
        return <MatchAnalysisPlusPage />;
      case "reports":
        return <ReportsPage />;
      case "agent":
        return <AgentPage />;
      case "recommendations":
        return <RecommendationsPage />;
      default:
        return <DashboardPage setRoute={setRoute} />;
    }
  }, [route]);

  return (
    <AppShell route={route} setRoute={setRoute}>
      {page}
    </AppShell>
  );
}
