import {
  BarChart3,
  Bot,
  CalendarDays,
  Clock3,
  FileText,
  Gauge,
  Flame,
  Layers3,
  Lightbulb,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Settings,
  Shield,
  Upload,
  Users,
  Video,
  Volume2,
  VolumeX,
  Waypoints,
  X
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type {
  MatchVisualLayers,
  MatchVisualLayerTrack,
  MatchSummary,
  Player,
  PrimaryTeamProfile,
  MatchAnalysisPlusRun,
  ReportResponse,
  RouteKey,
  RosterPlayer,
  Team
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
  const summary = latest?.latest_match_analysis_run?.summary;

  return (
    <>
      <section className="grid four">
        <StatCard title="Matches" value={matches.data?.items?.length ?? "-"} label="latest loaded" />
        <StatCard title="Latest status" value={latest ? statusBadge(latest.status) : "-"} label={latest?.title} />
        <StatCard title="YOLO detections" value={summary?.detections_count ?? "-"} label={summary?.model_mode || "waiting"} />
        <StatCard title="Stable tracks" value={summary?.tracks_count ?? "-"} label="Match Analysis +" />
      </section>

      <section className="split">
        <div className="section">
          <div className="section-header">
            <h2 className="section-title">Recent Matches</h2>
            <button className="button" onClick={matches.refresh} type="button">
              <RefreshCw size={16} /> Refresh
            </button>
          </div>
          <MatchesTable matches={matches.data?.items || []} onOpen={() => setRoute("match-analysis-plus")} />
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
                  : "-"}
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

type VisualWindowMode = "progress" | "range";

function formatAnalysisTime(seconds: number) {
  const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safeSeconds / 60);
  const remaining = Math.floor(safeSeconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function TrackLayerPicker({
  icon,
  label,
  tracks,
  selected,
  onChange,
  open,
  onOpenChange
}: {
  icon: ReactNode;
  label: string;
  tracks: MatchVisualLayerTrack[];
  selected: number[];
  onChange: (trackIds: number[]) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const selectedSet = new Set(selected);
  const teamGroups = Array.from(new Set(tracks.map((track) => track.team).filter((team) => team != null)))
    .sort((left, right) => Number(left) - Number(right));

  function toggle(trackId: number) {
    onChange(
      selectedSet.has(trackId)
        ? selected.filter((item) => item !== trackId)
        : [...selected, trackId].sort((left, right) => left - right)
    );
  }

  function selectGroup(trackIds: number[]) {
    onChange([...trackIds].sort((left, right) => left - right));
  }

  return (
    <div className={`layer-picker${open ? " open" : ""}`}>
      <button
        aria-expanded={open}
        className="layer-picker-trigger"
        onClick={() => onOpenChange(!open)}
        type="button"
      >
        {icon}
        <span>{label}</span>
        <span className="layer-count">{selected.length}</span>
      </button>
      {open ? (
        <div className="layer-menu">
          <div className="layer-menu-header">
            <strong>{label}</strong>
            <button
              aria-label={`Close ${label}`}
              className="icon-button"
              onClick={() => onOpenChange(false)}
              title={`Close ${label}`}
              type="button"
            >
              <X size={16} />
            </button>
          </div>
          <div className="layer-quick-actions" aria-label={`${label} quick selection`}>
            <button onClick={() => selectGroup(tracks.map((track) => track.track_id))} type="button">
              All players
            </button>
            {teamGroups.map((team) => (
              <button
                key={team}
                onClick={() => selectGroup(tracks.filter((track) => track.team === team).map((track) => track.track_id))}
                type="button"
              >
                Team {team}
              </button>
            ))}
            <button disabled={!selected.length} onClick={() => onChange([])} type="button">
              Clear
            </button>
          </div>
          <div className="layer-options">
            {tracks.map((track) => (
              <label className="layer-option" key={track.track_id}>
                <input
                  checked={selectedSet.has(track.track_id)}
                  onChange={() => toggle(track.track_id)}
                  type="checkbox"
                />
                <span className="track-swatch" style={{ backgroundColor: track.color }} />
                <span>Track {track.track_id}</span>
                <span className="muted">Team {track.team ?? "-"}</span>
              </label>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function TimeWindowPicker({
  duration,
  mode,
  onModeChange,
  rangeStart,
  rangeEnd,
  onRangeChange,
  onPlayRange,
  open,
  onOpenChange
}: {
  duration: number;
  mode: VisualWindowMode;
  onModeChange: (mode: VisualWindowMode) => void;
  rangeStart: number;
  rangeEnd: number;
  onRangeChange: (start: number, end: number) => void;
  onPlayRange: () => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const safeDuration = Math.max(0.1, duration);
  const minimumGap = Math.min(0.25, safeDuration);

  function updateStart(value: number) {
    onRangeChange(Math.max(0, Math.min(value, rangeEnd - minimumGap)), rangeEnd);
  }

  function updateEnd(value: number) {
    onRangeChange(rangeStart, Math.min(safeDuration, Math.max(value, rangeStart + minimumGap)));
  }

  return (
    <div className={`layer-picker time-window-picker${open ? " open" : ""}`}>
      <button
        aria-expanded={open}
        className="layer-picker-trigger"
        onClick={() => onOpenChange(!open)}
        type="button"
      >
        <Clock3 size={16} />
        <span>{mode === "progress" ? "Live progress" : "Time range"}</span>
        <span className="time-window-summary">
          {mode === "progress" ? "LIVE" : `${formatAnalysisTime(rangeStart)}-${formatAnalysisTime(rangeEnd)}`}
        </span>
      </button>
      {open ? (
        <div className="layer-menu time-window-menu">
          <div className="layer-menu-header">
            <strong>Analysis time window</strong>
            <button
              aria-label="Close time window"
              className="icon-button"
              onClick={() => onOpenChange(false)}
              title="Close time window"
              type="button"
            >
              <X size={16} />
            </button>
          </div>
          <div className="time-window-content">
            <div className="segmented-control" role="group" aria-label="Visualization time mode">
              <button
                className={mode === "progress" ? "active" : ""}
                onClick={() => onModeChange("progress")}
                type="button"
              >
                Live progress
              </button>
              <button
                className={mode === "range" ? "active" : ""}
                onClick={() => onModeChange("range")}
                type="button"
              >
                Fixed range
              </button>
            </div>
            {mode === "range" ? (
              <>
                <label className="time-range-field">
                  <span>From <strong>{formatAnalysisTime(rangeStart)}</strong></span>
                  <input
                    max={safeDuration}
                    min={0}
                    onChange={(event) => updateStart(Number(event.target.value))}
                    step="0.1"
                    type="range"
                    value={rangeStart}
                  />
                  <input
                    className="input"
                    max={safeDuration}
                    min={0}
                    onChange={(event) => updateStart(Number(event.target.value))}
                    step="0.1"
                    type="number"
                    value={Number(rangeStart.toFixed(1))}
                  />
                </label>
                <label className="time-range-field">
                  <span>To <strong>{formatAnalysisTime(rangeEnd)}</strong></span>
                  <input
                    max={safeDuration}
                    min={0}
                    onChange={(event) => updateEnd(Number(event.target.value))}
                    step="0.1"
                    type="range"
                    value={rangeEnd}
                  />
                  <input
                    className="input"
                    max={safeDuration}
                    min={0}
                    onChange={(event) => updateEnd(Number(event.target.value))}
                    step="0.1"
                    type="number"
                    value={Number(rangeEnd.toFixed(1))}
                  />
                </label>
                <div className="time-window-actions">
                  <button className="button primary" onClick={onPlayRange} type="button">
                    <Play size={15} /> Play range
                  </button>
                  <button
                    aria-label="Reset time range"
                    className="icon-button"
                    onClick={() => onRangeChange(0, safeDuration)}
                    title="Reset time range"
                    type="button"
                  >
                    <RotateCcw size={16} />
                  </button>
                </div>
              </>
            ) : (
              <p className="time-window-note">Movement paths and heatmaps accumulate with video playback.</p>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function nearestPitchProjection(samples: number[][], frame: number): number[] | null {
  if (!samples.length) return null;
  let low = 0;
  let high = samples.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (samples[middle][0] <= frame) low = middle + 1;
    else high = middle - 1;
  }
  const before = samples[Math.max(0, high)];
  const after = samples[Math.min(samples.length - 1, low)];
  const nearest = Math.abs(before[0] - frame) <= Math.abs(after[0] - frame) ? before : after;
  return nearest.length === 10 ? nearest.slice(1) : null;
}

function projectPitchPoint(matrix: number[], x: number, y: number): [number, number] | null {
  if (matrix.length !== 9) return null;
  const denominator = matrix[6] * x + matrix[7] * y + matrix[8];
  if (Math.abs(denominator) < 1e-9) return null;
  const projectedX = (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator;
  const projectedY = (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator;
  return Number.isFinite(projectedX) && Number.isFinite(projectedY)
    ? [projectedX, projectedY]
    : null;
}

function drawMovementOverlay(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  layers: MatchVisualLayers,
  selectedTrackIds: number[],
  startFrame: number,
  endFrame: number
) {
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;
  if (!cssWidth || !cssHeight) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const targetWidth = Math.round(cssWidth * pixelRatio);
  const targetHeight = Math.round(cssHeight * pixelRatio);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);
  if (!selectedTrackIds.length) return;

  const sourceWidth = layers.resolution[0] || video.videoWidth || 1;
  const sourceHeight = layers.resolution[1] || video.videoHeight || 1;
  const currentFrame = Math.min(
    layers.frames_processed - 1,
    Math.max(0, Math.round(video.currentTime * layers.fps))
  );
  const projection = nearestPitchProjection(layers.pitch_to_video, currentFrame);
  const selected = new Set(selectedTrackIds);
  const firstVisibleFrame = Math.max(0, Math.min(startFrame, endFrame));
  const lastVisibleFrame = Math.min(layers.frames_processed - 1, Math.max(startFrame, endFrame));
  const renderScale = Math.min(cssWidth / sourceWidth, cssHeight / sourceHeight);
  const renderedWidth = sourceWidth * renderScale;
  const renderedHeight = sourceHeight * renderScale;
  const offsetX = (cssWidth - renderedWidth) / 2;
  const offsetY = (cssHeight - renderedHeight) / 2;

  for (const track of layers.tracks) {
    if (!selected.has(track.track_id)) continue;
    const useMetricPath = Boolean(projection && track.pitch_path.length);
    const path = useMetricPath ? track.pitch_path : track.video_path;
    let previous: { frame: number; x: number; y: number; metricX: number; metricY: number } | null = null;
    let latest: { x: number; y: number } | null = null;

    for (const point of path) {
      const frame = point[0];
      if (frame < firstVisibleFrame) continue;
      if (frame > lastVisibleFrame) break;
      const projected = useMetricPath
        ? projectPitchPoint(projection as number[], point[1], point[2])
        : [point[1], point[2]] as [number, number];
      if (!projected) {
        previous = null;
        continue;
      }
      const x = offsetX + projected[0] * renderScale;
      const y = offsetY + projected[1] * renderScale;
      if (x < -cssWidth || x > cssWidth * 2 || y < -cssHeight || y > cssHeight * 2) {
        previous = null;
        continue;
      }

      if (previous) {
        const frameGap = frame - previous.frame;
        const metricJump = useMetricPath
          ? Math.hypot(point[1] - previous.metricX, point[2] - previous.metricY)
          : 0;
        if (frameGap <= layers.fps * 2 && (!useMetricPath || metricJump <= 2500)) {
          const recency = lastVisibleFrame > firstVisibleFrame
            ? (frame - firstVisibleFrame) / (lastVisibleFrame - firstVisibleFrame)
            : 1;
          context.save();
          context.globalAlpha = 0.28 + Math.min(1, recency) * 0.68;
          context.strokeStyle = track.color;
          context.lineWidth = 3;
          context.lineCap = "round";
          context.lineJoin = "round";
          context.shadowColor = "rgba(0, 0, 0, 0.55)";
          context.shadowBlur = 3;
          context.beginPath();
          context.moveTo(previous.x, previous.y);
          context.lineTo(x, y);
          context.stroke();
          context.restore();
        }
      }
      previous = { frame, x, y, metricX: point[1], metricY: point[2] };
      latest = { x, y };
    }

    if (latest) {
      context.save();
      context.fillStyle = track.color;
      context.strokeStyle = "#ffffff";
      context.lineWidth = 2;
      context.beginPath();
      context.arc(latest.x, latest.y, 5, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.restore();
    }
  }
}

function InteractiveAnalysisViewer({
  layers,
  layersError,
  layersLoading,
  onVideoError,
  videoError,
  videoObject
}: {
  layers: MatchVisualLayers | null;
  layersError: string | null;
  layersLoading: boolean;
  onVideoError: () => void;
  videoError: string | null;
  videoObject: string;
}) {
  const [movementTracks, setMovementTracks] = useState<number[]>([]);
  const [heatmapTracks, setHeatmapTracks] = useState<number[]>([]);
  const [openPanel, setOpenPanel] = useState<"movement" | "heatmap" | "time" | null>(null);
  const [windowMode, setWindowMode] = useState<VisualWindowMode>("progress");
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(0.1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0.1);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [fullscreen, setFullscreen] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const tracks = useMemo(
    () => [...(layers?.tracks || [])].sort((left, right) => left.track_id - right.track_id),
    [layers]
  );
  const movementOptions = tracks.filter((track) => track.pitch_path.length || track.video_path.length);
  const heatmapOptions = tracks.filter((track) => track.pitch_path.length);
  const totalFrames = Math.max(0, (layers?.frames_processed || 1) - 1);
  const currentFrame = layers
    ? Math.min(totalFrames, Math.max(0, Math.round(currentTime * layers.fps)))
    : 0;
  const activeStartFrame = layers && windowMode === "range"
    ? Math.min(totalFrames, Math.max(0, Math.round(rangeStart * layers.fps)))
    : 0;
  const activeEndFrame = layers && windowMode === "range"
    ? Math.min(totalFrames, Math.max(activeStartFrame, Math.round(rangeEnd * layers.fps)))
    : currentFrame;

  useEffect(() => {
    if (!layers) return;
    const layerDuration = Math.max(0.1, layers.duration_seconds || layers.frames_processed / Math.max(1, layers.fps));
    setDuration(layerDuration);
    setRangeStart(0);
    setRangeEnd(layerDuration);
  }, [layers]);

  useEffect(() => {
    function handleFullscreenChange() {
      setFullscreen(document.fullscreenElement === stageRef.current);
      setOpenPanel(null);
    }
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !layers) return;
    let animationFrame = 0;
    const render = () => {
      const liveFrame = Math.min(
        layers.frames_processed - 1,
        Math.max(0, Math.round(video.currentTime * layers.fps))
      );
      const firstFrame = windowMode === "range" ? Math.round(rangeStart * layers.fps) : 0;
      const lastFrame = windowMode === "range" ? Math.round(rangeEnd * layers.fps) : liveFrame;
      drawMovementOverlay(canvas, video, layers, movementTracks, firstFrame, lastFrame);
    };
    const animate = () => {
      render();
      if (!video.paused && !video.ended) animationFrame = requestAnimationFrame(animate);
    };
    const begin = () => {
      cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(animate);
    };
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    video.addEventListener("loadedmetadata", render);
    video.addEventListener("play", begin);
    video.addEventListener("pause", render);
    video.addEventListener("seeked", render);
    video.addEventListener("timeupdate", render);
    video.addEventListener("ended", render);
    render();
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      video.removeEventListener("loadedmetadata", render);
      video.removeEventListener("play", begin);
      video.removeEventListener("pause", render);
      video.removeEventListener("seeked", render);
      video.removeEventListener("timeupdate", render);
      video.removeEventListener("ended", render);
    };
  }, [layers, movementTracks, rangeEnd, rangeStart, windowMode]);

  function updatePlaybackTime() {
    const video = videoRef.current;
    if (!video) return;
    const nextTime = video.currentTime;
    setCurrentTime(nextTime);
    if (windowMode === "range" && !video.paused && nextTime >= rangeEnd - 0.03) {
      video.pause();
      video.currentTime = rangeEnd;
      setCurrentTime(rangeEnd);
    }
  }

  function handleLoadedMetadata() {
    const video = videoRef.current;
    if (!video) return;
    const mediaDuration = Number.isFinite(video.duration) && video.duration > 0
      ? video.duration
      : layers?.duration_seconds || 0.1;
    setDuration(mediaDuration);
    if (!layers || rangeEnd <= 0.1) setRangeEnd(mediaDuration);
  }

  async function playSelectedRange() {
    const video = videoRef.current;
    if (!video) return;
    setWindowMode("range");
    video.currentTime = rangeStart;
    setCurrentTime(rangeStart);
    try {
      await video.play();
    } catch {
      setPlaying(false);
    }
  }

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (!video.paused) {
      video.pause();
      return;
    }
    if (windowMode === "range" && (video.currentTime < rangeStart || video.currentTime >= rangeEnd)) {
      video.currentTime = rangeStart;
      setCurrentTime(rangeStart);
    }
    try {
      await video.play();
    } catch {
      setPlaying(false);
    }
  }

  function seekVideo(nextTime: number) {
    const video = videoRef.current;
    if (!video) return;
    const boundedTime = Math.max(0, Math.min(duration, nextTime));
    video.currentTime = boundedTime;
    setCurrentTime(boundedTime);
  }

  function toggleMute() {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !video.muted;
    setMuted(video.muted);
  }

  function updateVolume(nextVolume: number) {
    const video = videoRef.current;
    if (!video) return;
    const boundedVolume = Math.max(0, Math.min(1, nextVolume));
    video.volume = boundedVolume;
    video.muted = boundedVolume === 0;
    setVolume(boundedVolume);
    setMuted(video.muted);
  }

  async function toggleFullscreen() {
    const stage = stageRef.current;
    if (!stage) return;
    if (document.fullscreenElement === stage) {
      await document.exitFullscreen();
      return;
    }
    await stage.requestFullscreen();
  }

  const aspectRatio = layers?.resolution?.[0] && layers?.resolution?.[1]
    ? layers.resolution[0] / layers.resolution[1]
    : 16 / 9;

  return (
    <div className="card analysis-workspace">
      <div className="section-header">
        <div>
          <h2 className="section-title">Interactive Analysis Video</h2>
          <div className="toolbar compact-toolbar">
            <span className="badge">{layers?.frames_processed ?? "-"} frames</span>
            <span className="badge">{layers ? `${layers.duration_seconds}s` : "layers pending"}</span>
            <span className="badge">
              {windowMode === "progress"
                ? `Live 00:00-${formatAnalysisTime(currentTime)}`
                : `Range ${formatAnalysisTime(rangeStart)}-${formatAnalysisTime(rangeEnd)}`}
            </span>
          </div>
        </div>
      </div>

      {layersLoading ? <div className="layer-state">Loading visual layers...</div> : null}
      {layersError ? <div className="layer-state error">{layersError}</div> : null}
      {!layersLoading && !layersError && !layers ? (
        <div className="layer-state">Visual layers were not generated for this run.</div>
      ) : null}

      <div className={`analysis-live-layout${layers && heatmapTracks.length ? " has-side-map" : ""}`}>
        <div
          className="analysis-video-stage"
          ref={stageRef}
          style={{ aspectRatio: `${aspectRatio}` }}
        >
          <video
            className="analysis-video interactive"
            onClick={togglePlayback}
            onDoubleClick={toggleFullscreen}
            onDurationChange={handleLoadedMetadata}
            onEnded={() => setPlaying(false)}
            onError={onVideoError}
            onLoadedMetadata={handleLoadedMetadata}
            onPause={() => setPlaying(false)}
            onPlay={() => setPlaying(true)}
            onTimeUpdate={updatePlaybackTime}
            playsInline
            preload="metadata"
            ref={videoRef}
            src={api.objectUrl(videoObject)}
          />
          <canvas aria-hidden="true" className="movement-overlay" ref={canvasRef} />

          <div className="stage-layer-toolbar">
            <TrackLayerPicker
              icon={<Waypoints size={16} />}
              label="Track lines"
              onChange={setMovementTracks}
              onOpenChange={(open) => setOpenPanel(open ? "movement" : null)}
              open={openPanel === "movement"}
              selected={movementTracks}
              tracks={movementOptions}
            />
            <TrackLayerPicker
              icon={<Flame size={16} />}
              label="Heatmap"
              onChange={setHeatmapTracks}
              onOpenChange={(open) => setOpenPanel(open ? "heatmap" : null)}
              open={openPanel === "heatmap"}
              selected={heatmapTracks}
              tracks={heatmapOptions}
            />
            <TimeWindowPicker
              duration={duration}
              mode={windowMode}
              onModeChange={setWindowMode}
              onOpenChange={(open) => setOpenPanel(open ? "time" : null)}
              onPlayRange={playSelectedRange}
              onRangeChange={(start, end) => {
                setRangeStart(start);
                setRangeEnd(end);
              }}
              open={openPanel === "time"}
              rangeEnd={rangeEnd}
              rangeStart={rangeStart}
            />
            <button
              aria-label="Clear visual layers"
              className="stage-icon-button"
              disabled={!movementTracks.length && !heatmapTracks.length}
              onClick={() => {
                setMovementTracks([]);
                setHeatmapTracks([]);
                setOpenPanel(null);
              }}
              title="Clear visual layers"
              type="button"
            >
              <Layers3 size={17} />
            </button>
          </div>

          {movementTracks.length ? (
            <div className="video-layer-legend">
              {tracks.filter((track) => movementTracks.includes(track.track_id)).map((track) => (
                <span key={track.track_id}>
                  <i style={{ backgroundColor: track.color }} /> T{track.track_id}
                </span>
              ))}
            </div>
          ) : null}

          {layers && heatmapTracks.length ? (
            <CompactPitchHeatmap
              currentFrame={currentFrame}
              endFrame={activeEndFrame}
              layers={layers}
              selectedTrackIds={heatmapTracks}
              startFrame={activeStartFrame}
            />
          ) : null}

          <div className="analysis-player-controls">
            <button
              aria-label={playing ? "Pause video" : "Play video"}
              onClick={togglePlayback}
              title={playing ? "Pause" : "Play"}
              type="button"
            >
              {playing ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <span className="player-time">{formatAnalysisTime(currentTime)}</span>
            <input
              aria-label="Video timeline"
              className="player-seek"
              max={Math.max(0.1, duration)}
              min={0}
              onChange={(event) => seekVideo(Number(event.target.value))}
              step="0.05"
              type="range"
              value={Math.min(currentTime, duration)}
            />
            <span className="player-time">{formatAnalysisTime(duration)}</span>
            <button
              aria-label={muted ? "Unmute video" : "Mute video"}
              onClick={toggleMute}
              title={muted ? "Unmute" : "Mute"}
              type="button"
            >
              {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
            </button>
            <input
              aria-label="Video volume"
              className="player-volume"
              max={1}
              min={0}
              onChange={(event) => updateVolume(Number(event.target.value))}
              step="0.05"
              type="range"
              value={muted ? 0 : volume}
            />
            <button
              aria-label={fullscreen ? "Exit full screen" : "Enter full screen"}
              onClick={toggleFullscreen}
              title={fullscreen ? "Exit full screen" : "Full screen"}
              type="button"
            >
              {fullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
            </button>
          </div>
        </div>

        {layers && heatmapTracks.length ? (
          <LivePitchHeatmap
            currentFrame={currentFrame}
            endFrame={activeEndFrame}
            layers={layers}
            mode={windowMode}
            selectedTrackIds={heatmapTracks}
            startFrame={activeStartFrame}
          />
        ) : null}
      </div>
      {videoError ? <p className="badge error">{videoError}</p> : null}

      {layers ? (
        <FullMatchVisualReview
          heatmapTrackIds={heatmapTracks}
          layers={layers}
          movementTrackIds={movementTracks}
        />
      ) : null}
    </div>
  );
}

function hexToRgba(hex: string, alpha: number) {
  const normalized = hex.replace("#", "");
  const value = Number.parseInt(normalized, 16);
  const red = (value >> 16) & 255;
  const green = (value >> 8) & 255;
  const blue = value & 255;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function drawMetricPitch(
  context: CanvasRenderingContext2D,
  field: { x: number; y: number; width: number; height: number },
  pitchLength: number,
  pitchWidth: number
) {
  const x = (value: number) => field.x + value / pitchLength * field.width;
  const y = (value: number) => field.y + value / pitchWidth * field.height;
  const penaltyWidth = 4032;
  const goalAreaWidth = 1832;
  context.save();
  context.strokeStyle = "rgba(255, 255, 255, 0.86)";
  context.fillStyle = "rgba(255, 255, 255, 0.92)";
  context.lineWidth = 1.4;
  context.strokeRect(field.x, field.y, field.width, field.height);
  context.beginPath();
  context.moveTo(x(pitchLength / 2), field.y);
  context.lineTo(x(pitchLength / 2), field.y + field.height);
  context.stroke();
  context.beginPath();
  context.arc(x(pitchLength / 2), y(pitchWidth / 2), 915 / pitchLength * field.width, 0, Math.PI * 2);
  context.stroke();
  context.strokeRect(x(0), y((pitchWidth - penaltyWidth) / 2), x(1650) - x(0), y((pitchWidth + penaltyWidth) / 2) - y((pitchWidth - penaltyWidth) / 2));
  context.strokeRect(x(pitchLength - 1650), y((pitchWidth - penaltyWidth) / 2), x(pitchLength) - x(pitchLength - 1650), y((pitchWidth + penaltyWidth) / 2) - y((pitchWidth - penaltyWidth) / 2));
  context.strokeRect(x(0), y((pitchWidth - goalAreaWidth) / 2), x(550) - x(0), y((pitchWidth + goalAreaWidth) / 2) - y((pitchWidth - goalAreaWidth) / 2));
  context.strokeRect(x(pitchLength - 550), y((pitchWidth - goalAreaWidth) / 2), x(pitchLength) - x(pitchLength - 550), y((pitchWidth + goalAreaWidth) / 2) - y((pitchWidth - goalAreaWidth) / 2));
  for (const spotX of [1100, pitchLength - 1100]) {
    context.beginPath();
    context.arc(x(spotX), y(pitchWidth / 2), 2.5, 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

function drawHeatmapCanvas(
  canvas: HTMLCanvasElement,
  layers: MatchVisualLayers,
  selectedTrackIds: number[],
  startFrame: number,
  endFrame: number,
  markerFrame?: number
) {
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;
  if (!cssWidth || !cssHeight) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssWidth * pixelRatio);
  canvas.height = Math.round(cssHeight * pixelRatio);
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);
  context.fillStyle = "#166534";
  context.fillRect(0, 0, cssWidth, cssHeight);
  const padding = Math.max(14, cssWidth * 0.025);
  const field = { x: padding, y: padding, width: cssWidth - padding * 2, height: cssHeight - padding * 2 };
  const selected = new Set(selectedTrackIds);
  const firstVisibleFrame = Math.max(0, Math.min(startFrame, endFrame));
  const lastVisibleFrame = Math.min(layers.frames_processed - 1, Math.max(startFrame, endFrame));
  const columns = 56;
  const rows = 36;

  context.save();
  context.beginPath();
  context.rect(field.x, field.y, field.width, field.height);
  context.clip();
  context.globalCompositeOperation = "screen";
  for (const track of layers.tracks) {
    if (!selected.has(track.track_id) || !track.pitch_path.length) continue;
    const density = new Float32Array(columns * rows);
    for (const point of track.pitch_path) {
      if (point[0] < firstVisibleFrame) continue;
      if (point[0] > lastVisibleFrame) break;
      const column = Math.min(columns - 1, Math.max(0, Math.floor(point[1] / layers.pitch.length_cm * columns)));
      const row = Math.min(rows - 1, Math.max(0, Math.floor(point[2] / layers.pitch.width_cm * rows)));
      density[row * columns + column] += 1;
    }
    const peak = Math.max(...density, 1);
    const radius = Math.max(18, field.width / 17);
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const count = density[row * columns + column];
        if (!count) continue;
        const intensity = Math.sqrt(count / peak);
        const centerX = field.x + (column + 0.5) / columns * field.width;
        const centerY = field.y + (row + 0.5) / rows * field.height;
        const gradient = context.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        gradient.addColorStop(0, hexToRgba(track.color, 0.18 + intensity * 0.58));
        gradient.addColorStop(0.48, hexToRgba(track.color, 0.09 + intensity * 0.26));
        gradient.addColorStop(1, hexToRgba(track.color, 0));
        context.fillStyle = gradient;
        context.fillRect(centerX - radius, centerY - radius, radius * 2, radius * 2);
      }
    }
  }
  context.restore();
  drawMetricPitch(context, field, layers.pitch.length_cm, layers.pitch.width_cm);

  if (markerFrame != null) {
    const targetFrame = Math.min(lastVisibleFrame, Math.max(firstVisibleFrame, markerFrame));
    for (const track of layers.tracks) {
      if (!selected.has(track.track_id) || !track.pitch_path.length) continue;
      let marker: number[] | null = null;
      for (const point of track.pitch_path) {
        if (point[0] < firstVisibleFrame) continue;
        if (point[0] > targetFrame) break;
        marker = point;
      }
      if (!marker) continue;
      const markerX = field.x + marker[1] / layers.pitch.length_cm * field.width;
      const markerY = field.y + marker[2] / layers.pitch.width_cm * field.height;
      const markerRadius = Math.max(7, Math.min(12, cssWidth / 42));
      context.save();
      context.fillStyle = track.color;
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.6;
      context.beginPath();
      context.arc(markerX, markerY, markerRadius, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.fillStyle = "#ffffff";
      context.font = `800 ${Math.max(8, markerRadius)}px system-ui, sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(track.track_id), markerX, markerY + 0.5);
      context.restore();
    }
  }
}

function PitchHeatmapCanvas({
  layers,
  selectedTrackIds,
  startFrame,
  endFrame,
  markerFrame
}: {
  layers: MatchVisualLayers;
  selectedTrackIds: number[];
  startFrame: number;
  endFrame: number;
  markerFrame?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const render = () => drawHeatmapCanvas(canvas, layers, selectedTrackIds, startFrame, endFrame, markerFrame);
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    render();
    return () => observer.disconnect();
  }, [endFrame, layers, markerFrame, selectedTrackIds, startFrame]);

  return <canvas ref={canvasRef} />;
}

function TrackLegend({ layers, selectedTrackIds }: { layers: MatchVisualLayers; selectedTrackIds: number[] }) {
  const selectedTracks = layers.tracks.filter((track) => selectedTrackIds.includes(track.track_id));

  return (
    <div className="heatmap-legend">
      {selectedTracks.map((track) => (
        <span key={track.track_id}>
          <i style={{ backgroundColor: track.color }} /> T{track.track_id}
        </span>
      ))}
    </div>
  );
}

function CompactPitchHeatmap({
  layers,
  selectedTrackIds,
  startFrame,
  endFrame,
  currentFrame
}: {
  layers: MatchVisualLayers;
  selectedTrackIds: number[];
  startFrame: number;
  endFrame: number;
  currentFrame: number;
}) {
  return (
    <aside className="compact-heatmap-overlay" aria-label="Selected player heatmap radar">
      <div className="compact-heatmap-title"><Flame size={13} /> Selected heatmap</div>
      <div className="compact-heatmap-body">
        <div className="compact-pitch-stage">
          <PitchHeatmapCanvas
            endFrame={endFrame}
            layers={layers}
            markerFrame={currentFrame}
            selectedTrackIds={selectedTrackIds}
            startFrame={startFrame}
          />
        </div>
        <TrackLegend layers={layers} selectedTrackIds={selectedTrackIds} />
      </div>
    </aside>
  );
}

function LivePitchHeatmap({
  layers,
  selectedTrackIds,
  startFrame,
  endFrame,
  currentFrame,
  mode
}: {
  layers: MatchVisualLayers;
  selectedTrackIds: number[];
  startFrame: number;
  endFrame: number;
  currentFrame: number;
  mode: VisualWindowMode;
}) {
  const startSeconds = startFrame / Math.max(1, layers.fps);
  const endSeconds = endFrame / Math.max(1, layers.fps);

  return (
    <aside className="live-heatmap-panel">
      <div className="live-map-header">
        <div>
          <span className="eyebrow">Metric player heatmap</span>
          <strong>{mode === "progress" ? "Live progression" : "Selected interval"}</strong>
        </div>
        <span className="badge">{formatAnalysisTime(startSeconds)}-{formatAnalysisTime(endSeconds)}</span>
      </div>
      <div className="live-pitch-stage">
        <PitchHeatmapCanvas
          endFrame={endFrame}
          layers={layers}
          markerFrame={currentFrame}
          selectedTrackIds={selectedTrackIds}
          startFrame={startFrame}
        />
      </div>
      <TrackLegend layers={layers} selectedTrackIds={selectedTrackIds} />
    </aside>
  );
}

function drawMovementPitchCanvas(
  canvas: HTMLCanvasElement,
  layers: MatchVisualLayers,
  selectedTrackIds: number[],
  startFrame: number,
  endFrame: number
) {
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;
  if (!cssWidth || !cssHeight) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssWidth * pixelRatio);
  canvas.height = Math.round(cssHeight * pixelRatio);
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);
  context.fillStyle = "#166534";
  context.fillRect(0, 0, cssWidth, cssHeight);
  const padding = Math.max(14, cssWidth * 0.025);
  const field = { x: padding, y: padding, width: cssWidth - padding * 2, height: cssHeight - padding * 2 };
  const selected = new Set(selectedTrackIds);
  const firstVisibleFrame = Math.max(0, Math.min(startFrame, endFrame));
  const lastVisibleFrame = Math.min(layers.frames_processed - 1, Math.max(startFrame, endFrame));
  const mapX = (value: number) => field.x + value / layers.pitch.length_cm * field.width;
  const mapY = (value: number) => field.y + value / layers.pitch.width_cm * field.height;

  context.save();
  context.beginPath();
  context.rect(field.x, field.y, field.width, field.height);
  context.clip();
  for (const track of layers.tracks) {
    if (!selected.has(track.track_id)) continue;
    let previous: number[] | null = null;
    let latest: number[] | null = null;
    for (const point of track.pitch_path) {
      if (point[0] < firstVisibleFrame) continue;
      if (point[0] > lastVisibleFrame) break;
      if (previous) {
        const frameGap = point[0] - previous[0];
        const distance = Math.hypot(point[1] - previous[1], point[2] - previous[2]);
        if (frameGap <= layers.fps * 2 && distance <= 2500) {
          context.save();
          context.strokeStyle = track.color;
          context.globalAlpha = 0.88;
          context.lineWidth = 2.4;
          context.lineCap = "round";
          context.beginPath();
          context.moveTo(mapX(previous[1]), mapY(previous[2]));
          context.lineTo(mapX(point[1]), mapY(point[2]));
          context.stroke();
          context.restore();
        }
      }
      previous = point;
      latest = point;
    }
    if (latest) {
      const x = mapX(latest[1]);
      const y = mapY(latest[2]);
      context.save();
      context.fillStyle = track.color;
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.5;
      context.beginPath();
      context.arc(x, y, 9, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.fillStyle = "#ffffff";
      context.font = "800 9px system-ui, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(track.track_id), x, y + 0.5);
      context.restore();
    }
  }
  context.restore();
  drawMetricPitch(context, field, layers.pitch.length_cm, layers.pitch.width_cm);
}

function MovementPitchMap({
  layers,
  selectedTrackIds,
  startFrame,
  endFrame
}: {
  layers: MatchVisualLayers;
  selectedTrackIds: number[];
  startFrame: number;
  endFrame: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const render = () => drawMovementPitchCanvas(canvas, layers, selectedTrackIds, startFrame, endFrame);
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    render();
    return () => observer.disconnect();
  }, [endFrame, layers, selectedTrackIds, startFrame]);

  return <canvas ref={canvasRef} />;
}

function FullMatchVisualReview({
  layers,
  movementTrackIds,
  heatmapTrackIds
}: {
  layers: MatchVisualLayers;
  movementTrackIds: number[];
  heatmapTrackIds: number[];
}) {
  const finalFrame = Math.max(0, layers.frames_processed - 1);

  return (
    <section className="full-match-visuals">
      <div className="section-header">
        <div>
          <h3 className="card-title">Full match visual review</h3>
          <p className="muted small">Complete analyzed interval, independent from video playback.</p>
        </div>
        <span className="badge">{formatAnalysisTime(layers.duration_seconds)}</span>
      </div>
      <div className="full-match-visual-grid">
        <section className="analysis-visual-panel">
          <div className="visual-panel-header">
            <strong>Full match movement paths</strong>
            <span>{movementTrackIds.length} selected</span>
          </div>
          {movementTrackIds.length ? (
            <>
              <div className="full-pitch-stage">
                <MovementPitchMap
                  endFrame={finalFrame}
                  layers={layers}
                  selectedTrackIds={movementTrackIds}
                  startFrame={0}
                />
              </div>
              <TrackLegend layers={layers} selectedTrackIds={movementTrackIds} />
            </>
          ) : (
            <div className="visual-empty">Select players or a team from Track lines above.</div>
          )}
        </section>
        <section className="analysis-visual-panel">
          <div className="visual-panel-header">
            <strong>Full match heatmap</strong>
            <span>{heatmapTrackIds.length} selected</span>
          </div>
          {heatmapTrackIds.length ? (
            <>
              <div className="full-pitch-stage">
                <PitchHeatmapCanvas
                  endFrame={finalFrame}
                  layers={layers}
                  selectedTrackIds={heatmapTrackIds}
                  startFrame={0}
                />
              </div>
              <TrackLegend layers={layers} selectedTrackIds={heatmapTrackIds} />
            </>
          ) : (
            <div className="visual-empty">Select players or a team from Heatmap above.</div>
          )}
        </section>
      </div>
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
  const visualLayersObject = summary?.visual_layers?.object_name || null;
  const visualLayers = useAsyncData<MatchVisualLayers | null>(
    () => visualLayersObject
      ? api.getMatchVisualLayers(visualLayersObject)
      : Promise.resolve(null),
    [visualLayersObject]
  );

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

          <InteractiveAnalysisViewer
            key={selectedRun.id}
            layers={visualLayers.data}
            layersError={visualLayers.error}
            layersLoading={visualLayers.loading}
            onVideoError={() => setVideoError("Browser could not play this generated video. Re-run the analysis or check the worker logs.")}
            videoError={videoError}
            videoObject={selectedRun.output_object}
          />

          <section className="grid two">
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
                    <tr><th>Interactive layers</th><td>{summary.visual_layers?.status || "Not generated"}</td></tr>
                    <tr><th>Radar frames rendered</th><td>{summary.radar?.rendered_frames ?? 0}</td></tr>
                    <tr><th>Team 1 control</th><td>{summary.team_ball_control?.team_1_percent ?? 0}%</td></tr>
                    <tr><th>Team 2 control</th><td>{summary.team_ball_control?.team_2_percent ?? 0}%</td></tr>
                    <tr><th>Output</th><td>{summary.output_object}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card">
              <h2 className="section-title">Artifacts</h2>
              <div className="artifact-actions">
                <a className="button" href={api.objectUrl(selectedRun.output_object)} target="_blank" rel="noreferrer">
                  Open video
                </a>
                {selectedRun.summary_object ? (
                  <a className="button" href={api.objectUrl(selectedRun.summary_object)} target="_blank" rel="noreferrer">
                    Open JSON
                  </a>
                ) : null}
                {summary.visual_layers?.object_name ? (
                  <a className="button" href={api.objectUrl(summary.visual_layers.object_name)} target="_blank" rel="noreferrer">
                    Open layer data
                  </a>
                ) : null}
              </div>
              <div className="layer-run-stats">
                <span><strong>{summary.visual_layers?.tracks_count ?? 0}</strong> selectable tracks</span>
                <span><strong>{summary.visual_layers?.movement_sample_rate_hz ?? 0} Hz</strong> movement paths</span>
                <span><strong>{summary.visual_layers?.heatmap_sample_rate_hz ?? 0} Hz</strong> heatmaps</span>
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
    { role: "agent", text: "Select a match in Match Analysis + or Reports, then ask for a tactical explanation, player plan, or training idea." }
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
