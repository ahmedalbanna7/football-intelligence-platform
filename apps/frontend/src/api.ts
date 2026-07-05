import type {
  IdentityAssignment,
  MatchSummary,
  Player,
  PrimaryTeamProfile,
  ProcessingJob,
  FirstAnalysisSummary,
  MatchAnalysisPlusRun,
  ReportResponse,
  RosterPlayer,
  Team,
  TrackSummary,
  YoloStatus
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

type RequestOptions = RequestInit & {
  json?: unknown;
};

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  let body = options.body;

  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    body
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  baseUrl: API_BASE_URL,

  listMatches(limit = 20) {
    return request<{ items: MatchSummary[]; limit: number; offset: number }>(
      `/matches/?limit=${limit}`
    );
  },

  getYoloStatus() {
    return request<YoloStatus>("/ai/yolo/status");
  },

  getProcessing(matchId: number) {
    return request<{
      match_id: number;
      match_status: string;
      job: ProcessingJob | null;
    }>(`/matches/${matchId}/processing`);
  },

  getReport(matchId: number) {
    return request<ReportResponse>(`/matches/${matchId}/report`);
  },

  reportPdfUrl(matchId: number) {
    return `${API_BASE_URL}/matches/${matchId}/report.pdf`;
  },

  getIdentityAssignments(matchId: number) {
    return request<{
      match_id: number;
      source: string;
      job_id?: number | null;
      assignments: IdentityAssignment[];
    }>(`/matches/${matchId}/identity-assignments`);
  },

  getTracks(matchId: number) {
    return request<{
      match_id: number;
      match_status: string;
      job_id: number;
      job_status: string;
      tracks: TrackSummary[];
    }>(`/matches/${matchId}/tracks`);
  },

  saveTrackAssignment(
    matchId: number,
    payload: {
      track_id: number;
      player_name: string;
      team_context?: string | null;
      shirt_number?: number | null;
      position?: string | null;
    }
  ) {
    return request(`/matches/${matchId}/track-assignments`, {
      method: "POST",
      json: payload
    });
  },

  autoAssignTracks(matchId: number) {
    return request(`/matches/${matchId}/auto-assign-tracks`, {
      method: "POST"
    });
  },

  objectUrl(objectName: string) {
    return `${API_BASE_URL}/matches/artifacts/object?object_name=${encodeURIComponent(objectName)}`;
  },

  uploadMatch(form: FormData) {
    return request<{ match_id: number; video_id: number; status: string }>(
      "/matches/upload",
      {
        method: "POST",
        body: form
      }
    );
  },

  deleteMatch(matchId: number) {
    return request<{ match_id: number; deleted: boolean; deleted_objects: number }>(
      `/matches/${matchId}`,
      {
        method: "DELETE"
      }
    );
  },

  getFirstAnalysis(matchId: number) {
    return request<{ exists: boolean; summary: FirstAnalysisSummary | null }>(
      `/first-analysis/${matchId}`
    );
  },

  runFirstAnalysis(matchId: number, maxFrames = 450) {
    return request<FirstAnalysisSummary>(
      `/first-analysis/${matchId}/run?max_frames=${encodeURIComponent(maxFrames)}`,
      {
        method: "POST"
      }
    );
  },

  getMatchAnalysisPlus(matchId: number) {
    return request<{
      match_id: number;
      match_title: string;
      runs: MatchAnalysisPlusRun[];
      latest: MatchAnalysisPlusRun | null;
    }>(`/match-analysis-plus/${matchId}`);
  },

  runMatchAnalysisPlus(matchId: number, payload: { mode: string; max_frames: number }) {
    return request<MatchAnalysisPlusRun>(`/match-analysis-plus/${matchId}/run`, {
      method: "POST",
      json: payload
    });
  },

  getMatchAnalysisPlusModes() {
    return request<{
      items: Array<{ value: string; label: string; description: string }>;
    }>("/match-analysis-plus/options/modes");
  },

  getPrimaryTeam() {
    return request<PrimaryTeamProfile>("/primary-team/");
  },

  savePrimaryTeam(form: FormData) {
    return request<PrimaryTeamProfile>("/primary-team/", {
      method: "POST",
      body: form
    });
  },

  createTeam(payload: string | { name: string; team_type?: string; notes?: string | null }) {
    const body = typeof payload === "string" ? { name: payload } : payload;
    return request<Team>("/teams/", {
      method: "POST",
      json: body
    });
  },

  listTeams(teamType?: string) {
    const suffix = teamType ? `?team_type=${encodeURIComponent(teamType)}` : "";
    return request<{ items: Team[] }>(`/teams/${suffix}`);
  },

  saveTeamProfile(teamId: number, form: FormData) {
    return request<Team>(`/teams/${teamId}/profile`, {
      method: "POST",
      body: form
    });
  },

  deleteTeam(teamId: number) {
    return request<{ team_id: number; deleted: boolean; deleted_players: number }>(
      `/teams/${teamId}`,
      {
        method: "DELETE"
      }
    );
  },

  listTeamMatches(teamId: number) {
    return request<{
      team: Team;
      matches: Array<{
        id: number;
        title: string;
        status: string;
        match_type?: string;
        analysis_scope?: string;
        opponent_team_name?: string | null;
        created_at?: string;
      }>;
    }>(`/teams/${teamId}/matches`);
  },

  createTeamPlayer(teamId: number, payload: Partial<Player>) {
    return request<Player>(`/teams/${teamId}/players`, {
      method: "POST",
      json: payload
    });
  },

  deleteTeamPlayer(teamId: number, playerId: number) {
    return request<{ team_id: number; player_id: number; deleted: boolean }>(
      `/teams/${teamId}/players/${playerId}`,
      {
        method: "DELETE"
      }
    );
  },

  listTeamPlayers(teamId: number) {
    return request<{ team: Team; players: Player[] }>(`/teams/${teamId}/players`);
  },

  listPrimaryTeamPlayers() {
    return request<{ items: RosterPlayer[] }>("/primary-team/players");
  },

  savePrimaryTeamPlayer(payload: {
    player_name: string;
    shirt_number: number;
    position?: string | null;
    primary_zone?: string | null;
    secondary_zones?: string[];
    position_label?: string | null;
    preferred_side?: string | null;
    notes?: string | null;
  }) {
    return request<RosterPlayer>("/primary-team/players", {
      method: "POST",
      json: payload
    });
  },

  deletePrimaryTeamPlayer(entryId: number) {
    return request<{ entry_id: number; deleted: boolean }>(
      `/primary-team/players/${entryId}`,
      {
        method: "DELETE"
      }
    );
  },

  patchPlayer(playerId: number, payload: Partial<Player>) {
    return request<Player>(`/players/${playerId}`, {
      method: "PATCH",
      json: payload
    });
  },

  saveLineup(
    matchId: number,
    payload: {
      team_id: number | null;
      players: Array<{
        player_id: number;
        jersey_number?: number | null;
        starting_zone?: string | null;
        expected_zones?: string[];
        is_starter?: boolean;
        start_minute?: number;
      }>;
    }
  ) {
    return request(`/matches/${matchId}/lineup`, {
      method: "POST",
      json: payload
    });
  },

  getLineup(matchId: number) {
    return request(`/matches/${matchId}/lineup`);
  },

  saveSubstitutions(
    matchId: number,
    payload: {
      substitutions: Array<{
        team_id?: number | null;
        minute: number;
        second?: number | null;
        player_out_id?: number | null;
        player_in_id: number;
        player_in_zone?: string | null;
        expected_zones?: string[];
        notes?: string | null;
      }>;
    }
  ) {
    return request(`/matches/${matchId}/substitutions`, {
      method: "POST",
      json: payload
    });
  }
};
