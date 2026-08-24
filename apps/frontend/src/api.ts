import type {
  MatchVisualLayers,
  MatchSummary,
  Player,
  PrimaryTeamProfile,
  MatchAnalysisPlusRun,
  MatchAnalysisReportV2,
  ReportResponse,
  RosterPlayer,
  Team,
  TrackingQualityResponse,
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

  getReport(matchId: number) {
    return request<ReportResponse>(`/matches/${matchId}/report`);
  },

  reportPdfUrl(matchId: number) {
    return `${API_BASE_URL}/matches/${matchId}/report.pdf`;
  },

  getMatchAnalysisReport(matchId: number, runId: number) {
    return request<MatchAnalysisReportV2>(
      `/match-analysis-plus/${matchId}/runs/${runId}/report`
    );
  },

  matchAnalysisReportPdfUrl(matchId: number, runId: number) {
    return `${API_BASE_URL}/match-analysis-plus/${matchId}/runs/${runId}/report.pdf`;
  },

  compareMatchAnalysisReports(cases: Array<{ match_id: number; run_id: number }>) {
    return request<Record<string, unknown>>("/match-analysis-plus/reports/compare", {
      method: "POST",
      json: { cases }
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

  getMatchAnalysisPlus(matchId: number) {
    return request<{
      match_id: number;
      match_title: string;
      runs: MatchAnalysisPlusRun[];
      latest: MatchAnalysisPlusRun | null;
    }>(`/match-analysis-plus/${matchId}`);
  },

  runMatchAnalysisPlus(matchId: number, payload: {
    mode: string;
    max_frames: number;
    start_frame?: number;
    calibration_points?: Array<{
      image_x: number;
      image_y: number;
      pitch_x: number;
      pitch_y: number;
    }>;
    reuse_run_id?: number | null;
  }) {
    return request<MatchAnalysisPlusRun>(`/match-analysis-plus/${matchId}/run`, {
      method: "POST",
      json: payload
    });
  },

  calibrationFrameUrl(matchId: number, frameIndex: number) {
    return `${API_BASE_URL}/match-analysis-plus/${matchId}/calibration-frame?frame_index=${Math.max(0, frameIndex)}`;
  },

  getMatchAnalysisPlusModes() {
    return request<{
      items: Array<{ value: string; label: string; description: string }>;
    }>("/match-analysis-plus/options/modes");
  },

  getMatchVisualLayers(objectName: string) {
    return request<MatchVisualLayers>(
      `/matches/artifacts/object?object_name=${encodeURIComponent(objectName)}`
    );
  },

  getTrackingQuality(matchId: number, runId: number) {
    return request<TrackingQualityResponse>(
      `/match-analysis-plus/${matchId}/runs/${runId}/quality`
    );
  },

  createTrackCorrection(
    matchId: number,
    runId: number,
    payload: {
      action: string;
      source_track_id: number;
      target_track_id?: number | null;
      split_frame?: number | null;
      assigned_player_id?: number | null;
      assigned_team_number?: number | null;
      assigned_role_name?: string | null;
      note?: string | null;
    }
  ) {
    return request<TrackingQualityResponse>(
      `/match-analysis-plus/${matchId}/runs/${runId}/quality/corrections`,
      { method: "POST", json: payload }
    );
  },

  undoTrackCorrection(matchId: number, runId: number, correctionId: number) {
    return request<TrackingQualityResponse>(
      `/match-analysis-plus/${matchId}/runs/${runId}/quality/corrections/${correctionId}/undo`,
      { method: "POST" }
    );
  },

  recalculateTrackingQuality(matchId: number, runId: number) {
    return request<{
      run_id: number;
      object_name: string;
      corrections_applied: number;
      tracks_count: number;
      quality: TrackingQualityResponse;
    }>(`/match-analysis-plus/${matchId}/runs/${runId}/quality/recalculate`, {
      method: "POST"
    });
  },

  benchmarkTrackingQuality(
    matchId: number,
    runId: number,
    groundTruth: Record<string, unknown>,
    iouThreshold = 0.5
  ) {
    return request<{
      metrics: Record<string, unknown>;
      quality: TrackingQualityResponse;
    }>(`/match-analysis-plus/${matchId}/runs/${runId}/quality/benchmark`, {
      method: "POST",
      json: { ground_truth: groundTruth, iou_threshold: iouThreshold }
    });
  },

  buildTrackingGroundTruthDraft(
    matchId: number,
    runId: number,
    payload: {
      start_frame: number;
      end_frame: number;
      sample_every_frames: number;
      track_ids?: number[];
      scenario?: string;
      camera_style?: string;
      critical?: boolean;
    }
  ) {
    return request<{
      object_name: string;
      frame_count: number;
      annotation_count: number;
      ground_truth: Record<string, unknown>;
    }>(`/match-analysis-plus/${matchId}/runs/${runId}/quality/ground-truth/draft`, {
      method: "POST",
      json: payload
    });
  },

  suggestTrackingCriticalRanges(
    matchId: number,
    runId: number,
    payload: { padding_frames?: number; max_ranges?: number } = {}
  ) {
    return request<{
      run_id: number;
      predictions_object: string;
      engine: string;
      status: string;
      events_detected: number;
      ranges: Array<{
        start_frame: number;
        end_frame: number;
        peak_frame: number;
        frame_count: number;
        scenarios: string[];
        track_ids: number[];
        severity: number;
        event_count: number;
      }>;
      instructions: string[];
    }>(`/match-analysis-plus/${matchId}/runs/${runId}/quality/critical-ranges/suggest`, {
      method: "POST",
      json: {
        padding_frames: payload.padding_frames ?? 20,
        max_ranges: payload.max_ranges ?? 12
      }
    });
  },

  buildTrackingReleasePlan(
    matchId: number,
    payload: {
      clip_size: number;
      overlap_frames: number;
      camera_style: string;
      critical_ranges?: Array<{ start_frame: number; end_frame: number; scenario: string }>;
    }
  ) {
    return request<{
      match_id: number;
      video_id: number;
      source_frames: number;
      fps: number;
      clip_size: number;
      overlap_frames: number;
      clips_count: number;
      critical_clips_count: number;
      clips: Array<{
        index: number;
        start_frame: number;
        end_frame: number;
        frame_count: number;
        camera_style: string;
        critical: boolean;
        scenarios: string[];
        run_request: {
          mode: string;
          start_frame: number;
          max_frames: number;
        };
      }>;
    }>(`/match-analysis-plus/${matchId}/quality/release-gate/plan`, {
      method: "POST",
      json: payload
    });
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
