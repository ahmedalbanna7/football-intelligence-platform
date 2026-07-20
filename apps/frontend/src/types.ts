export type RouteKey =
  | "dashboard"
  | "settings"
  | "my-team"
  | "teams"
  | "matches"
  | "match-analysis-plus"
  | "reports"
  | "agent"
  | "recommendations";

export type MatchSummary = {
  id: number;
  title: string;
  status: string;
  match_context?: MatchContext;
  latest_match_analysis_run?: MatchAnalysisPlusRun | null;
};

export type MatchContext = {
  match_category?: string;
  match_type?: string;
  matchup_type?: string;
  analysis_scope?: string;
  primary_team_name?: string | null;
  primary_team_id?: number | null;
  opponent_team_name?: string | null;
  opponent_team_id?: number | null;
  another_team_name?: string | null;
  another_team_id?: number | null;
  formation?: string | null;
  primary_formation?: string | null;
  another_formation?: string | null;
  tactical_identity?: {
    primary_team_id?: number | null;
    formation?: string | null;
    lineup?: LineupEntry[];
    substitutions?: SubstitutionEntry[];
  };
};

export type Team = {
  id: number;
  name: string;
  team_type?: string;
  primary_kit_image_object_name?: string | null;
  alternate_kit_image_object_name?: string | null;
  notes?: string | null;
};

export type Player = {
  id: number;
  team_id: number;
  name: string;
  jersey_number?: number | null;
  age?: number;
  position?: string;
  primary_zone?: string | null;
  secondary_zones?: string[];
  position_label?: string | null;
  preferred_side?: string | null;
  notes?: string | null;
};

export type PrimaryTeamProfile = {
  id?: number;
  team_name?: string;
  primary_kit_image_object_name?: string | null;
  alternate_kit_image_object_name?: string | null;
};

export type RosterPlayer = {
  id: number;
  team_context: string;
  player_name: string;
  shirt_number: number;
  position?: string | null;
  primary_zone?: string | null;
  secondary_zones?: string[];
  position_label?: string | null;
  preferred_side?: string | null;
  notes?: string | null;
  created_at?: string;
  updated_at?: string;
};

export type LineupEntry = {
  id?: number;
  player_id: number;
  player_name?: string | null;
  player?: {
    player_id: number;
    name: string;
    jersey_number?: number | null;
    primary_zone?: string | null;
  } | null;
  jersey_number?: number | null;
  starting_zone?: string | null;
  expected_zones?: string[];
  is_starter?: boolean;
  start_minute?: number;
};

export type SubstitutionEntry = {
  id?: number;
  minute: number;
  second?: number | null;
  player_out_id?: number | null;
  player_in_id: number;
  player_in_zone?: string | null;
  expected_zones?: string[];
  notes?: string | null;
};

export type IdentityAssignment = {
  track_id: number;
  team_context?: string | null;
  resolved_player_id?: number | null;
  resolved_player?: {
    player_id: number;
    name?: string | null;
    jersey_number?: number | null;
    zone?: string | null;
  } | null;
  confidence?: number | null;
  zone?: string | null;
  candidates?: Array<{
    player_id: number;
    name?: string | null;
    score: number;
    reasons?: Record<string, number | null>;
  }>;
};

export type ReportResponse = {
  match_id: number;
  match_status: string;
  report?: {
    status?: string;
    data?: {
      summary?: Record<string, unknown>;
      counts?: Record<string, number>;
      teams?: Record<string, unknown>;
      charts?: {
        team_distance?: Array<Record<string, unknown>>;
        player_speed?: Array<Record<string, unknown>>;
      };
      identity?: {
        assignments?: IdentityAssignment[];
        manual_assignments?: IdentityAssignment[];
        manual_resolved_count?: number;
      };
    };
  } | null;
};

export type MatchVisualLayerTrack = {
  track_id: number;
  team?: number | null;
  color: string;
  frames: number;
  first_frame?: number | null;
  last_frame?: number | null;
  video_path: number[][];
  pitch_path: number[][];
  identity_confidence?: number | null;
  switch_risk?: "low" | "medium" | "high" | null;
  player_id?: number | null;
  player_name?: string | null;
  jersey_number?: number | null;
};

export type MatchVisualLayers = {
  schema_version: number;
  coordinate_systems: {
    video: string;
    pitch: string;
    ground_plane_z_cm: number;
  };
  fps: number;
  frames_processed: number;
  duration_seconds: number;
  resolution: number[];
  movement_sample_rate_hz: number;
  heatmap_sample_rate_hz: number;
  pitch: {
    length_cm: number;
    width_cm: number;
  };
  pitch_to_video: number[][];
  tracks: MatchVisualLayerTrack[];
};

export type MatchAnalysisPlusSummary = {
  status: string;
  engine: string;
  model: string;
  model_mode?: string;
  output_codec?: string;
  output_content_type?: string;
  frames_processed: number;
  max_frames: number;
  fps: number;
  processing_fps?: number;
  resolution: number[];
  detections_count: number;
  class_counts?: Record<string, number>;
  confidence?: {
    avg?: number | null;
    min?: number | null;
    max?: number | null;
  };
  tracks_count: number;
  raw_tracks_count?: number;
  player_filter?: {
    engine: string;
    raw_player_detections: number;
    kept_player_detections: number;
    rejected_implausible_shape: number;
    rejected_field_fixtures: number;
  };
  ball_filter?: {
    engine: string;
    raw_ball_observations: number;
    kept_ball_observations: number;
    filtered_static_candidates: number;
    static_hits_threshold: number;
    pitch_stabilized_observations?: number;
  };
  team_classifier?: {
    engine: string;
    kit_anchors_bgr: Record<string, number[]>;
    classified_tracks: number;
    color_observations: number;
    anchor_initializations: number;
  };
  radar?: {
    engine: string;
    model_available: boolean;
    calibration_mode?: string | null;
    calibration_attempts: number;
    successful_calibrations: number;
    goal_geometry_calibrations?: number;
    rendered_frames: number;
    last_visible_keypoints: number;
    last_inliers: number;
    last_reprojection_error_cm?: number | null;
    last_line_alignment_score?: number | null;
    camera_tracking?: {
      engine: string;
      attempts: number;
      successes: number;
      failures: number;
      last_inliers: number;
      last_inlier_ratio?: number | null;
      last_reprojection_error_px?: number | null;
    };
    pitch_template?: {
      name: string;
      length_cm: number;
      width_cm: number;
    };
    projection_model?: string;
    coordinate_system: string;
    errors: number;
  };
  metric_tracking?: {
    coordinate_system: string;
    ground_plane_z_cm: number;
    trajectory_sample_rate_hz: number;
    heatmap_ready: boolean;
  };
  visual_layers?: {
    status: string;
    object_name: string;
    schema_version: number;
    tracks_count: number;
    movement_sample_rate_hz: number;
    heatmap_sample_rate_hz: number;
  };
  tracks?: Array<{
    track_id: number;
    team?: number | null;
    frames?: number;
    distance_m?: number;
    last_speed_kmh?: number;
    movement_samples?: number;
    heatmap_samples?: number;
  }>;
  team_ball_control?: {
    team_1_percent: number;
    team_2_percent: number;
  };
  elapsed_ms: number;
  output_object?: string;
  summary_object?: string;
  thumbnail_object?: string | null;
  source_project?: string;
  worker?: string;
  notes?: string[];
};

export type MatchAnalysisPlusRun = {
  id: number;
  match_id: number;
  video_id: number;
  mode: string;
  status: string;
  source: string;
  max_frames: number;
  output_object?: string | null;
  summary_object?: string | null;
  thumbnail_object?: string | null;
  summary?: MatchAnalysisPlusSummary | null;
  error_message?: string | null;
  created_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  quality?: {
    status: string;
    average_identity_confidence?: number | null;
    tracks_needing_review: number;
    benchmark_status: string;
    idf1?: number | null;
    hota?: number | null;
  } | null;
};

export type TrackingQualityAssessment = {
  id: number;
  status: string;
  tracker_engine?: string | null;
  reid_enabled: boolean;
  reid_model?: string | null;
  average_identity_confidence?: number | null;
  suspected_id_switches: number;
  fragmented_tracks: number;
  tracks_needing_review: number;
  benchmark_status: string;
  id_switches?: number | null;
  idf1?: number | null;
  hota?: number | null;
  fragmentation?: number | null;
  predictions_object?: string | null;
  ground_truth_object?: string | null;
  metrics?: Record<string, unknown> | null;
  thresholds?: Record<string, number> | null;
  updated_at?: string;
  reviewed_at?: string | null;
};

export type TrackReviewItem = {
  id: number;
  track_id: number;
  canonical_track_id: number;
  team?: number | null;
  assigned_player_id?: number | null;
  assigned_player?: {
    id: number;
    name: string;
    jersey_number?: number | null;
  } | null;
  status: string;
  identity_confidence: number;
  reid_confidence: number;
  motion_consistency: number;
  team_consistency: number;
  switch_risk: "low" | "medium" | "high";
  fragment_count: number;
  raw_id_transitions: number;
  first_frame?: number | null;
  last_frame?: number | null;
  observation_count: number;
  raw_track_ids: number[];
  issue_codes: string[];
  crop_objects: Array<{
    frame: number;
    object_name: string;
    confidence?: number | null;
  }>;
  observations: Array<{
    frame: number;
    track_id: number;
    raw_track_id?: number | null;
    bbox: number[];
    confidence?: number | null;
  }>;
};

export type TrackReviewCorrection = {
  id: number;
  action: string;
  source_track_id?: number | null;
  target_track_id?: number | null;
  split_frame?: number | null;
  assigned_player_id?: number | null;
  assigned_team_number?: number | null;
  note?: string | null;
  undone: boolean;
  created_at?: string;
};

export type TrackingQualityResponse = {
  run_id: number;
  match_id: number;
  assessment: TrackingQualityAssessment;
  tracks: TrackReviewItem[];
  corrections: TrackReviewCorrection[];
  players: Array<{
    id: number;
    name: string;
    jersey_number?: number | null;
    team_id: number;
  }>;
  correction_id?: number;
  recalculation?: Record<string, unknown> | null;
};

export type YoloStatus = {
  status: string;
  engine: string;
  mode: string;
  model: string;
  model_file_exists: boolean;
  model_loaded: boolean;
  error?: string | null;
  confidence: number;
  image_size: number;
  device: string;
  max_detections_per_frame: number;
  allowed_class_names: string[];
  classes: Record<string, string>;
};
