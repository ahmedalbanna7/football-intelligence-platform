export type RouteKey =
  | "dashboard"
  | "settings"
  | "my-team"
  | "teams"
  | "matches"
  | "analysis"
  | "first-analysis"
  | "match-analysis-plus"
  | "reports"
  | "agent"
  | "recommendations";

export type MatchSummary = {
  id: number;
  title: string;
  status: string;
  match_context?: MatchContext;
  latest_processing_job?: ProcessingJob | null;
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

export type ProcessingJob = {
  id: number;
  video_id: number;
  status: string;
  started_at?: string;
  finished_at?: string;
  error_message?: string | null;
  result_summary?: ProcessingSummary | null;
};

export type ProcessingSummary = {
  status?: string;
  detections?: {
    status?: string;
    engine?: string | null;
    model?: string | null;
    device?: string | null;
    fallback_reason?: string | null;
    frames_processed?: number | null;
    frames_requested?: number | null;
    frames_skipped?: number | null;
    detections_count?: number | null;
    class_counts?: Record<string, number> | null;
    raw_class_counts?: Record<string, number> | null;
    confidence?: {
      min?: number | null;
      max?: number | null;
      avg?: number | null;
    } | null;
    elapsed_ms?: number | null;
  };
  tracking?: {
    tracks_count?: number | null;
    engine?: string | null;
    mode?: string | null;
    detections_received?: number | null;
  };
  events?: {
    events_count?: number | null;
    tracks_received?: number | null;
    event_types?: string[] | null;
  };
  tactical_identity?: {
    engine?: string | null;
    assignments_count?: number | null;
    resolved_count?: number | null;
  };
  artifacts?: {
    status?: string | null;
    storage?: string | null;
    paths?: Record<string, string> | null;
    track_observations_count?: number | null;
    detections_count?: number | null;
    tracks_count?: number | null;
  };
  crops?: {
    status?: string | null;
    crops_count?: number | null;
    jersey_crops_count?: number | null;
    crops_prefix?: string | null;
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

export type TrackAssignment = {
  id: number;
  track_id: number;
  team_context: string;
  player_name: string;
  shirt_number?: number | null;
  position?: string | null;
  created_at?: string;
  updated_at?: string;
};

export type TrackSummary = {
  track_id: number;
  object_key?: string | null;
  class_name?: string | null;
  team_context?: string | null;
  team_assignment_source?: string | null;
  team_assignment_confidence?: number | null;
  recognized_shirt_number?: number | null;
  shirt_number_confidence?: number | null;
  shirt_number_source?: string | null;
  frames_count?: number;
  first_frame?: number | null;
  last_frame?: number | null;
  crop_samples?: Array<{
    frame_index: number;
    crop_path?: string | null;
    jersey_crop_path?: string | null;
  }>;
  dominant_colors?: Array<{
    hex?: string;
    rgb?: number[];
    coverage?: number;
  }>;
  kit_match_score?: number | null;
  assignment?: TrackAssignment | null;
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

export type FirstAnalysisSummary = {
  status: string;
  engine: string;
  model: string;
  match_id: number;
  input_object: string;
  output_object: string;
  summary_object: string;
  original_project_path: string;
  output_codec?: string;
  output_content_type?: string;
  frames_processed: number;
  max_frames: number;
  fps: number;
  resolution: number[];
  detections_count: number;
  player_tracks_count: number;
  ball_observations: number;
  elapsed_ms: number;
  team_ball_control: {
    team_1_percent: number;
    team_2_percent: number;
  };
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
  tracks?: Array<{
    track_id: number;
    team?: number | null;
    frames?: number;
    distance_m?: number;
    last_speed_kmh?: number;
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
