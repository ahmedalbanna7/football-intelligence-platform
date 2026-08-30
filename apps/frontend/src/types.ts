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
  goalkeeper_kit_image_object_name?: string | null;
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
  goalkeeper_kit_image_object_name?: string | null;
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
  canonical_track_id?: number;
  team?: number | null;
  role_name?: string | null;
  team_confidence?: number | null;
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
  corrections_applied?: number;
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
  pitch_calibration?: Array<{
    frame: number;
    confidence: number;
    source: string;
    reliable: boolean;
  }>;
  ball?: {
    track_id: number;
    pitch_path: Array<{
      frame: number;
      x: number;
      y: number;
      predicted: boolean;
      confidence: number;
    }>;
  };
  tracks: MatchVisualLayerTrack[];
};

export type MatchAnalysisPlusSummary = {
  status: string;
  engine: string;
  model: string;
  model_mode?: string;
  model_selection?: {
    strategy: string;
    selected: string;
    reason: string;
    preview_image_size: number;
    analysis_image_size: number;
    candidates: Record<string, {
      valid_players?: number;
      raw_players?: number;
      confidence_sum?: number;
      error?: string;
    }>;
  };
  ball_model?: string;
  ball_detection_mode?: string;
  pitch_model?: string;
  pitch_model_selection?: {
    strategy: string;
    selected?: string | null;
    reason?: string;
    preview_frames?: number[];
    candidates: Record<string, {
      path?: string;
      wide_view_frames?: number;
      valid_homographies?: number;
      visible_keypoints_total?: number;
      mean_inlier_ratio?: number;
      median_reprojection_error_cm?: number | null;
      error?: string;
    }>;
  };
  output_codec?: string;
  output_content_type?: string;
  frames_processed: number;
  max_frames: number;
  source_start_frame?: number;
  source_end_frame?: number;
  fps: number;
  processing_fps?: number;
  resolution: number[];
  detections_count: number;
  class_counts?: Record<string, number>;
  participant_role_counts?: Record<string, number>;
  track_role_counts?: Record<string, number>;
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
    specialized_detector_observations?: number;
  };
  pitch_occupancy_filter?: {
    engine: string;
    raw_player_candidates: number;
    kept_player_candidates: number;
    rejected_outside_pitch: number;
    rejected_non_field_foot: number;
    metric_decisions: number;
    visual_fallback_decisions: number;
  };
  ball_filter?: {
    engine: string;
    raw_ball_observations: number;
    kept_ball_observations: number;
    filtered_static_candidates: number;
    static_hits_threshold: number;
    pitch_stabilized_observations?: number;
    penalty_spot_rejections?: number;
    outside_pitch_rejections?: number;
    tracker?: {
      engine: string;
      observed_frames: number;
      interpolated_frames: number;
      rejected_motion_gate: number;
      current_confidence: number;
      maximum_interpolation_streak?: number;
      mahalanobis_rejections?: number;
    };
    quality_gate?: {
      status: "passed" | "needs_review";
      failed_conditions: string[];
      possession_ready: boolean;
      pass_detection_ready: boolean;
    };
  };
  team_classifier?: {
    engine: string;
    kit_anchors_bgr: Record<string, number[]>;
    classified_tracks: number;
    track_confidence?: Record<string, number>;
    color_observations: number;
    ambiguous_observations?: number;
    official_tracks?: number[];
    goalkeeper_tracks?: number[];
    anchor_initializations: number;
    assignment_sources?: Record<string, string>;
    goalkeeper_reference_matches?: number;
    quality_gate?: {
      status: "passed" | "needs_review";
      failed_conditions: string[];
      average_track_confidence: number;
      ambiguous_observation_ratio: number;
      similar_kits_detected: boolean;
      shadow_invariant_color_distance: boolean;
    };
  };
  kit_references?: {
    source: string;
    teams?: Record<string, {
      label?: string;
      loaded?: string[];
      missing?: string[];
      colors?: number[][];
    }>;
  };
  radar?: {
    engine: string;
    model_available: boolean;
    calibration_mode?: string | null;
    calibration_source?: string;
    confidence?: {
      current: number;
      average: number;
      minimum: number;
      reliable_frames: number;
      total_frames: number;
      threshold: number;
    };
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
    camera_cuts?: Array<{
      frame: number;
      recovered_frame?: number | null;
      recovery_frames?: number | null;
    }>;
    quality_gate?: {
      status: "passed" | "needs_manual_calibration";
      metric_outputs_verified: boolean;
      failed_conditions: string[];
      reliable_ratio: number;
      longest_unreliable_streak_frames: number;
      camera_cuts_detected: number;
      camera_cuts_recovered: number;
      unrecovered_camera_cuts: number;
      manual_fallback_available: boolean;
      manual_fallback_used: boolean;
      conditions: Array<{
        code: string;
        passed: boolean;
        value: unknown;
        required: unknown;
      }>;
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
    quality_verified?: boolean;
    quality_gate_status?: string;
    distance_speed_units?: string;
  };
  visual_layers?: {
    status: string;
    object_name: string;
    schema_version: number;
    tracks_count: number;
    movement_sample_rate_hz: number;
    heatmap_sample_rate_hz: number;
    corrections_applied?: number;
    canonical_identity_overlay?: boolean;
  };
  canonical_analytics?: {
    object_name: string;
    tracks_count: number;
    analytics_tracks_count: number;
    corrections_applied: number;
    role_counts: Record<string, number>;
    excluded_roles: Record<string, number>;
    teams: Record<string, {
      team?: number | null;
      players_count: number;
      total_distance_m: number;
      movement_samples: number;
      heatmap_samples: number;
    }>;
  };
  canonical_report?: {
    status: string;
    object_name: string;
    schema_version: string;
    teams_count: number;
    players_count: number;
  };
  canonical_video?: {
    status: string;
    strategy: string;
    base_object_name?: string | null;
    overlay_object_name: string;
  };
  analytics_real_v1?: {
    object_name?: string;
    status: string;
    analysis_scope: string;
    selected_teams: number[];
    quality_gate: {
      status: string;
      failed_conditions: string[];
      metric_outputs_released: boolean;
      ball_outputs_released: boolean;
    };
    players: Array<Record<string, unknown>>;
    teams: Record<string, Record<string, unknown>>;
    passing_candidates: Record<string, unknown>;
    formations: Record<string, unknown>;
    space_control: Record<string, unknown>;
  };
  reports_v2?: {
    status: string;
    schema_version: string;
    teams_count: number;
    players_count: number;
    artifacts: {
      json: string;
      pdf: string;
      team_chart: string;
      player_heatmaps: string;
    };
  };
  performance?: {
    engine: string;
    processing_fps: number;
    configured_device: string;
    cuda_available: boolean;
    gpu_active: boolean;
    cache_hit_frames: number;
    yolo_inference_frames: number;
    yolo_skipped_frames: number;
    detection_seconds: number;
    rendering_seconds: number;
    detection_cache?: {
      status: string;
      object_name: string;
      source: string;
      reusable_without_yolo: boolean;
    };
  };
  tracks?: Array<{
    track_id: number;
    canonical_track_id?: number;
    team?: number | null;
    role_name?: string;
    player_name?: string | null;
    frames?: number;
    distance_m?: number;
    last_speed_kmh?: number;
    average_speed_kmh?: number;
    max_speed_kmh?: number;
    movement_samples?: number;
    heatmap_samples?: number;
  }>;
  team_ball_control?: {
    team_1_percent: number;
    team_2_percent: number;
  };
  possession?: {
    engine: string;
    team_1_percent: number;
    team_2_percent: number;
    player_frames: Record<string, number>;
    transitions: number;
    completed_passes?: number;
    turnovers?: number;
    events?: Array<{
      type: "completed_pass" | "turnover" | "possession_change";
      frame: number;
      start_frame: number;
      duration_frames: number;
      from_track_id?: number | null;
      to_track_id: number;
      from_team?: number | null;
      to_team?: number | null;
      travel_m?: number | null;
      confidence: number;
    }>;
    unassigned_frames: number;
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
  analysis_config?: {
    start_frame?: number;
    reuse_run_id?: number | null;
    runtime_progress?: {
      stage: string;
      processed_frames: number;
      total_frames?: number | null;
      percent?: number | null;
      processing_fps?: number;
      eta_seconds?: number | null;
      cache_hit_frames?: number;
    };
    calibration_points?: Array<{
      image_x: number;
      image_y: number;
      pitch_x: number;
      pitch_y: number;
    }>;
  };
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

export type MatchAnalysisReportV2 = {
  schema_version: "reports_v2";
  engine: string;
  generated_at: string;
  analysis_scope: string;
  quality: Record<string, { status?: string; failed_conditions?: string[] }>;
  match: Record<string, unknown>;
  teams: Array<Record<string, unknown>>;
  players: Array<Record<string, unknown>>;
  events: Record<string, unknown>;
  charts: Record<string, unknown>;
  heatmaps: Array<Record<string, unknown>>;
  artifacts: {
    json: string;
    pdf: string;
    team_chart: string;
    player_heatmaps: string;
  };
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
  release_gate_status: string;
  release_gate?: {
    status: string;
    conditions: Array<{
      code: string;
      label: string;
      passed: boolean;
      actual?: unknown;
      required?: unknown;
      missing?: string[];
      status?: string;
    }>;
    thresholds?: Record<string, unknown>;
    unresolved_fragments?: number;
  } | null;
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
  role_name: "player" | "goalkeeper" | "referee" | "assistant_referee" | "staff_outside_pitch";
  role_confidence: number;
  role_locked: boolean;
  role_evidence: string[];
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
  assigned_role_name?: string | null;
  note?: string | null;
  undone: boolean;
  created_at?: string;
};

export type GroundTruthVerification = {
  status: "draft" | "verified";
  annotator?: string | null;
  reviewed_at?: string | null;
};

export type TrackingGroundTruthObject = {
  identity_id: string;
  bbox: [number, number, number, number];
  source_frame?: number | null;
  source_track_id?: number | null;
  source_raw_track_id?: number | null;
  team?: number | null;
  role_name?: TrackReviewItem["role_name"];
  review_state: "unverified" | "verified";
};

export type TrackingGroundTruthFrame = {
  frame: number;
  source_frame?: number | null;
  review_state?: "unverified" | "verified";
  objects: TrackingGroundTruthObject[];
};

export type TrackingGroundTruthDocument = {
  schema_version: "tracking_ground_truth.v2" | "tracking_ground_truth.v3";
  resolution?: [number, number];
  fps?: number;
  coverage?: string;
  verification: GroundTruthVerification;
  source?: Record<string, unknown>;
  clips?: Array<Record<string, unknown>>;
  instructions?: string[];
  frames: TrackingGroundTruthFrame[];
  [key: string]: unknown;
};

export type BallGroundTruthState = "visible" | "occluded" | "out_of_frame" | "uncertain";

export type BallGroundTruthFrame = {
  frame: number;
  source_frame?: number | null;
  state: BallGroundTruthState;
  review_state: "unverified" | "verified";
  ball?: {
    center: [number, number];
    bbox?: [number, number, number, number];
    airborne?: boolean | null;
    height_cm?: number | null;
    candidate_confidence?: number | null;
    candidate_predicted?: boolean;
  } | null;
};

export type BallGroundTruthDocument = {
  schema_version: "ball_ground_truth.v1";
  resolution?: [number, number];
  fps?: number;
  verification: GroundTruthVerification;
  source?: Record<string, unknown>;
  clips?: Array<Record<string, unknown>>;
  release_thresholds?: Record<string, number>;
  frames: BallGroundTruthFrame[];
  [key: string]: unknown;
};

export type GroundTruthValidation = {
  status: string;
  frame_count: number;
  verified_frames: number;
  ready_for_evaluation: boolean;
  annotation_count?: number;
  identity_count?: number;
  state_counts?: Record<string, number>;
};

export type TrackingQualityResponse = {
  run_id: number;
  match_id: number;
  annotation_video_object?: string | null;
  source_start_frame?: number;
  assessment: TrackingQualityAssessment;
  annotations?: {
    tracking?: Record<string, unknown>;
    ball?: Record<string, unknown>;
  };
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
