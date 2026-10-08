export type StateKey = 'P' | 'I' | 'N' | 'Q';

export interface RuntimeSnapshot {
  runtime_id: string | null;
  state_status: 'LIVE_RESEARCH' | 'WAITING_FOR_LIVE_RUNTIME' | string;
  source: string | null;
  subject_id: string | null;
  event_id: string | null;
  time: number | null;
  state: Partial<Record<StateKey, number>> | null;
  ci: Partial<Record<StateKey, number>> | null;
  metrics: {
    observations?: number;
    pet_uncertainty_channels?: number;
    state_trace?: number | null;
    oos_temporal_leakage?: number;
  } | null;
  provenance: Provenance | null;
  pit: PITEnvelope | null;
  oos: OOSResult | null;
  pet_kinetic_posterior: PETPosterior | null;
  pet_to_state_assimilation: PETAssimilation | null;
  trajectory: Trajectory | null;
  prediction: Record<string, unknown> | null;
  evidence: Record<string, unknown> | null;
  evidence_id: string | null;
  updated_at?: number;
}

export interface Provenance {
  source_name: string;
  dataset_id: string;
  dataset_version: string;
  model_version: string;
  processing_pipeline: string;
  processing_version: string;
  raw_hashes: string[];
  feature_hashes: string[];
  observation_hash: string;
  result_hash: string | null;
  retrieval_uri: string | null;
  access_tier: string;
}

export interface PITEnvelope {
  as_of: string;
  acquisition_max: string;
  result_max: string;
  ingest_time: string;
}

export interface OOSResult {
  status: string;
  temporal_leakage: boolean;
  metrics: Record<string, number>;
}

export interface PETPosterior {
  names: string[];
  mean: number[];
  covariance: number[][];
  uncertainty_sources: string[];
  kinetic_model: string;
  inference_version: string;
}

export interface PETAssimilation {
  observable_names: string[];
  effective_covariance: number[][];
  parameter_uncertainty_contribution: number[][];
  innovation: number[];
  innovation_covariance: number[][];
  kalman_gain: number[][];
  method: string;
  assumptions: string[];
}

export interface Trajectory {
  times?: number[];
  states?: Record<string, number[]>;
  ci95?: Record<string, number[]>;
  [key: string]: unknown;
}

export interface CapabilitySet {
  state_space: boolean;
  pinq: boolean;
  mri: boolean;
  pet_frame_integrated: boolean;
  pet_aif_uncertainty: boolean;
  pet_uncertainty_to_state_posterior: boolean;
  pet_to_neuro_observation_registry_required: boolean;
  hierarchical_map: boolean;
  bayesian_runtime: boolean;
  numpyro_available: boolean;
  runtime_publish: boolean;
  runtime_waiting_without_data: boolean;
  websocket_stream: boolean;
  synthetic_reference: boolean;
}

export interface SourceCatalog {
  [key: string]: unknown;
}
