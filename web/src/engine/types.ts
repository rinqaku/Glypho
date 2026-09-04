import type { ModelName, Quality } from './models';

export interface Point { x: number; y: number }

export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
  quad: [Point, Point, Point, Point];
  centerX: number;
  centerY: number;
  cropWidth: number;
  cropHeight: number;
  angle: number;
  score: number;
}

export interface Candidate {
  text: string;
  confidence: number;
  alternative?: { text: string; confidence: number };
}

export interface OcrLine extends Region {
  text: string;
  confidence: number;
  alternative?: { text: string; confidence: number };
  script?: string;
  language?: string;
}

export interface TimingBreakdown {
  preprocessMs: number;
  detectorInferenceMs: number;
  postprocessMs: number;
  recognitionMs: number;
  elapsedMs: number;
}

export interface OcrResult extends TimingBreakdown {
  text: string;
  lines: OcrLine[];
  provider: string;
  wasmThreads: number;
  quality: Quality;
  languages: string[];
}

export type EnginePhase = 'idle' | 'downloading' | 'initializing' | 'detecting' | 'recognizing' | 'ready' | 'done' | 'error';

export interface ProgressEvent {
  phase: EnginePhase;
  message: string;
  progress?: number;
  model?: ModelName;
  loadedBytes?: number;
  totalBytes?: number;
  cached?: boolean;
}

export interface ModelState {
  model: ModelName;
  status: 'idle' | 'downloading' | 'cached' | 'initializing' | 'ready' | 'error';
  progress: number;
  provider?: string;
  loadedBytes?: number;
  totalBytes?: number;
  message?: string;
}