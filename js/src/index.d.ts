export type Segmentation = 'auto' | 'single_block' | 'single_line' | 'sparse_text';
export type Quality = 'fast' | 'balanced' | 'accurate' | 'maximum';
export type Device = 'auto' | 'cpu' | 'cuda' | 'coreml' | 'openvino';

export interface GlyphoOptions {
  binary?: string | URL;
  models?: string | URL;
  quality?: Quality;
  device?: Device;
  languages?: string[];
  threads?: number;
  offline?: boolean;
  timeoutMs?: number;
}

export interface RecognitionOptions {
  languages?: string[];
  segmentation?: Segmentation;
  minConfidence?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface Point {
  x: number;
  y: number;
}

export interface Quad {
  points: [Point, Point, Point, Point];
}

export interface TextWord {
  id: string;
  quad: Quad;
  text: string;
  confidence?: number;
}

export interface TextLine {
  id: string;
  order: number;
  quad: Quad;
  text: string;
  corrected_text?: string;
  alternatives?: Array<{ text: string; confidence: number }>;
  confidence?: number;
  language?: string;
  script?: string;
  direction: 'auto' | 'left_to_right' | 'right_to_left' | 'vertical';
  legibility: 'clear' | 'ambiguous' | 'unreadable';
  flags?: string[];
  evaluation: {
    detection: boolean;
    recognition: boolean;
  };
  source: 'manual' | 'model' | 'imported';
  words: TextWord[];
  ignored: boolean;
}

export interface GlyphoDocument {
  schema_version: 'glypho.annotation.v1';
  coordinate_system: 'pixel_top_left';
  image: {
    id: string;
    file_name: string;
    width: number;
    height: number;
    sha256?: string;
  };
  text: string;
  corrected_text?: string;
  lines: TextLine[];
  language_hints: string[];
  metadata: Record<string, unknown>;
}

export class GlyphoError extends Error {}

export interface RuntimeInfo {
  runtime: string;
  quality: Quality;
  model: string;
  languages: string[];
  models_dir: string;
  requested_device: Device;
  device: Device;
  fallback_reason?: string;
}

export class Glypho {
  constructor(options?: GlyphoOptions);
  recognize(image: string | URL, options?: RecognitionOptions): Promise<GlyphoDocument>;
  warmup(options?: Pick<RecognitionOptions, 'languages' | 'timeoutMs' | 'signal'>): Promise<{ warmed: true }>;
  info(options?: Pick<RecognitionOptions, 'timeoutMs' | 'signal'>): Promise<RuntimeInfo>;
  close(): Promise<void>;
}

export function resolveBinary(configured?: string | URL): string;
