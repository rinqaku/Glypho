import * as ort from 'onnxruntime-web/webgpu';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import ortWasmModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';

import { MODELS, QUALITY, detectorFor, primaryRecognizerFor, type Artifact, type ModelName, type Quality } from './models';
import {
  containsScript, lineLanguage, normalizeLanguages, recognizerPlan, scriptTag,
  validateProfileLanguages, type ScriptKind,
} from './languages';
import { distance, extractRegions, normalizeAngle, quadBounds, sortReadingOrder } from './geometry';
import type { Candidate, OcrResult, ProgressEvent, Region } from './types';

const RECOGNITION_HEIGHT = 48;
const BASE_RECOGNITION_WIDTH = 320;
const MAX_RECOGNITION_WIDTH = 3200;
const DEFAULT_MIN_CONFIDENCE = 0.80;
const ARTIFACT_INFLIGHT = new Map<string, Promise<ArrayBuffer>>();

type ProgressSink = (event: ProgressEvent) => void;
type BrowserImage = ImageBitmap | OffscreenCanvas | HTMLImageElement;

interface RecognizerEntry {
  session: ort.InferenceSession;
  dictionary: string[];
  provider: string;
}

export type RuntimeMode = 'auto' | 'wasm' | 'webgpu';

interface WebOcrOptions {
  quality?: Quality;
  languages?: string[] | string;
  language?: string;
  runtime?: RuntimeMode;
  minConfidence?: number;
}

export class WebOcr {
  readonly quality: Quality;
  readonly languages: string[];
  readonly autoLanguages: boolean;
  readonly runtime: RuntimeMode;
  readonly minConfidence: number;
  private detector?: ort.InferenceSession;
  private detectorProvider = 'not loaded';
  private recognizers = new Map<ModelName, RecognizerEntry>();
  private sessionProviders = new Map<string, string>();
  private initTail: Promise<unknown> = Promise.resolve();
  provider = 'not loaded';
  wasmThreads = 1;

  constructor({ quality = 'balanced', languages, language, runtime = 'auto', minConfidence = DEFAULT_MIN_CONFIDENCE }: WebOcrOptions = {}) {
    if (!QUALITY[quality]) throw new Error(`Unsupported quality profile: ${quality}`);
    this.quality = quality;
    this.languages = normalizeLanguages(languages ?? (language ? [language] : []));
    validateProfileLanguages(quality, this.languages);
    this.autoLanguages = this.languages.length === 0;
    this.runtime = runtime;
    this.minConfidence = clamp(minConfidence, 0, 1);
  }

  async load(progress: ProgressSink = () => {}): Promise<void> {
    this.wasmThreads = configureWasm();
    const detectorName = detectorFor(this.quality);
    const initialRecognizer = mainRecognizer(this.quality, recognizerPlan(this.quality, this.languages));
    const modelNames = [...new Set<ModelName>([
      detectorName,
      initialRecognizer,
      'v5-latin-rec',
      'v5-eslav-rec',
      'v5-korean-rec',
    ])];

    progress({ phase: 'initializing', message: 'Preparing models…', progress: 0.02 });

    // Downloads are intentionally parallel. Session creation remains serialized below,
    // because compiling several WebGPU graphs concurrently can stall Chromium.
    const prefetch = this.prefetchArtifacts(modelNames, progress);
    const detectorTask = this.ensureDetector(detectorName, progress);
    const recognizerTask = this.ensureRecognizer(initialRecognizer, progress);

    await prefetch;
    progress({ phase: 'initializing', message: 'Starting OCR runtime…', progress: 0.92 });
    await Promise.all([detectorTask, recognizerTask]);

    this.provider = providerLabel(this.sessionProviders.values());
    progress({ phase: 'ready', message: 'Ready', progress: 1 });
  }

  async recognize(image: BrowserImage, progress: ProgressSink = () => {}): Promise<OcrResult> {
    if (!this.detector) await this.load(progress);

    const started = performance.now();
    const profile = QUALITY[this.quality];
    const detectorSide = detectorInputLimit(this.quality, this.detectorProvider);
    progress({ phase: 'detecting', message: `Preparing image · max ${detectorSide}px`, progress: 0.05 });
    const preprocessStarted = performance.now();
    const detectorInput = await detectionTensor(image, detectorSide);
    const preprocessMs = performance.now() - preprocessStarted;

    progress({
      phase: 'detecting',
      message: `Detecting text · ${detectorInput.width}×${detectorInput.height}`,
      progress: 0.16,
    });
    const inferenceStarted = performance.now();
    const detectionOutput = await this.detector!.run({
      [this.detector!.inputNames[0]]: detectorInput.tensor,
    });
    const detectorInferenceMs = performance.now() - inferenceStarted;
    const heatmap = detectionOutput[this.detector!.outputNames[0]];
    if (!heatmap || heatmap.data.length === 0 || typeof heatmap.data[0] !== 'number') {
      throw new Error('Glypho detector returned a non-numeric tensor.');
    }
    const numericHeatmap = {
      dims: heatmap.dims,
      data: heatmap.data as ArrayLike<number>,
    };

    progress({ phase: 'detecting', message: 'Tracing DB text contours…', progress: 0.42 });
    const postprocessStarted = performance.now();
    const regions = await extractRegions(numericHeatmap, image.width, image.height, profile);
    const postprocessMs = performance.now() - postprocessStarted;

    if (regions.length === 0) {
      const elapsedMs = performance.now() - started;
      return {
        text: '', lines: [], provider: this.provider, wasmThreads: this.wasmThreads,
        quality: this.quality, languages: this.languages,
        elapsedMs, preprocessMs, detectorInferenceMs, postprocessMs, recognitionMs: 0,
      };
    }

    progress({
      phase: 'recognizing',
      message: `Recognizing ${regions.length} text region${regions.length === 1 ? '' : 's'}…`,
      progress: 0.52,
    });
    const recognitionStarted = performance.now();
    const candidates = await this.recognizeRegions(image, regions, progress);
    const recognitionMs = performance.now() - recognitionStarted;

    const lines = [];
    for (let index = 0; index < regions.length; index += 1) {
      const candidate = candidates[index];
      if (!candidate?.text.trim() || candidate.confidence < this.minConfidence) continue;
      const region = regions[index];
      lines.push({
        ...region,
        text: candidate.text,
        confidence: candidate.confidence,
        alternative: candidate.alternative,
        script: scriptTag(candidate.text),
        language: lineLanguage(candidate.text, this.languages),
      });
    }

    const ordered = sortReadingOrder(lines);
    const elapsedMs = performance.now() - started;
    progress({ phase: 'done', message: `Finished · ${ordered.length} regions`, progress: 1 });
    return {
      text: ordered.map((line) => line.text.trim()).filter(Boolean).join('\n'),
      lines: ordered,
      provider: this.provider,
      wasmThreads: this.wasmThreads,
      quality: this.quality,
      languages: this.languages,
      elapsedMs,
      preprocessMs,
      detectorInferenceMs,
      postprocessMs,
      recognitionMs,
    };
  }

  async close(): Promise<void> {
    const sessions = [this.detector, ...[...this.recognizers.values()].map((entry) => entry.session)].filter(Boolean) as ort.InferenceSession[];
    await Promise.allSettled(sessions.map((session) => session.release()));
    this.detector = undefined;
    this.recognizers.clear();
    this.sessionProviders.clear();
    this.provider = 'not loaded';
    this.detectorProvider = 'not loaded';
  }

  private async prefetchArtifacts(models: ModelName[], progress: ProgressSink): Promise<void> {
    const artifacts: Array<{
      model: ModelName;
      artifactName: 'inference.onnx' | 'inference.yml';
      artifact: Artifact;
    }> = [];

    for (const model of models) {
      for (const artifactName of ['inference.onnx', 'inference.yml'] as const) {
        const artifact = MODELS[model].artifacts[artifactName];
        if (artifact) artifacts.push({ model, artifactName, artifact });
      }
    }

    const totalBytes = artifacts.reduce((sum, item) => sum + item.artifact.bytes, 0);
    const loaded = new Map<string, number>();

    const updateOverall = (key: string, bytes: number) => {
      loaded.set(key, Math.max(loaded.get(key) ?? 0, bytes));
      const loadedBytes = [...loaded.values()].reduce((sum, value) => sum + value, 0);
      progress({
        phase: 'downloading',
        message: 'Downloading models…',
        progress: totalBytes > 0 ? Math.min(0.9, (loadedBytes / totalBytes) * 0.9) : 0.9,
        loadedBytes,
        totalBytes,
      });
    };

    await Promise.all(artifacts.map(async ({ model, artifactName, artifact }) => {
      const key = `${model}:${artifactName}`;
      await fetchArtifact(MODELS[model], artifactName, model, (event) => {
        progress(event);
        if (typeof event.loadedBytes === 'number') {
          updateOverall(key, Math.min(event.loadedBytes, artifact.bytes));
        }
      });
      updateOverall(key, artifact.bytes);
    }));
  }

  private async ensureDetector(name: ModelName, progress: ProgressSink): Promise<ort.InferenceSession> {
    if (this.detector) return this.detector;
    const bytes = await fetchArtifact(MODELS[name], 'inference.onnx', name, progress);
    progress({ phase: 'initializing', model: name, message: `Initializing ${MODELS[name].shortLabel}…`, progress: 0.88 });
    const created = await this.serializedSessionCreate(bytes, false);
    this.detector = created.session;
    this.detectorProvider = created.provider;
    this.sessionProviders.set(`det:${name}`, created.provider);
    progress({ phase: 'ready', model: name, message: `${MODELS[name].shortLabel} ready`, progress: 1 });
    return this.detector;
  }

  private async ensureRecognizer(name: ModelName, progress: ProgressSink): Promise<RecognizerEntry> {
    const existing = this.recognizers.get(name);
    if (existing) return existing;

    const [modelBytes, configBytes] = await Promise.all([
      fetchArtifact(MODELS[name], 'inference.onnx', name, progress),
      fetchArtifact(MODELS[name], 'inference.yml', name, progress),
    ]);
    progress({ phase: 'initializing', model: name, message: `Initializing ${MODELS[name].shortLabel}…`, progress: 0.92 });
    const specialist = name === 'v5-latin-rec' || name === 'v5-eslav-rec' || name === 'v5-korean-rec';
    const created = await this.serializedSessionCreate(modelBytes, specialist);
    const entry: RecognizerEntry = {
      session: created.session,
      dictionary: parseDictionary(new TextDecoder().decode(configBytes)),
      provider: created.provider,
    };
    this.recognizers.set(name, entry);
    this.sessionProviders.set(`rec:${name}`, created.provider);
    this.provider = providerLabel(this.sessionProviders.values());
    progress({ phase: 'ready', model: name, message: `${MODELS[name].shortLabel} ready`, progress: 1 });
    return entry;
  }

  private serializedSessionCreate(bytes: ArrayBuffer, forceWasm: boolean): Promise<{ session: ort.InferenceSession; provider: string }> {
    const task = this.initTail.then(async () => {
      await yieldToBrowser();
      return createSession(bytes, forceWasm, this.runtime, this.quality);
    });
    this.initTail = task.then(() => undefined, () => undefined);
    return task;
  }

  private async recognizeRegions(image: BrowserImage, regions: Region[], progress: ProgressSink): Promise<Array<Candidate | undefined>> {
    const plan = recognizerPlan(this.quality, this.languages);
    const primaryName = mainRecognizer(this.quality, plan);
    const primary = await this.recognizeWith(primaryName, image, regions, undefined, progress, 0.52, 0.68);
    const selected = primary.map((candidate) => candidate && ({ ...candidate }));

    if (this.autoLanguages) {
      // Smart auto mode: the universal recognizer is authoritative when it is already
      // very confident. Specialists only run for ambiguous regions or when the primary
      // output exposes their script. This keeps an ordinary Latin photo from compiling
      // the Korean WebGPU graph for no benefit, while explicit `ko`, `ru`, etc. still
      // force the corresponding pack.
      const stages: Array<[ModelName, ScriptKind, number, number]> = [
        ['v5-latin-rec', 'latin', 0.68, 0.78],
        ['v5-eslav-rec', 'cyrillic', 0.78, 0.89],
        ['v5-korean-rec', 'korean', 0.89, 0.98],
      ];
      for (const [name, script, from, to] of stages) {
        const indices = autoSpecialistCropIndices(primary, regions.length, script);
        if (!indices.length) continue;
        const specialist = await this.recognizeWith(name, image, regions, indices, progress, from, to);
        mergeSpecialist(selected, specialist, script);
      }
      return selected;
    }

    if (plan.primary && plan.latin) {
      const latin = await this.recognizeWith('v5-latin-rec', image, regions, undefined, progress, 0.68, 0.78);
      mergeSpecialist(selected, latin, 'latin');
    }
    if (plan.cyrillic && primaryName !== 'v5-eslav-rec') {
      const indices = specialistCropIndices(primary, regions.length, 'cyrillic');
      if (indices.length) {
        const cyrillic = await this.recognizeWith('v5-eslav-rec', image, regions, indices, progress, 0.78, 0.90);
        mergeSpecialist(selected, cyrillic, 'cyrillic');
      }
    }
    if (plan.korean && primaryName !== 'v5-korean-rec') {
      const korean = await this.recognizeWith('v5-korean-rec', image, regions, undefined, progress, 0.90, 0.98);
      mergeSpecialist(selected, korean, 'korean');
    }
    return selected;
  }

  private async recognizeWith(
    name: ModelName,
    image: BrowserImage,
    regions: Region[],
    indices: number[] | undefined,
    progress: ProgressSink,
    progressFrom: number,
    progressTo: number,
  ): Promise<Array<Candidate | undefined>> {
    const recognizer = await this.ensureRecognizer(name, progress);
    const requested = indices ?? regions.map((_, index) => index);
    if (requested.length === 0) return new Array(regions.length);
    const batches = makeRecognitionBatches(regions, requested, recognitionBatchProfile(QUALITY[this.quality], recognizer.provider));
    const candidates: Array<Candidate | undefined> = new Array(regions.length);
    let completed = 0;

    for (const batch of batches) {
      const input = recognitionTensor(image, regions, batch);
      const output = await recognizer.session.run({ [recognizer.session.inputNames[0]]: input });
      const decoded = decodeBatch(output[recognizer.session.outputNames[0]], recognizer.dictionary);
      for (let offset = 0; offset < batch.length; offset += 1) {
        const candidate = decoded[offset];
        if (candidate?.text.trim()) candidates[batch[offset]] = candidate;
      }
      completed += batch.length;
      const fraction = completed / requested.length;
      progress({
        phase: 'recognizing', model: name,
        message: `${MODELS[name].shortLabel} · ${Math.min(completed, requested.length)}/${requested.length}`,
        progress: progressFrom + (progressTo - progressFrom) * fraction,
      });
      await yieldToBrowser();
    }
    return candidates;
  }
}

function mainRecognizer(quality: Quality, plan: ReturnType<typeof recognizerPlan>): ModelName {
  if (plan.primary) return primaryRecognizerFor(quality);
  if (plan.latin) return 'v5-latin-rec';
  if (plan.cyrillic) return 'v5-eslav-rec';
  if (plan.korean) return 'v5-korean-rec';
  return primaryRecognizerFor(quality);
}

export async function fetchArtifact(
  modelEntry,
  name,
  modelName: ModelName,
  progress: ProgressSink = () => {},
): Promise<ArrayBuffer> {
  const artifact = modelEntry.artifacts?.[name] ?? modelEntry[name];
  if (!artifact) throw new Error(`Model artifact is unavailable: ${name}`);
  const existing = ARTIFACT_INFLIGHT.get(artifact.url);
  if (existing) return existing;

  const request = fetchArtifactInner(artifact, modelName, progress);
  ARTIFACT_INFLIGHT.set(artifact.url, request);
  try {
    return await request;
  } finally {
    ARTIFACT_INFLIGHT.delete(artifact.url);
  }
}

async function fetchArtifactInner(artifact, modelName: ModelName, progress: ProgressSink): Promise<ArrayBuffer> {
  const cache = 'caches' in globalThis ? await caches.open('glypho-models-v3') : undefined;
  const cached = await cache?.match(artifact.url);
  if (cached) {
    const bytes = await cached.arrayBuffer();
    if (await verified(bytes, artifact)) {
      progress({
        phase: 'downloading', model: modelName, cached: true,
        loadedBytes: artifact.bytes, totalBytes: artifact.bytes, progress: 1,
        message: `${MODELS[modelName].shortLabel} · cached`,
      });
      return bytes;
    }
    await cache?.delete(artifact.url);
  }

  progress({
    phase: 'downloading', model: modelName, loadedBytes: 0, totalBytes: artifact.bytes,
    progress: 0, message: `Downloading ${MODELS[modelName].shortLabel}…`,
  });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);
  let bytes: ArrayBuffer;
  try {
    const response = await fetch(artifact.url, {
      mode: 'cors', credentials: 'omit', signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Model download failed with HTTP ${response.status}`);
    bytes = await readBounded(response, artifact.bytes, (loaded) => {
      progress({
        phase: 'downloading', model: modelName, loadedBytes: loaded, totalBytes: artifact.bytes,
        progress: Math.min(1, loaded / artifact.bytes),
        message: `Downloading ${MODELS[modelName].shortLabel}…`,
      });
    });
  } finally {
    clearTimeout(timeout);
  }

  if (!(await verified(bytes, artifact))) throw new Error('Downloaded model failed SHA-256 verification');
  await cache?.put(artifact.url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
  progress({
    phase: 'downloading', model: modelName, cached: true,
    loadedBytes: artifact.bytes, totalBytes: artifact.bytes, progress: 1,
    message: `${MODELS[modelName].shortLabel} · verified`,
  });
  return bytes;
}

export async function readBounded(
  response: Response,
  expectedBytes: number,
  onProgress: (loaded: number) => void = () => {},
): Promise<ArrayBuffer> {
  const contentLength = Number(response.headers.get('Content-Length'));
  if (Number.isFinite(contentLength) && contentLength > 0 && contentLength !== expectedBytes) {
    throw new Error('Model download has an unexpected size');
  }
  if (!response.body) {
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== expectedBytes) throw new Error('Model download has an unexpected size');
    onProgress(bytes.byteLength);
    return bytes;
  }

  const reader = response.body.getReader();
  const result = new Uint8Array(expectedBytes);
  let offset = 0;
  let lastReported = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.byteLength > expectedBytes) {
      await reader.cancel();
      throw new Error('Model download exceeded its registered size');
    }
    result.set(value, offset);
    offset += value.byteLength;
    // Avoid flooding React with a progress event for every tiny network chunk.
    if (offset - lastReported >= 256 * 1024 || offset === expectedBytes) {
      onProgress(offset);
      lastReported = offset;
    }
  }
  if (offset !== expectedBytes) throw new Error('Model download has an unexpected size');
  onProgress(offset);
  return result.buffer;
}

async function verified(bytes: ArrayBuffer, artifact): Promise<boolean> {
  if (bytes.byteLength !== artifact.bytes) return false;
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hex = [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('');
  return hex === artifact.sha256;
}

function configureWasm() {
  // ORT labels harmless graph-cleanup diagnostics as warnings. Firefox DevTools can
  // render the WASM logger output in red, which makes them look like inference failures.
  // Keep real errors visible while hiding routine optimizer noise.
  ort.env.debug = false;
  ort.env.logLevel = 'error';

  const threads = globalThis.crossOriginIsolated
    ? Math.min(4, globalThis.navigator?.hardwareConcurrency || 1)
    : 1;
  ort.env.wasm.numThreads = Math.max(1, threads);
  ort.env.wasm.simd = true;

  // Do not let the bundler/worker context guess where ORT's runtime lives.
  // Vite emits these two assets with the production build and ORT loads them
  // from the same origin. This keeps threaded WASM reliable behind CSP/COEP.
  const origin = globalThis.location?.origin;
  if (origin) {
    ort.env.wasm.wasmPaths = {
      wasm: new URL(ortWasmUrl, origin).href,
      mjs: new URL(ortWasmModuleUrl, origin).href,
    };
  }

  // Never leave the UI stuck on "Starting OCR runtime…" forever if a browser
  // or deployment policy prevents ORT's WASM runtime from initializing.
  ort.env.wasm.initTimeout = 20_000;
  return Math.max(1, threads);
}

async function createSession(
  bytes: ArrayBuffer,
  forceWasm = false,
  runtime: RuntimeMode = 'auto',
  quality: Quality = 'balanced',
): Promise<{ session: ort.InferenceSession; provider: string }> {
  const preferWebGpu = !forceWasm && (
    runtime === 'webgpu' || (runtime === 'auto' && (quality === 'accurate' || quality === 'maximum'))
  );
  if (preferWebGpu && globalThis.navigator && 'gpu' in globalThis.navigator) {
    try {
      return {
        session: await ort.InferenceSession.create(bytes, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
          enableMemPattern: true,
        }),
        provider: 'WebGPU',
      };
    } catch (error) {
      if (runtime === 'webgpu') console.warn('Glypho WebGPU initialization failed; using WASM.', error);
      else console.info('Glypho WebGPU unavailable; using threaded WASM.', error);
    }
  }

  return {
    session: await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableMemPattern: true,
    }),
    provider: 'WASM',
  };
}

function providerLabel(values: Iterable<string>): string {
  const providers = new Set(values);
  if (providers.size === 0) return 'not loaded';
  if (providers.size === 1) return [...providers][0];
  return [...providers].join(' + ');
}

export function detectorInputLimit(quality: Quality, _provider: string): number {
  const configured = QUALITY[quality]?.maxSide;
  if (!configured) throw new Error(`Unsupported quality profile: ${quality}`);
  // Keep the same detector resolution as native Glypho. The OCR worker prevents a
  // slow WASM fallback from freezing the page; silently reducing resolution hurt
  // small-text recall and made Web results diverge from the Rust runtime.
  return configured;
}

async function detectionTensor(image, limit) {
  const scale = Math.min(1, limit / Math.max(image.width, image.height));
  const width = Math.max(32, Math.round((image.width * scale) / 32) * 32);
  const height = Math.max(32, Math.round((image.height * scale) / 32) * 32);
  const canvas = createCanvas(width, height);
  const context = canvas.getContext('2d', { willReadFrequently: true, alpha: false });
  context.drawImage(image, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height).data;
  const plane = width * height;
  const data = new Float32Array(plane * 3);

  for (let index = 0; index < plane; index += 1) {
    const pixel = index * 4;
    // PaddleOCR ONNX preprocessing is BGR + ImageNet mean/std.
    data[index] = (pixels[pixel + 2] / 255 - 0.485) / 0.229;
    data[plane + index] = (pixels[pixel + 1] / 255 - 0.456) / 0.224;
    data[plane * 2 + index] = (pixels[pixel] / 255 - 0.406) / 0.225;
    if ((index & 0x3ffff) === 0x3ffff) await yieldToBrowser();
  }
  return { tensor: new ort.Tensor('float32', data, [1, 3, height, width]), width, height };
}


function recognitionBatchProfile(profile, provider) {
  if (provider !== 'WebGPU') return profile;
  return {
    ...profile,
    batchSize: profile.batchSize === 16 ? 32 : 24,
    widthBudget: Math.max(profile.widthBudget, profile.batchSize === 16 ? 24_576 : 16_384),
  };
}

export function makeRecognitionBatches(regions, requestedIndices, profile = QUALITY.balanced) {
  const items = requestedIndices
    .map((index) => ({ index, width: targetRecognitionWidth(regions[index]) }))
    .sort((left, right) => left.width - right.width);
  const batches = [];
  let batch = [];
  let maxWidth = 0;

  for (const item of items) {
    const nextMaxWidth = Math.max(maxWidth, item.width);
    const nextCount = batch.length + 1;
    const exceedsCount = nextCount > profile.batchSize;
    const exceedsBudget = nextMaxWidth * nextCount > profile.widthBudget;
    // Avoid putting a very wide crop in a batch of narrow crops; padded pixels are pure waste.
    const shapeMismatch = batch.length >= 2 && item.width > Math.max(256, maxWidth * 1.85);
    if (batch.length && (exceedsCount || exceedsBudget || shapeMismatch)) {
      batches.push(batch);
      batch = [];
      maxWidth = 0;
    }
    batch.push(item.index);
    maxWidth = Math.max(maxWidth, item.width);
  }
  if (batch.length) batches.push(batch);
  return batches;
}

function targetRecognitionContentWidth(region) {
  const ratio = region.cropWidth / Math.max(1, region.cropHeight);
  return Math.min(
    MAX_RECOGNITION_WIDTH,
    Math.max(1, Math.ceil(RECOGNITION_HEIGHT * ratio)),
  );
}

function targetRecognitionWidth(region) {
  const contentWidth = targetRecognitionContentWidth(region);
  return Math.min(
    MAX_RECOGNITION_WIDTH,
    Math.max(BASE_RECOGNITION_WIDTH, Math.ceil(contentWidth / 8) * 8),
  );
}

function recognitionTensor(image, regions, indices) {
  // Match OAR/Paddle recognition preprocessing: [3,48,320] is the minimum
  // tensor shape; wider crops expand the dynamic batch width. Zero padding is
  // intentional because Paddle pads the already-normalized tensor with 0.0.
  const width = Math.max(
    BASE_RECOGNITION_WIDTH,
    ...indices.map((index) => targetRecognitionWidth(regions[index])),
  );
  const plane = width * RECOGNITION_HEIGHT;
  const data = new Float32Array(indices.length * plane * 3);

  indices.forEach((regionIndex, batch) => {
    const region = regions[regionIndex];
    const resizedWidth = Math.min(width, targetRecognitionContentWidth(region));
    writePerspectiveCrop(data, batch, plane, width, image, region, resizedWidth);
  });

  return new ort.Tensor('float32', data, [indices.length, 3, RECOGNITION_HEIGHT, width]);
}

function drawRotatedCrop(context, image, region, destinationWidth, destinationHeight) {
  const angle = Math.abs(region.angle) < Math.PI / 360 ? 0 : region.angle;
  if (angle === 0) {
    context.drawImage(
      image,
      region.x, region.y, region.width, region.height,
      0, 0, destinationWidth, destinationHeight,
    );
    return;
  }

  const scaleX = destinationWidth / Math.max(1, region.cropWidth);
  const scaleY = destinationHeight / Math.max(1, region.cropHeight);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const a = cos * scaleX;
  const c = sin * scaleX;
  const b = -sin * scaleY;
  const d = cos * scaleY;
  const e = destinationWidth / 2 - a * region.centerX - c * region.centerY;
  const f = destinationHeight / 2 - b * region.centerX - d * region.centerY;
  context.setTransform(a, b, c, d, e, f);
  // Sample only the axis-aligned bounds that contain the rotated quad. This avoids
  // asking the canvas backend to transform a multi-megapixel source for every crop.
  context.drawImage(
    image,
    region.x, region.y, region.width, region.height,
    region.x, region.y, region.width, region.height,
  );
  context.setTransform(1, 0, 0, 1, 0, 0);
}

function writeRgbaCrop(data, batch, plane, tensorWidth, pixels, cropWidth) {
  for (let y = 0; y < RECOGNITION_HEIGHT; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      const source = (y * cropWidth + x) * 4;
      const target = y * tensorWidth + x;
      data[(batch * 3) * plane + target] = pixels[source + 2] / 127.5 - 1;
      data[(batch * 3 + 1) * plane + target] = pixels[source + 1] / 127.5 - 1;
      data[(batch * 3 + 2) * plane + target] = pixels[source] / 127.5 - 1;
    }
  }
}

export function perspectiveSeverity(quad) {
  if (!quad || quad.length !== 4) return 0;
  const top = distance(quad[0], quad[1]);
  const bottom = distance(quad[3], quad[2]);
  const left = distance(quad[0], quad[3]);
  const right = distance(quad[1], quad[2]);
  if (Math.min(top, bottom, left, right) <= 1e-6) return 0;

  const widthTaper = Math.abs(top - bottom) / Math.max(top, bottom);
  const heightTaper = Math.abs(left - right) / Math.max(left, right);
  const topAngle = Math.atan2(quad[1].y - quad[0].y, quad[1].x - quad[0].x);
  const bottomAngle = Math.atan2(quad[2].y - quad[3].y, quad[2].x - quad[3].x);
  const leftAngle = Math.atan2(quad[3].y - quad[0].y, quad[3].x - quad[0].x);
  const rightAngle = Math.atan2(quad[2].y - quad[1].y, quad[2].x - quad[1].x);
  const horizontalSkew = normalizedParallelAngleDelta(topAngle, bottomAngle) / (Math.PI / 4);
  const verticalSkew = normalizedParallelAngleDelta(leftAngle, rightAngle) / (Math.PI / 4);
  return Math.max(widthTaper, heightTaper, horizontalSkew, verticalSkew);
}

function normalizedParallelAngleDelta(a, b) {
  let delta = Math.abs(a - b) % Math.PI;
  if (delta > Math.PI / 2) delta = Math.PI - delta;
  return delta;
}

function writePerspectiveCrop(data, batch, plane, tensorWidth, image, region, destinationWidth) {
  const quad = region.quad;
  const bounds = quadBounds(quad);
  const sourceX = Math.max(0, Math.floor(bounds.x));
  const sourceY = Math.max(0, Math.floor(bounds.y));
  const sourceRight = Math.min(image.width, Math.ceil(bounds.x + bounds.width));
  const sourceBottom = Math.min(image.height, Math.ceil(bounds.y + bounds.height));
  const sourceWidth = Math.max(1, sourceRight - sourceX);
  const sourceHeight = Math.max(1, sourceBottom - sourceY);

  // Bound temporary RGBA memory. Large photographs can otherwise allocate tens of MB
  // for a single perspective crop even though the recognizer consumes only 48px high.
  const maxSourcePixels = 2_000_000;
  const sampleScale = Math.min(1, Math.sqrt(maxSourcePixels / (sourceWidth * sourceHeight)));
  const sampledWidth = Math.max(1, Math.ceil(sourceWidth * sampleScale));
  const sampledHeight = Math.max(1, Math.ceil(sourceHeight * sampleScale));
  const sourceCanvas = createCanvas(sampledWidth, sampledHeight);
  const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
  sourceContext.drawImage(
    image,
    sourceX, sourceY, sourceWidth, sourceHeight,
    0, 0, sampledWidth, sampledHeight,
  );
  const pixels = sourceContext.getImageData(0, 0, sampledWidth, sampledHeight).data;

  for (let y = 0; y < RECOGNITION_HEIGHT; y += 1) {
    const t = (y + 0.5) / RECOGNITION_HEIGHT;
    for (let x = 0; x < destinationWidth; x += 1) {
      const u = (x + 0.5) / destinationWidth;
      const point = projectivePoint(quad, u, t);
      const localX = (point.x - sourceX) * sampleScale - 0.5;
      const localY = (point.y - sourceY) * sampleScale - 0.5;
      const [red, green, blue] = bicubicRgb(pixels, sampledWidth, sampledHeight, localX, localY);
      const target = y * tensorWidth + x;
      data[(batch * 3) * plane + target] = blue / 127.5 - 1;
      data[(batch * 3 + 1) * plane + target] = green / 127.5 - 1;
      data[(batch * 3 + 2) * plane + target] = red / 127.5 - 1;
    }
  }
}

function projectivePoint(quad, u, v) {
  const [p0, p1, p2, p3] = quad;
  const dx1 = p1.x - p2.x;
  const dx2 = p3.x - p2.x;
  const dx3 = p0.x - p1.x + p2.x - p3.x;
  const dy1 = p1.y - p2.y;
  const dy2 = p3.y - p2.y;
  const dy3 = p0.y - p1.y + p2.y - p3.y;
  let g = 0;
  let h = 0;
  const denominator = dx1 * dy2 - dx2 * dy1;
  if ((Math.abs(dx3) > 1e-7 || Math.abs(dy3) > 1e-7) && Math.abs(denominator) > 1e-9) {
    g = (dx3 * dy2 - dx2 * dy3) / denominator;
    h = (dx1 * dy3 - dx3 * dy1) / denominator;
  }
  const a = p1.x - p0.x + g * p1.x;
  const b = p3.x - p0.x + h * p3.x;
  const c = p0.x;
  const d = p1.y - p0.y + g * p1.y;
  const e = p3.y - p0.y + h * p3.y;
  const f = p0.y;
  const z = g * u + h * v + 1;
  return { x: (a * u + b * v + c) / z, y: (d * u + e * v + f) / z };
}

function cubicWeight(value) {
  const x = Math.abs(value);
  if (x <= 1) return 1.5 * x * x * x - 2.5 * x * x + 1;
  if (x < 2) return -0.5 * x * x * x + 2.5 * x * x - 4 * x + 2;
  return 0;
}

function bicubicRgb(pixels, width, height, x, y) {
  const baseX = Math.floor(x);
  const baseY = Math.floor(y);
  const result = [0, 0, 0];
  let weightSum = 0;
  for (let oy = -1; oy <= 2; oy += 1) {
    const sy = clamp(baseY + oy, 0, height - 1);
    const wy = cubicWeight(y - (baseY + oy));
    for (let ox = -1; ox <= 2; ox += 1) {
      const sx = clamp(baseX + ox, 0, width - 1);
      const weight = wy * cubicWeight(x - (baseX + ox));
      const offset = (sy * width + sx) * 4;
      result[0] += pixels[offset] * weight;
      result[1] += pixels[offset + 1] * weight;
      result[2] += pixels[offset + 2] * weight;
      weightSum += weight;
    }
  }
  if (Math.abs(weightSum) > 1e-8) {
    result[0] /= weightSum;
    result[1] /= weightSum;
    result[2] /= weightSum;
  }
  return result.map((value) => clamp(value, 0, 255));
}

function decodeBatch(output, dictionary) {
  const [batch, time, vocabulary] = output.dims;
  const characters = ['\0', ...dictionary, ' '];
  const decoded = [];
  for (let item = 0; item < batch; item += 1) {
    let previous = -1;
    let text = '';
    let confidence = 0;
    let count = 0;
    for (let step = 0; step < time; step += 1) {
      const offset = (item * time + step) * vocabulary;
      let token = 0;
      let score = output.data[offset];
      for (let index = 1; index < vocabulary; index += 1) {
        if (output.data[offset + index] >= score) {
          token = index;
          score = output.data[offset + index];
        }
      }
      if (token !== 0 && token !== previous && characters[token]) {
        text += characters[token];
        confidence += clamp(score, 0, 1);
        count += 1;
      }
      previous = token;
    }
    decoded.push({ text, confidence: count ? confidence / count : 0 });
  }
  return decoded;
}

function autoSpecialistCropIndices(primary, cropCount, script: ScriptKind) {
  const indices = [];
  for (let index = 0; index < cropCount; index += 1) {
    const candidate = primary[index];
    if (!candidate || candidate.confidence < 0.90 || containsScript(candidate.text, script)) {
      indices.push(index);
    }
  }
  return indices;
}

function specialistCropIndices(primary, cropCount, script: ScriptKind = 'cyrillic') {
  const indices = [];
  for (let index = 0; index < cropCount; index += 1) {
    const candidate = primary[index];
    if (!candidate || candidate.confidence < 0.82 || containsScript(candidate.text, script)) indices.push(index);
  }
  return indices;
}

function mergeSpecialist(selected, specialists, script: ScriptKind) {
  for (let index = 0; index < selected.length; index += 1) {
    const specialist = specialists[index];
    if (!specialist) continue;
    const primary = selected[index];
    if (!primary) {
      selected[index] = { ...specialist };
      continue;
    }
    selected[index] = selectCandidate(primary, specialist, script);
  }
}

export function selectCandidate(primary, specialist, script: ScriptKind) {
  const containsExpectedScript = containsScript(specialist.text, script);
  const useSpecialist = containsExpectedScript
    ? specialist.confidence >= primary.confidence - 0.18
    : specialist.confidence > primary.confidence + 0.08;
  if (useSpecialist) {
    return {
      ...specialist,
      alternative: { text: primary.text, confidence: primary.confidence },
    };
  }
  return {
    ...primary,
    alternative: { text: specialist.text, confidence: specialist.confidence },
  };
}

function parseDictionary(config) {
  const lines = config.split(/\r?\n/);
  const start = lines.indexOf('  character_dict:');
  if (start < 0) throw new Error('The model dictionary is missing');
  const characters = [];
  for (const line of lines.slice(start + 1)) {
    if (!line.startsWith('  - ')) break;
    const value = line.slice(4);
    if (value.startsWith("'") && value.endsWith("'")) {
      characters.push(value.slice(1, -1).replaceAll("''", "'"));
    } else if (value.startsWith('"') && value.endsWith('"')) {
      characters.push(JSON.parse(value));
    } else {
      characters.push(value);
    }
  }
  return characters;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function createCanvas(width, height) {
  if ('OffscreenCanvas' in globalThis) return new OffscreenCanvas(width, height);
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  throw new Error('OffscreenCanvas is required for Glypho Web workers');
}

async function yieldToBrowser() {
  if (globalThis.scheduler?.yield) {
    await globalThis.scheduler.yield();
    return;
  }
  await new Promise((resolve) => setTimeout(resolve, 0));
}