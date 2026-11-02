import { MODELS, type Artifact, type ModelEntry, type ModelName } from './models';
import type { ProgressSink } from './types';

const ARTIFACT_INFLIGHT = new Map<string, Promise<ArrayBuffer>>();

export async function fetchArtifact(
  modelEntry: ModelEntry,
  name: 'inference.onnx' | 'inference.yml',
  modelName: ModelName,
  progress: ProgressSink = () => {},
): Promise<ArrayBuffer> {
  const artifact = modelEntry.artifacts[name];
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

async function fetchArtifactInner(artifact: Artifact, modelName: ModelName, progress: ProgressSink): Promise<ArrayBuffer> {
  const cache = await openModelCache();
  const cached = cache ? await readCached(cache, artifact) : undefined;
  if (cached) {
    progress({
      phase: 'downloading', model: modelName, cached: true,
      loadedBytes: artifact.bytes, totalBytes: artifact.bytes, progress: 1,
      message: `${MODELS[modelName].shortLabel} · cached`,
    });
    return cached;
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
  let stored = false;
  if (cache) {
    try {
      await cache.put(artifact.url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
      stored = true;
    } catch (error) {
      console.warn('Glypho model cache write failed; using verified bytes in memory.', error);
    }
  }
  progress({
    phase: 'downloading', model: modelName, cached: stored,
    loadedBytes: artifact.bytes, totalBytes: artifact.bytes, progress: 1,
    message: `${MODELS[modelName].shortLabel} · verified${stored ? '' : ' · memory only'}`,
  });
  return bytes;
}

async function openModelCache(): Promise<Cache | undefined> {
  try {
    return 'caches' in globalThis ? await caches.open('glypho-models-v3') : undefined;
  } catch (error) {
    console.warn('Glypho model cache is unavailable; using memory only.', error);
    return undefined;
  }
}

async function readCached(cache: Cache, artifact: Artifact): Promise<ArrayBuffer | undefined> {
  try {
    const response = await cache.match(artifact.url);
    if (!response) return undefined;
    const bytes = await readBounded(response, artifact.bytes);
    if (await verified(bytes, artifact)) return bytes;
  } catch (error) {
    console.warn('Glypho model cache read failed; downloading a verified copy.', error);
  }
  try {
    await cache.delete(artifact.url);
  } catch (error) {
    console.warn('Glypho could not remove a cached model.', error);
  }
  return undefined;
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

async function verified(bytes: ArrayBuffer, artifact: Artifact): Promise<boolean> {
  if (bytes.byteLength !== artifact.bytes) return false;
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hex = [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('');
  return hex === artifact.sha256;
}
