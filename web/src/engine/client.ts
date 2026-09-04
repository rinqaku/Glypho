import type { Quality } from './models';
import type { RuntimeMode } from './ocr';
import type { OcrResult, ProgressEvent } from './types';

const MAX_IMAGE_BYTES = 50 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 50_000_000;

type Pending = {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
};

export interface ConfigureResult {
  provider: string;
  wasmThreads: number;
}

export class GlyphoWebClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private listeners = new Set<(event: ProgressEvent) => void>();

  constructor() {
    this.worker = this.createWorker();
  }

  onProgress(listener: (event: ProgressEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  configure(quality: Quality, languages: string[], runtime: RuntimeMode = 'auto'): Promise<ConfigureResult> {
    const id = this.nextId++;
    const promise = this.request<ConfigureResult>(id);
    this.worker.postMessage({ type: 'configure', id, quality, languages, runtime });
    return promise;
  }

  async recognize(file: File): Promise<OcrResult> {
    validateFileSize(file);
    const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    try {
      validateBitmapSize(bitmap);
    } catch (error) {
      bitmap.close();
      throw error;
    }
    const id = this.nextId++;
    const promise = this.request<OcrResult>(id);
    this.worker.postMessage({ type: 'recognize', id, bitmap }, [bitmap]);
    return promise;
  }

  restart(): void {
    const old = this.worker;
    this.worker = this.createWorker();
    old.postMessage({ type: 'dispose' });
    old.terminate();
    for (const pending of this.pending.values()) {
      pending.reject(new Error('Glypho Web runtime restarted.'));
    }
    this.pending.clear();
  }

  dispose(): void {
    this.worker.postMessage({ type: 'dispose' });
    this.worker.terminate();
    for (const pending of this.pending.values()) pending.reject(new Error('Glypho Web runtime disposed.'));
    this.pending.clear();
    this.listeners.clear();
  }

  private request<T>(id: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
    });
  }

  private createWorker(): Worker {
    const worker = new Worker(new URL('./ocr.worker.ts', import.meta.url), { type: 'module', name: 'glypho-ocr' });
    worker.onmessage = (message: MessageEvent) => {
      const payload = message.data;
      if (payload.type === 'progress') {
        for (const listener of this.listeners) listener(payload.event);
        return;
      }
      if (payload.type === 'configured' || payload.type === 'result') {
        const pending = this.pending.get(payload.id);
        if (!pending) return;
        this.pending.delete(payload.id);
        pending.resolve(payload.type === 'result'
          ? payload.result
          : { provider: payload.provider, wasmThreads: payload.wasmThreads });
        return;
      }
      if (payload.type === 'error') {
        const pending = this.pending.get(payload.id);
        if (!pending) return;
        this.pending.delete(payload.id);
        pending.reject(new Error(payload.error));
      }
    };
    worker.onerror = (event) => {
      const error = new Error(event.message || 'Glypho Web worker crashed.');
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    };
    return worker;
  }
}

export async function inspectImage(file: File): Promise<{ width: number; height: number }> {
  validateFileSize(file);
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  try {
    validateBitmapSize(bitmap);
    return { width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}

function validateFileSize(file: File): void {
  if (file.size > MAX_IMAGE_BYTES) {
    throw new Error('Image is larger than the 50 MB safety limit.');
  }
}

function validateBitmapSize(bitmap: ImageBitmap): void {
  if (bitmap.width * bitmap.height > MAX_IMAGE_PIXELS) {
    throw new Error('Image exceeds the 50 megapixel safety limit.');
  }
}