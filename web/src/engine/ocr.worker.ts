/// <reference lib="webworker" />

import { WebOcr, type RuntimeMode } from './ocr';
import type { Quality } from './models';
import type { ProgressEvent } from './types';

type ConfigureMessage = {
  type: 'configure';
  id: number;
  quality: Quality;
  languages: string[];
  runtime: RuntimeMode;
};

type RecognizeMessage = {
  type: 'recognize';
  id: number;
  bitmap: ImageBitmap;
};

type DisposeMessage = { type: 'dispose' };
type IncomingMessage = ConfigureMessage | RecognizeMessage | DisposeMessage;

let engine: WebOcr | undefined;
let ready: Promise<void> = Promise.resolve();
let generation = 0;

function emitProgress(event: ProgressEvent): void {
  self.postMessage({ type: 'progress', event });
}

self.onmessage = async (message: MessageEvent<IncomingMessage>) => {
  const payload = message.data;

  if (payload.type === 'dispose') {
    generation += 1;
    const current = engine;
    engine = undefined;
    await current?.close();
    self.close();
    return;
  }

  if (payload.type === 'configure') {
    const myGeneration = ++generation;
    const previous = engine;
    engine = new WebOcr({ quality: payload.quality, languages: payload.languages, runtime: payload.runtime });
    await previous?.close();
    ready = engine.load((event) => {
      if (myGeneration === generation) emitProgress(event);
    });

    try {
      await ready;
      if (myGeneration !== generation) return;
      self.postMessage({
        type: 'configured',
        id: payload.id,
        provider: engine.provider,
        wasmThreads: engine.wasmThreads,
      });
    } catch (error) {
      if (myGeneration !== generation) return;
      self.postMessage({ type: 'error', id: payload.id, error: serializeError(error) });
    }
    return;
  }

  if (payload.type === 'recognize') {
    const active = engine;
    if (!active) {
      payload.bitmap.close();
      self.postMessage({ type: 'error', id: payload.id, error: 'Glypho Web is not configured yet.' });
      return;
    }

    try {
      await ready;
      const result = await active.recognize(payload.bitmap, emitProgress);
      self.postMessage({ type: 'result', id: payload.id, result });
    } catch (error) {
      self.postMessage({ type: 'error', id: payload.id, error: serializeError(error) });
    } finally {
      payload.bitmap.close();
    }
  }
};

function serializeError(error: unknown): string {
  if (error instanceof Error) {
    const label = `${error.name}: ${error.message}`;
    if (!error.stack) return label;
    return error.stack.includes(error.message) ? error.stack : `${label}\n${error.stack}`;
  }
  if (error && typeof error === 'object') {
    const value = error as { name?: unknown; message?: unknown; stack?: unknown };
    const name = typeof value.name === 'string' ? value.name : 'Error';
    const message = typeof value.message === 'string' ? value.message : String(error);
    const stack = typeof value.stack === 'string' ? value.stack : '';
    const label = `${name}: ${message}`;
    return stack && !stack.includes(message) ? `${label}\n${stack}` : (stack || label);
  }
  return String(error);
}
