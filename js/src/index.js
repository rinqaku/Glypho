import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);

export class GlyphoError extends Error {
  constructor(message, options) {
    super(message, options);
    this.name = 'GlyphoError';
  }
}

export class Glypho {
  constructor(options = {}) {
    this.binary = resolveBinary(options.binary);
    this.models = options.models ? fileURLToPathSafe(options.models) : undefined;
    this.quality = options.quality ?? 'balanced';
    this.device = options.device ?? 'auto';
    this.languages = [...(options.languages ?? [])];
    this.threads = options.threads;
    this.offline = options.offline ?? false;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    validateQuality(this.quality);
    validateDevice(this.device);
    validateThreads(this.threads);
    validateTimeout(this.timeoutMs);
    this.nextId = 1;
    this.pending = new Map();
    this.stderr = '';
    this.worker = undefined;
  }

  async recognize(image, options = {}) {
    const imagePath = path.resolve(fileURLToPathSafe(image));
    const imageStat = await stat(imagePath);
    if (!imageStat.isFile()) {
      throw new GlyphoError(`Input is not a file: ${imagePath}`);
    }
    const minConfidence = options.minConfidence ?? 0.8;
    if (!Number.isFinite(minConfidence) || minConfidence < 0 || minConfidence > 1) {
      throw new RangeError('minConfidence must be between 0 and 1');
    }
    const segmentation = options.segmentation ?? 'sparse_text';
    if (!['auto', 'single_block', 'single_line', 'sparse_text'].includes(segmentation)) {
      throw new TypeError(`Unsupported segmentation mode: ${segmentation}`);
    }
    const result = await this.request(
      {
        method: 'recognize',
        image: imagePath,
        languages: options.languages ?? this.languages,
        segmentation,
        min_confidence: minConfidence,
      },
      options,
    );
    if (result?.schema_version !== 'glypho.annotation.v1') {
      throw new GlyphoError(`Unsupported Glypho schema: ${result?.schema_version}`);
    }
    return result;
  }

  async warmup(options = {}) {
    return this.request(
      { method: 'warmup', languages: options.languages ?? this.languages },
      options,
    );
  }

  async info(options = {}) {
    return this.request({ method: 'info' }, options);
  }

  async close() {
    const worker = this.worker;
    this.worker = undefined;
    if (!worker || worker.exitCode !== null) {
      return;
    }
    worker.stdin.end();
    await new Promise((resolve) => worker.once('exit', resolve));
  }

  request(payload, options = {}) {
    const worker = this.startWorker();
    const id = this.nextId++;
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    validateTimeout(timeoutMs);
    return new Promise((resolve, reject) => {
      const finish = (error, value) => {
        const pending = this.pending.get(id);
        if (!pending) {
          return;
        }
        this.pending.delete(id);
        clearTimeout(pending.timer);
        options.signal?.removeEventListener('abort', pending.abort);
        if (error) {
          reject(error);
        } else {
          resolve(value);
        }
      };
      const abort = () => finish(options.signal?.reason ?? new GlyphoError('Request aborted'));
      const timer = setTimeout(
        () => finish(new GlyphoError(`Glypho timed out after ${timeoutMs} ms`)),
        timeoutMs,
      );
      this.pending.set(id, { finish, timer, abort });
      options.signal?.addEventListener('abort', abort, { once: true });
      worker.stdin.write(`${JSON.stringify({ id, ...payload })}\n`, (error) => {
        if (error) {
          finish(new GlyphoError(`Could not write to Glypho worker: ${error.message}`));
        }
      });
    });
  }

  startWorker() {
    if (this.worker && this.worker.exitCode === null) {
      return this.worker;
    }
    const args = ['serve', '--quality', this.quality, '--device', this.device];
    if (this.models) {
      args.push('--models', this.models);
    }
    if (this.threads !== undefined) {
      args.push('--threads', String(this.threads));
    }
    if (this.offline) {
      args.push('--offline');
    }
    const worker = spawn(this.binary, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });
    this.worker = worker;
    this.stderr = '';
    worker.stderr.setEncoding('utf8');
    worker.stderr.on('data', (chunk) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-64 * 1024);
    });
    createInterface({ input: worker.stdout }).on('line', (line) => this.handleResponse(line));
    worker.once('error', (error) => this.failPending(new GlyphoError(error.message, { cause: error })));
    worker.once('exit', (code, signal) => {
      const detail = this.stderr.trim();
      const suffix = detail || `worker exited with ${signal ?? code}`;
      this.failPending(new GlyphoError(`Glypho worker stopped: ${suffix}`));
    });
    return worker;
  }

  handleResponse(line) {
    let response;
    try {
      response = JSON.parse(line);
    } catch (error) {
      this.failPending(new GlyphoError('Glypho worker returned invalid JSON', { cause: error }));
      return;
    }
    const pending = this.pending.get(response.id);
    if (!pending) {
      return;
    }
    if (response.ok) {
      pending.finish(undefined, response.result);
    } else {
      pending.finish(new GlyphoError(response.error ?? 'Glypho request failed'));
    }
  }

  failPending(error) {
    for (const pending of [...this.pending.values()]) {
      pending.finish(error);
    }
  }
}

export function resolveBinary(configured) {
  if (configured) {
    return fileURLToPathSafe(configured);
  }
  if (process.env.GLYPHO_BIN) {
    return process.env.GLYPHO_BIN;
  }
  const packageName = platformPackage();
  if (packageName) {
    try {
      const packageJson = require.resolve(`${packageName}/package.json`);
      const name = process.platform === 'win32' ? 'glypho.exe' : 'glypho';
      return path.join(path.dirname(packageJson), 'bin', name);
    } catch (error) {
      if (error?.code !== 'MODULE_NOT_FOUND') {
        throw error;
      }
    }
  }
  const packageDirectory = path.dirname(fileURLToPath(import.meta.url));
  const root = path.resolve(packageDirectory, '..', '..');
  const name = process.platform === 'win32' ? 'glypho.exe' : 'glypho';
  for (const profile of ['release', 'debug']) {
    const candidate = path.join(root, 'target', profile, name);
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  throw new GlyphoError(`No Glypho binary package is available for ${process.platform}-${process.arch}`);
}

function platformPackage() {
  const suffix = {
    'darwin-arm64': 'darwin-arm64',
    'darwin-x64': 'darwin-x64',
    'linux-arm64': 'linux-arm64',
    'linux-x64': 'linux-x64',
    'win32-arm64': 'win32-arm64',
    'win32-x64': 'win32-x64',
  }[`${process.platform}-${process.arch}`];
  if (!suffix) {
    return undefined;
  }
  return suffix.startsWith('win32-')
    ? `@rinqaku/glypho-ocr-${suffix}`
    : `glypho-ocr-${suffix}`;
}

function fileURLToPathSafe(value) {
  if (value instanceof URL) {
    if (value.protocol !== 'file:') {
      throw new TypeError('Only file: URLs are supported');
    }
    return fileURLToPath(value);
  }
  return String(value);
}

function validateTimeout(timeoutMs) {
  if (!Number.isFinite(timeoutMs) || timeoutMs < 1 || timeoutMs > 300_000) {
    throw new RangeError('timeoutMs must be between 1 and 300000');
  }
}

function validateQuality(quality) {
  if (!['fast', 'balanced', 'accurate', 'maximum'].includes(quality)) {
    throw new TypeError(`Unsupported quality mode: ${quality}`);
  }
}

function validateDevice(device) {
  if (!['auto', 'cpu', 'cuda', 'coreml', 'openvino'].includes(device)) {
    throw new TypeError(`Unsupported device: ${device}`);
  }
}

function validateThreads(threads) {
  if (threads !== undefined && (!Number.isInteger(threads) || threads < 1 || threads > 64)) {
    throw new RangeError('threads must be between 1 and 64');
  }
}
