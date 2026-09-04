import { AnimatePresence, motion } from 'framer-motion';
import {
  CheckCircle2, Cpu, Github, Globe2, Languages, LoaderCircle, MonitorUp, ScanLine,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react';

import { Dropzone } from './components/Dropzone';
import { FancySelect } from './components/FancySelect';
import { ModelRail } from './components/ModelRail';
import { Preview } from './components/Preview';
import { ProfileSwitch } from './components/ProfileSwitch';
import { ResultPanel } from './components/ResultPanel';
import { GlyphoWebClient, inspectImage } from './engine/client';
import type { RuntimeMode } from './engine/ocr';
import { detectorFor, modelBytes, primaryRecognizerFor, type ModelName, type Quality } from './engine/models';
import type { ModelState, OcrResult, ProgressEvent } from './engine/types';

const SPECIALISTS: ModelName[] = ['v5-latin-rec', 'v5-eslav-rec', 'v5-korean-rec'];
const QUALITY_OPTIONS: Array<{ value: Quality; label: string; hint: string }> = [
  { value: 'fast', label: 'Fast', hint: 'Low latency' },
  { value: 'balanced', label: 'Balanced', hint: 'Default' },
  { value: 'accurate', label: 'Accurate', hint: 'More detail' },
  { value: 'maximum', label: 'Maximum', hint: 'Accuracy first' },
];

const RUNTIME_OPTIONS: Array<{ value: RuntimeMode; label: string; icon: ReactNode }> = [
  { value: 'auto', label: 'Auto', icon: <ScanLine size={16} /> },
  { value: 'wasm', label: 'WASM', icon: <Cpu size={16} /> },
  { value: 'webgpu', label: 'WebGPU', icon: <MonitorUp size={16} /> },
];

type LanguagePreset = 'auto' | 'en' | 'en-ru' | 'ja-en' | 'ko-en' | 'custom';
const LANGUAGE_OPTIONS: Array<{ value: LanguagePreset; label: string; icon: ReactNode }> = [
  { value: 'auto', label: 'Auto multilingual', icon: <Globe2 size={16} /> },
  { value: 'en', label: 'English', icon: <Languages size={16} /> },
  { value: 'en-ru', label: 'English + Russian', icon: <Languages size={16} /> },
  { value: 'ja-en', label: 'Japanese + English', icon: <Languages size={16} /> },
  { value: 'ko-en', label: 'Korean + English', icon: <Languages size={16} /> },
  { value: 'custom', label: 'Custom hints', icon: <Languages size={16} /> },
];

type StatusTone = 'danger' | 'loading' | 'ready';

export default function App() {
  const client = useRef<GlyphoWebClient | null>(null);
  if (!client.current) client.current = new GlyphoWebClient();

  const [quality, setQuality] = useState<Quality>('balanced');
  const [runtime, setRuntime] = useState<RuntimeMode>('auto');
  const [languagePreset, setLanguagePreset] = useState<LanguagePreset>('auto');
  const [languageText, setLanguageText] = useState('');
  const [file, setFile] = useState<File>();
  const [previewUrl, setPreviewUrl] = useState<string>();
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>();
  const [result, setResult] = useState<OcrResult>();
  const [selectedLine, setSelectedLine] = useState<number>();
  const [modelStates, setModelStates] = useState<Partial<Record<ModelName, ModelState>>>({});
  const [provider, setProvider] = useState('Preparing…');
  const [wasmThreads, setWasmThreads] = useState(1);
  const [engineReady, setEngineReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Preparing models…');
  const [progress, setProgress] = useState(0.03);
  const [error, setError] = useState<string>();
  const [dragOverlay, setDragOverlay] = useState(false);
  const generation = useRef(0);
  const dragDepth = useRef(0);

  const languages = useMemo(() => languageText
    .split(/[\s,;]+/)
    .map((value) => value.trim())
    .filter(Boolean), [languageText]);

  const visibleModels = useMemo<ModelName[]>(() => {
    const unique = new Set<ModelName>([
      detectorFor(quality),
      primaryRecognizerFor(quality),
      ...SPECIALISTS,
    ]);
    return [...unique];
  }, [quality]);

  const modelBootProgress = useMemo(() => {
    let loaded = 0;
    let total = 0;
    for (const model of visibleModels) {
      const weight = modelBytes(model);
      total += weight;
      const state = modelStates[model];
      const fraction = state?.status === 'ready' || state?.status === 'cached'
        ? 1
        : clamp(state?.progress ?? 0, 0, 1);
      loaded += weight * fraction;
    }
    return total > 0 ? loaded / total : 0;
  }, [modelStates, visibleModels]);

  const statusTone: StatusTone = error
    ? 'danger'
    : engineReady && !busy
      ? 'ready'
      : 'loading';

  const overallProgress = useMemo(() => {
    if (error) return 0;
    if (engineReady && !busy) return 1;
    if (busy) return clamp(progress, 0.02, 1);
    return clamp(Math.max(progress, modelBootProgress), 0.03, 0.98);
  }, [busy, engineReady, error, modelBootProgress, progress]);

  const readinessLabel = error
    ? 'Not ready'
    : busy
      ? 'Running'
      : engineReady
        ? 'Ready'
        : 'Preparing';

  useEffect(() => {
    const unsubscribe = client.current!.onProgress((event) => handleProgress(event, setModelStates, setStatus, setProgress));
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    const id = ++generation.current;
    setEngineReady(false);
    setError(undefined);
    setProvider('Preparing…');
    setModelStates({});
    setStatus('Preparing models…');
    setProgress(0.03);

    const timer = window.setTimeout(async () => {
      try {
        client.current!.restart();
        const configured = await client.current!.configure(quality, languages, runtime);
        if (id !== generation.current) return;
        setProvider(configured.provider);
        setWasmThreads(configured.wasmThreads);
        setEngineReady(true);
        setStatus('Ready');
        setProgress(1);
      } catch (cause) {
        if (id !== generation.current) return;
        const message = readableError(cause);
        setError(message);
        setStatus('Runtime could not be prepared');
        setProgress(0);
      }
    }, 180);

    return () => window.clearTimeout(timer);
  }, [quality, languageText, runtime]);

  useEffect(() => () => { client.current?.dispose(); }, []);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const chooseFile = async (next: File) => {
    try {
      const dimensions = await inspectImage(next);
      const url = URL.createObjectURL(next);
      setPreviewUrl(url);
      setFile(next);
      setResult(undefined);
      setSelectedLine(undefined);
      setError(undefined);
      setImageSize(dimensions);
    } catch (cause) {
      setError(readableError(cause));
      setStatus('Image could not be opened');
      setImageSize(undefined);
    }
  };

  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const items = Array.from(event.clipboardData?.items ?? []);
      const imageItem = items.find((item) => item.kind === 'file' && item.type.startsWith('image/'));
      const image = imageItem?.getAsFile();
      if (!image) return;
      event.preventDefault();
      void chooseFile(image);
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, []);

  useEffect(() => {
    const hasImageFiles = (event: DragEvent) => {
      const items = Array.from(event.dataTransfer?.items ?? []).filter((item) => item.kind === 'file');
      if (items.length > 0) return items.some((item) => item.type.startsWith('image/'));
      return Array.from(event.dataTransfer?.types ?? []).includes('Files');
    };

    const handleDragEnter = (event: DragEvent) => {
      if (!hasImageFiles(event)) return;
      dragDepth.current += 1;
      setDragOverlay(true);
    };

    const handleDragOver = (event: DragEvent) => {
      if (!hasImageFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
      setDragOverlay(true);
    };

    const handleDragLeave = (event: DragEvent) => {
      if (!hasImageFiles(event)) return;
      if (event.relatedTarget === null) {
        dragDepth.current = 0;
        setDragOverlay(false);
        return;
      }
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragOverlay(false);
    };

    const handleDrop = (event: DragEvent) => {
      dragDepth.current = 0;
      setDragOverlay(false);
      if (!hasImageFiles(event) || event.defaultPrevented) return;
      const image = Array.from(event.dataTransfer?.files ?? []).find((candidate) => candidate.type.startsWith('image/'));
      if (!image) return;
      event.preventDefault();
      void chooseFile(image);
    };

    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('drop', handleDrop);
    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('drop', handleDrop);
    };
  }, []);

  const recognize = async () => {
    if (!file || !engineReady || busy) return;
    setBusy(true);
    setResult(undefined);
    setSelectedLine(undefined);
    setError(undefined);
    setStatus('Recognizing…');
    setProgress(0.02);
    try {
      const output = await client.current!.recognize(file);
      setResult(output);
      setProvider(output.provider);
      setWasmThreads(output.wasmThreads);
      setStatus(`Done · ${output.lines.length} text region${output.lines.length === 1 ? '' : 's'}`);
      setProgress(1);
    } catch (cause) {
      const message = readableError(cause);
      setError(message);
      setStatus('Recognition failed');
      setProgress(0);
    } finally {
      setBusy(false);
    }
  };

  const chooseLanguagePreset = (preset: LanguagePreset) => {
    setLanguagePreset(preset);
    const values: Record<Exclude<LanguagePreset, 'custom'>, string> = {
      auto: '', en: 'en', 'en-ru': 'en, ru', 'ja-en': 'ja, en', 'ko-en': 'ko, en',
    };
    if (preset !== 'custom') setLanguageText(values[preset]);
  };

  return (
    <div className="app-shell">
      <AnimatePresence>
        {dragOverlay && (
          <motion.div
            className="global-drop-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            aria-hidden="true"
          >
            <motion.div
              className="global-drop-overlay__pill"
              initial={{ opacity: 0, scale: 0.92, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 6 }}
              transition={{ type: 'spring', stiffness: 320, damping: 28 }}
            >
              <span className="global-drop-overlay__icon">⇩</span>
              <strong>Drag here</strong>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <header className="topbar">
        <motion.span
          className="brand"
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        >
          Glypho Web
        </motion.span>
        <motion.div
          className="topbar__meta"
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.04, ease: [0.22, 1, 0.36, 1] }}
        >
          <span>100% on-device</span>
          <a
            href="https://github.com/rinqaku/Glypho"
            target="_blank"
            rel="noreferrer"
            title="Glypho project on GitHub"
          >
            <Github size={16} /> GitHub
          </a>
        </motion.div>
      </header>

      <main id="top" className="layout">
        <motion.section
          className="surface controls-panel"
          initial={{ opacity: 0, y: 18, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="controls-grid">
            <div className="field field--wide">
              <label>Profile</label>
              <ProfileSwitch
                value={quality}
                options={QUALITY_OPTIONS}
                onChange={setQuality}
                disabled={busy}
              />
            </div>

            <div className="field">
              <label>Runtime</label>
              <FancySelect<RuntimeMode>
                value={runtime}
                options={RUNTIME_OPTIONS}
                onChange={setRuntime}
                disabled={busy}
                ariaLabel="OCR runtime"
              />
            </div>

            <div className="field">
              <label>Languages</label>
              <FancySelect<LanguagePreset>
                value={languagePreset}
                options={LANGUAGE_OPTIONS}
                onChange={chooseLanguagePreset}
                disabled={busy}
                ariaLabel="Language hints"
              />
            </div>
          </div>

          <AnimatePresence initial={false}>
            {languagePreset === 'custom' && (
              <motion.div
                className="language-input"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 54 }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
              >
                <Languages size={16} />
                <input
                  value={languageText}
                  onChange={(event) => setLanguageText(event.target.value)}
                  placeholder="en, ru, ja"
                  spellCheck={false}
                  disabled={busy}
                />
              </motion.div>
            )}
          </AnimatePresence>

          <ModelRail models={visibleModels} states={modelStates} />
        </motion.section>

        <motion.section
          className="surface drop-row"
          initial={{ opacity: 0, y: 18, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.55, delay: 0.05, ease: [0.22, 1, 0.36, 1] }}
        >
          <Dropzone file={file} busy={busy} onFile={chooseFile} />
        </motion.section>

        <section className="workspace">
          <motion.div
            className="surface workspace__surface"
            initial={{ opacity: 0, y: 18, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.55, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          >
            <Preview
              src={previewUrl}
              result={result}
              scanning={busy}
              imageSize={imageSize}
              onSelectLine={setSelectedLine}
            />
          </motion.div>
          <motion.div
            className="surface workspace__surface"
            initial={{ opacity: 0, y: 18, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.55, delay: 0.14, ease: [0.22, 1, 0.36, 1] }}
          >
            <ResultPanel
              result={result}
              provider={provider}
              wasmThreads={wasmThreads}
              selectedLine={selectedLine}
            />
          </motion.div>
        </section>

        <motion.section
          className="surface status-bar"
          initial={{ opacity: 0, y: 18, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.55, delay: 0.18, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="status-bar__main">
            <div className="status-bar__header">
              <div className={`status-indicator status-indicator--${statusTone}`}>
                {statusTone === 'ready'
                  ? <CheckCircle2 size={15} />
                  : statusTone === 'loading'
                    ? <LoaderCircle size={15} className="spin" />
                    : <span className="status-indicator__dot" />}
                <strong>{readinessLabel}</strong>
              </div>
              <span className="status-bar__message">{status}</span>
            </div>
            <div className="progress-track">
              <motion.i
                className={`progress-track__fill progress-track__fill--${statusTone}`}
                animate={{ width: `${Math.round(overallProgress * 100)}%` }}
                transition={{ type: 'spring', stiffness: 170, damping: 24, mass: 0.9 }}
              />
            </div>
          </div>

          <motion.button
            className="primary-button"
            type="button"
            onClick={recognize}
            disabled={!file || !engineReady || busy}
            whileHover={file && engineReady && !busy ? { scale: 1.015 } : undefined}
            whileTap={file && engineReady && !busy ? { scale: 0.985 } : undefined}
          >
            {busy ? <><span className="button-spinner" /> Recognizing…</> : 'Recognize'}
          </motion.button>
        </motion.section>

        <AnimatePresence>
          {error && (
            <motion.div
              className="surface error-banner"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <span className="status-indicator__dot" />
              <div>
                <strong>Glypho Web needs attention</strong>
                <span>{error}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function handleProgress(
  event: ProgressEvent,
  setModelStates: Dispatch<SetStateAction<Partial<Record<ModelName, ModelState>>>>,
  setStatus: Dispatch<SetStateAction<string>>,
  setProgress: Dispatch<SetStateAction<number>>,
): void {
  if (event.model && ['downloading', 'initializing', 'ready', 'error'].includes(event.phase)) {
    setModelStates((current) => ({
      ...current,
      [event.model!]: {
        model: event.model!,
        status: modelStatus(event),
        progress: event.progress ?? current[event.model!]?.progress ?? 0,
        loadedBytes: event.loadedBytes ?? current[event.model!]?.loadedBytes,
        totalBytes: event.totalBytes ?? current[event.model!]?.totalBytes,
        message: event.message,
      },
    }));
  }

  // Only model-less events drive the single global progress bar. This prevents a
  // small model finishing early from jumping the bar to 100% while larger models download.
  if (!event.model) {
    setStatus(event.message);
    if (typeof event.progress === 'number') setProgress(event.progress);
  }
}

function modelStatus(event: ProgressEvent): ModelState['status'] {
  if (event.phase === 'error') return 'error';
  if (event.phase === 'initializing') return 'initializing';
  if (event.phase === 'downloading') return event.cached ? 'cached' : 'downloading';
  if (event.phase === 'ready') return event.cached ? 'cached' : 'ready';
  return 'idle';
}

function readableError(cause: unknown): string {
  if (cause instanceof Error) return cause.message.split('\n')[0];
  return String(cause);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}