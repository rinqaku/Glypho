import { motion } from 'framer-motion';
import { Check, Clipboard, Gauge, Languages, Timer, WandSparkles } from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';

import type { OcrResult } from '../engine/types';

interface Props {
  result?: OcrResult;
  provider: string;
  wasmThreads: number;
  selectedLine?: number;
}

export function ResultPanel({ result, provider, wasmThreads, selectedLine }: Props) {
  const [copied, setCopied] = useState(false);
  const textContainer = useRef<HTMLDivElement>(null);
  const lineRefs = useRef<Array<HTMLSpanElement | null>>([]);

  useEffect(() => {
    if (selectedLine === undefined) return;
    const container = textContainer.current;
    const line = lineRefs.current[selectedLine];
    if (!container || !line) return;

    const targetTop = line.offsetTop - container.offsetTop - container.clientHeight / 2 + line.clientHeight / 2;
    container.scrollTo({ top: Math.max(0, targetTop), behavior: 'smooth' });

    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(line);
    selection?.removeAllRanges();
    selection?.addRange(range);
  }, [result, selectedLine]);

  const copy = async () => {
    if (!result?.text) return;
    await navigator.clipboard.writeText(result.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <aside className="result-panel">
      <div className="result-panel__head">
        <div>
          <span className="eyebrow">Result</span>
          <strong>{result ? `${result.lines.length} region${result.lines.length === 1 ? '' : 's'}` : 'Ready'}</strong>
        </div>
        <button type="button" className="icon-button" onClick={copy} disabled={!result?.text}>
          {copied ? <Check size={16} /> : <Clipboard size={16} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>

      <div ref={textContainer} className={`result-text ${result?.text ? '' : 'is-empty'}`}>
        {result?.lines.length
          ? result.lines.map((line, index) => (
            <span
              className="result-text__line"
              key={`${line.text}-${index}`}
              ref={(node) => { lineRefs.current[index] = node; }}
            >
              {line.text}
            </span>
          ))
          : 'Recognized text will appear here.'}
      </div>

      <div className="metrics">
        <Metric icon={<Timer size={14} />} label="Total" value={result ? formatMs(result.elapsedMs) : '—'} />
        <Metric icon={<Gauge size={14} />} label="Detect" value={result ? formatMs(result.detectorInferenceMs + result.postprocessMs) : '—'} />
        <Metric icon={<WandSparkles size={14} />} label="Read" value={result ? formatMs(result.recognitionMs) : '—'} />
        <Metric icon={<Languages size={14} />} label="Runtime" value={provider === 'WASM' ? `${provider} · ${wasmThreads}t` : provider} />
      </div>

      {result && (
        <motion.div className="timing-bar" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <i style={{ flex: Math.max(1, result.preprocessMs) }} title={`Preprocess ${formatMs(result.preprocessMs)}`} />
          <i style={{ flex: Math.max(1, result.detectorInferenceMs) }} title={`Detector ${formatMs(result.detectorInferenceMs)}`} />
          <i style={{ flex: Math.max(1, result.postprocessMs) }} title={`Postprocess ${formatMs(result.postprocessMs)}`} />
          <i style={{ flex: Math.max(1, result.recognitionMs) }} title={`Recognition ${formatMs(result.recognitionMs)}`} />
        </motion.div>
      )}
    </aside>
  );
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return <div className="metric"><span>{icon}{label}</span><strong>{value}</strong></div>;
}

function formatMs(value: number): string {
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}