import { motion } from 'framer-motion';
import { Minus, Plus, Search, Undo2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import type { OcrResult } from '../engine/types';

interface Props {
  src?: string;
  result?: OcrResult;
  scanning: boolean;
  imageSize?: { width: number; height: number };
  onSelectLine?: (index: number) => void;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 5;

export function Preview({ src, result, scanning, imageSize, onSelectLine }: Props) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const stage = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; panX: number; panY: number; pointerId: number } | null>(null);

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [src]);

  useEffect(() => {
    const node = stage.current;
    if (!node || !src) return;

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      const delta = event.deltaY < 0 ? 0.18 : -0.18;
      setZoom((current) => {
        const next = clamp(current + delta, MIN_ZOOM, MAX_ZOOM);
        if (next === 1) setPan({ x: 0, y: 0 });
        return next;
      });
    };

    node.addEventListener('wheel', handleWheel, { passive: false });
    return () => node.removeEventListener('wheel', handleWheel);
  }, [src]);

  const updateZoom = (next: number) => {
    const clamped = clamp(next, MIN_ZOOM, MAX_ZOOM);
    setZoom(clamped);
    if (clamped === 1) setPan({ x: 0, y: 0 });
  };

  if (!src) {
    return (
      <div className="preview preview--empty">
        <Search size={28} />
        <span>Your image will appear here</span>
      </div>
    );
  }

  return (
    <div className="preview">
      <div className="preview__toolbar">
        <strong>Preview</strong>
        <div className="preview__tools">
          <button type="button" onClick={() => updateZoom(zoom - 0.25)} aria-label="Zoom out"><Minus size={15} /></button>
          <span>{Math.round(zoom * 100)}%</span>
          <button type="button" onClick={() => updateZoom(zoom + 0.25)} aria-label="Zoom in"><Plus size={15} /></button>
          <button type="button" onClick={() => { updateZoom(1); setPan({ x: 0, y: 0 }); }} aria-label="Reset zoom"><Undo2 size={15} /></button>
        </div>
      </div>
      <div
        ref={stage}
        className={`preview__stage ${zoom > 1 ? 'is-zoomed' : ''}`}
      >
        <motion.div
          className="preview__transform"
          style={{ x: pan.x, y: pan.y }}
          animate={{ scale: zoom }}
          transition={{ type: 'spring', stiffness: 250, damping: 28, mass: 0.7 }}
          onPointerDown={(event) => {
            if (zoom <= 1) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            drag.current = {
              x: event.clientX,
              y: event.clientY,
              panX: pan.x,
              panY: pan.y,
              pointerId: event.pointerId,
            };
          }}
          onPointerMove={(event) => {
            if (!drag.current || drag.current.pointerId !== event.pointerId || zoom <= 1) return;
            setPan({
              x: drag.current.panX + event.clientX - drag.current.x,
              y: drag.current.panY + event.clientY - drag.current.y,
            });
          }}
          onPointerUp={(event) => {
            if (drag.current?.pointerId === event.pointerId) drag.current = null;
          }}
          onPointerCancel={() => { drag.current = null; }}
        >
          <div className="preview__image-wrap">
            <img src={src} alt="OCR source" draggable={false} />
            {result && result.lines.length > 0 && imageSize && (
              <svg className="preview__overlay" viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="none">
                {result.lines.map((line, index) => (
                  <motion.polygon
                    key={`${line.text}-${index}`}
                    points={line.quad.map((point) => `${point.x},${point.y}`).join(' ')}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{ delay: Math.min(index * 0.035, 0.45), duration: 0.35 }}
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectLine?.(index);
                    }}
                  />
                ))}
              </svg>
            )}
            {scanning && (
              <motion.div
                key="active-scan-line"
                className="scan-line"
                initial={{ top: '3%', opacity: 0 }}
                animate={{ top: ['3%', '96%', '3%'], opacity: [0, 1, 1, 0] }}
                transition={{ repeat: Infinity, duration: 3.1, ease: 'easeInOut' }}
              />
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}