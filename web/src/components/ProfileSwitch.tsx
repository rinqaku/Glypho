import { motion } from 'framer-motion';
import { useLayoutEffect, useRef, useState } from 'react';

import type { Quality } from '../engine/models';

export interface ProfileOption {
  value: Quality;
  label: string;
  hint: string;
}

interface IndicatorGeometry {
  x: number;
  width: number;
  height: number;
}

export function ProfileSwitch({
  value,
  options,
  disabled = false,
  onChange,
}: {
  value: Quality;
  options: ProfileOption[];
  disabled?: boolean;
  onChange: (value: Quality) => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const buttons = useRef(new Map<Quality, HTMLButtonElement>());
  const [indicator, setIndicator] = useState<IndicatorGeometry>({ x: 4, width: 0, height: 0 });

  useLayoutEffect(() => {
    const measure = () => {
      const container = root.current;
      const selected = buttons.current.get(value);
      if (!container || !selected) return;
      const containerRect = container.getBoundingClientRect();
      const selectedRect = selected.getBoundingClientRect();
      setIndicator({
        x: selectedRect.left - containerRect.left,
        width: selectedRect.width,
        height: selectedRect.height,
      });
    };

    measure();
    const observer = new ResizeObserver(measure);
    if (root.current) observer.observe(root.current);
    for (const button of buttons.current.values()) observer.observe(button);
    window.addEventListener('resize', measure);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [value, options]);

  return (
    <div ref={root} className="segmented" role="tablist" aria-label="Quality profile">
      <motion.span
        aria-hidden="true"
        className="segmented__active"
        initial={false}
        animate={{ x: indicator.x, width: indicator.width, height: indicator.height }}
        transition={{ type: 'spring', stiffness: 300, damping: 30, mass: 0.78 }}
      />
      {options.map((option) => {
        const active = value === option.value;
        return (
          <button
            ref={(node) => {
              if (node) buttons.current.set(option.value, node);
              else buttons.current.delete(option.value);
            }}
            type="button"
            key={option.value}
            className={`segmented__option ${active ? 'is-active' : ''}`}
            onClick={() => onChange(option.value)}
            disabled={disabled}
            role="tab"
            aria-selected={active}
          >
            <strong>{option.label}</strong>
            <small>{option.hint}</small>
          </button>
        );
      })}
    </div>
  );
}