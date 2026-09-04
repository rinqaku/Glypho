import { AnimatePresence, motion } from 'framer-motion';
import { Check, ChevronDown } from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';

export interface FancyOption<T extends string> {
  value: T;
  label: string;
  description?: string;
  icon?: ReactNode;
}

const ROW_HEIGHT = 46;
const TRIGGER_HEIGHT = 52;
const MENU_PADDING = 6;
const MENU_GAP = 4;

export function FancySelect<T extends string>({
  value, options, onChange, disabled = false, ariaLabel,
}: {
  value: T;
  options: FancyOption<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
  ariaLabel: string;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const selected = options.find((option) => option.value === value) ?? options[0];
  const expandedHeight = TRIGGER_HEIGHT + MENU_PADDING * 2 + options.length * ROW_HEIGHT + Math.max(0, options.length - 1) * MENU_GAP + 5;

  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, []);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  return (
    <div className={`fancy-select ${open ? 'is-open' : ''}`} ref={root}>
      <motion.div
        className="fancy-select__shell"
        initial={false}
        animate={{ height: open ? expandedHeight : TRIGGER_HEIGHT }}
        transition={{ type: 'spring', stiffness: 330, damping: 34, mass: 0.86 }}
      >
        <button
          type="button"
          className={`fancy-select__trigger ${open ? 'is-open' : ''}`}
          onClick={() => !disabled && setOpen((current) => !current)}
          disabled={disabled}
          aria-label={ariaLabel}
          aria-expanded={open}
        >
          <span className="fancy-select__value">
            {selected.icon && <span className="fancy-select__icon">{selected.icon}</span>}
            <strong>{selected.label}</strong>
          </span>
          <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}>
            <ChevronDown size={16} />
          </motion.span>
        </button>

        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              className="fancy-select__menu"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            >
              {options.map((option) => {
                const active = option.value === value;
                return (
                  <button
                    type="button"
                    key={option.value}
                    className={active ? 'is-active' : ''}
                    onClick={() => { onChange(option.value); setOpen(false); }}
                  >
                    <span className="fancy-select__value">
                      {option.icon && <span className="fancy-select__icon">{option.icon}</span>}
                      <strong>{option.label}</strong>
                    </span>
                    {active ? <Check size={15} /> : null}
                  </button>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}