import { motion } from 'framer-motion';
import { Check, CircleAlert, Download, LoaderCircle } from 'lucide-react';

import { MODELS, type ModelName } from '../engine/models';
import type { ModelState } from '../engine/types';

interface Props {
  models: ModelName[];
  states: Partial<Record<ModelName, ModelState>>;
}

export function ModelRail({ models, states }: Props) {
  return (
    <div className="model-rail">
      {models.map((model) => {
        const state = states[model] ?? { model, status: 'idle', progress: 0 };
        const active = state.status === 'downloading' || state.status === 'initializing';
        const done = state.status === 'ready' || state.status === 'cached';
        return (
          <motion.div
            layout
            key={model}
            className={`model-pill model-pill--${state.status}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <span className="model-pill__icon">
              {state.status === 'error'
                ? <CircleAlert size={13} />
                : done
                  ? <Check size={13} />
                  : active
                    ? <LoaderCircle size={13} className="spin" />
                    : <Download size={13} />}
            </span>
            <span className="model-pill__label">{MODELS[model].shortLabel}</span>
          </motion.div>
        );
      })}
    </div>
  );
}