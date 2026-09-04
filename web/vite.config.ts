import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Threaded ONNX Runtime WASM needs cross-origin isolation. These headers make
// local dev/preview behave like the production deployment configured by _headers.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
};

export default defineConfig({
  plugins: [react()],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    sourcemap: false,
  },
});