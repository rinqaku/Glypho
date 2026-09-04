import { createRoot } from 'react-dom/client';

import App from './App';
import './styles.css';

// Do not wrap the app in React.StrictMode: development StrictMode intentionally
// re-runs effects, which would initialize/download heavyweight OCR sessions twice.
createRoot(document.getElementById('root')!).render(<App />);