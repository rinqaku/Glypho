#!/usr/bin/env node

import { spawn } from 'node:child_process';

import { resolveBinary } from './index.js';


const child = spawn(resolveBinary(), process.argv.slice(2), {
  stdio: 'inherit',
  windowsHide: true,
});
child.once('error', (error) => {
  console.error(`glypho: ${error.message}`);
  process.exitCode = 1;
});
child.once('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exitCode = code ?? 1;
});
