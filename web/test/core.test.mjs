import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const root = path.resolve(import.meta.dirname, '..');
const temp = mkdtempSync(path.join(tmpdir(), 'glypho-web-test-'));
const localTsc = path.join(root, 'node_modules', '.bin', process.platform === 'win32' ? 'tsc.cmd' : 'tsc');
const tsc = existsSync(localTsc) ? localTsc : 'tsc';

try {
  execFileSync(tsc, [
    '--target', 'ES2022',
    '--module', 'CommonJS',
    '--moduleResolution', 'Node',
    '--rootDir', path.join(root, 'src', 'engine'),
    '--outDir', temp,
    '--skipLibCheck', 'true',
    '--strict', 'false',
    path.join(root, 'src', 'engine', 'models.ts'),
    path.join(root, 'src', 'engine', 'types.ts'),
    path.join(root, 'src', 'engine', 'languages.ts'),
    path.join(root, 'src', 'engine', 'geometry.ts'),
  ], { stdio: 'inherit' });
  writeFileSync(path.join(temp, 'package.json'), '{"type":"commonjs"}\n');

  const require = createRequire(import.meta.url);
  const languages = require(path.join(temp, 'languages.js'));
  const geometry = require(path.join(temp, 'geometry.js'));

  assert.deepEqual(languages.normalizeLanguages('EN, rus; ja'), ['en', 'ru', 'ja']);
  assert.deepEqual(languages.recognizerPlan('balanced', []), {
    primary: true, latin: true, cyrillic: true, korean: true,
  });
  assert.equal(languages.scriptTag('Привет'), 'Cyrl');
  assert.equal(languages.scriptTag('안녕하세요'), 'Kore');
  assert.equal(languages.scriptTag('日本語かな'), 'Jpan');
  assert.throws(() => languages.validateProfileLanguages('fast', ['ja']), /Japanese/);

  const angle = Math.PI / 7;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const source = [
    [-20, -5], [20, -5], [20, 5], [-20, 5],
  ].map(([x, y]) => ({ x: 50 + x * cos - y * sin, y: 40 + x * sin + y * cos }));
  const rect = geometry.minimumAreaRect(source);
  assert.ok(rect);
  assert.ok(Math.abs(rect.width - 40) < 0.5);
  assert.ok(Math.abs(rect.height - 10) < 0.5);

  const width = 64;
  const height = 32;
  const heatmap = new Float32Array(width * height);
  for (let y = 10; y <= 18; y += 1) {
    for (let x = 12; x <= 44; x += 1) heatmap[y * width + x] = 0.93;
  }
  const regions = await geometry.extractRegions(
    { dims: [1, 1, height, width], data: heatmap },
    640,
    320,
    { detectorThreshold: 0.3, boxThreshold: 0.6, unclipRatio: 1.5 },
  );
  assert.equal(regions.length, 1);
  assert.ok(regions[0].width > 300);
  assert.ok(regions[0].height > 80);
  assert.ok(regions[0].score > 0.9);

  const sorted = geometry.sortReadingOrder([
    { x: 200, y: 12, width: 50, height: 20, id: 'b' },
    { x: 10, y: 10, width: 80, height: 20, id: 'a' },
    { x: 15, y: 80, width: 50, height: 20, id: 'c' },
  ]);
  assert.deepEqual(sorted.map((item) => item.id), ['a', 'b', 'c']);

  console.log('Glypho Web core tests: 11 assertions passed.');
} finally {
  rmSync(temp, { recursive: true, force: true });
}