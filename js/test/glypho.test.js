import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { chmod, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { test } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Glypho } from '../src/index.js';


const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, '..', '..');
const samplePng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAUAAAABAAQAAAABDnJOzAAABR0lEQVR42u2VMU7DQBBF39iW7AIpLikQ2iNwg/hYKRBscjC06TgExUaiCJ0jUbiwPRSx4+wqEYgKIW81s/975s/fkSzKz07CTJyJf4oo1c4KshU2ZOwMKlKI2K3ZipwRO17BIlABmPcRqvZhxU+A6jkBDyj9gHRx6xqAF6AGevSaRr+0YMhoANrcWenzJtXHhVu66asM55oPyreCrAGaW+MGqHwIKwoFsB+dMuUJKy74WFB+6+PmqMAc87oq/ADV116mRyd/O+EQEw8ZJFS0gaapSVBRsKF2SNhfW4pwHqG9RFwDRiZ/U8X3EXExxi7sfR8peRpDBShdM84g9peLG22LH31uY2JVnedF7Ye8s0Fr1JN6Fmt1qmtNW1Dtc4UVqTo9nSRyL0HO3iYAboILORsvWrOUVTDM3RDkNlhcmf8zM3Em/jviF/NMZs4TGkYWAAAAAElFTkSuQmCC',
  'base64',
);


async function createSample() {
  const directory = await mkdtemp(path.join(tmpdir(), 'glypho-node-test-'));
  const image = path.join(directory, 'sample.png');
  await writeFile(image, samplePng);
  return { directory, image };
}


function runtimeOptions() {
  const models = path.join(root, 'models', 'installed');
  return existsSync(models) ? { models, offline: true } : {};
}

test('recognizes a synthetic image with a persistent worker', async () => {
  const sample = await createSample();
  const glypho = new Glypho({ languages: ['en'], ...runtimeOptions() });
  try {
    const before = await glypho.info();
    const document = await glypho.recognize(
      sample.image,
      { segmentation: 'single_line' },
    );

    assert.equal(before.quality, 'balanced');
    assert.equal(document.image.width, 320);
    assert.match(document.text, /GLYPHO/i);
    assert.match(document.text, /TEST/i);
    assert.ok(document.lines.length > 0);
  } finally {
    await glypho.close();
    await rm(sample.directory, { recursive: true, force: true });
  }
});

test('rejects invalid confidence before spawning Glypho', async () => {
  const sample = await createSample();
  const glypho = new Glypho();

  try {
    await assert.rejects(
      glypho.recognize(sample.image, { minConfidence: 2 }),
      RangeError,
    );
  } finally {
    await rm(sample.directory, { recursive: true, force: true });
  }
});

test('accepts the maximum model profile', () => {
  const glypho = new Glypho({ quality: 'maximum' });

  assert.equal(glypho.quality, 'maximum');
});

test('defaults to automatic language routing', () => {
  const glypho = new Glypho();

  assert.deepEqual(glypho.languages, []);
});

test('does not start a worker for an already aborted request', async () => {
  const glypho = new Glypho({ binary: '/definitely/missing/glypho' });
  const controller = new AbortController();
  controller.abort(new Error('cancelled before dispatch'));

  await assert.rejects(
    glypho.request({ method: 'info' }, { signal: controller.signal }),
    /cancelled before dispatch/,
  );
  assert.equal(glypho.worker, undefined);
});

test('terminates a stuck worker after timeout', { skip: process.platform === 'win32' }, async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'glypho-node-timeout-'));
  const binary = path.join(directory, 'glypho-stuck');
  await writeFile(binary, '#!/bin/sh\nwhile read line; do :; done\n');
  await chmod(binary, 0o700);
  const glypho = new Glypho({ binary, timeoutMs: 50 });

  try {
    await assert.rejects(glypho.info(), /timed out after 50 ms/);
    assert.equal(glypho.worker, undefined);
  } finally {
    await glypho.close();
    await rm(directory, { recursive: true, force: true });
  }
});
