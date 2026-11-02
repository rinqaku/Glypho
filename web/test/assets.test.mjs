import assert from 'node:assert/strict';

export async function testAssets({ fetchArtifact }) {
  const originalFetch = globalThis.fetch;
  const originalCaches = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  const originalWarn = console.warn;
  const bytes = new Uint8Array([1, 2, 3]);
  const sha256 = Buffer.from(await crypto.subtle.digest('SHA-256', bytes)).toString('hex');
  const model = { artifacts: { 'inference.onnx': { url: 'https://example.invalid/model', bytes: 3, sha256 } } };
  const failed = async () => { throw new DOMException('storage unavailable', 'QuotaExceededError'); };
  let requests = 0;

  const useCache = (open) => Object.defineProperty(globalThis, 'caches', {
    configurable: true, value: { open },
  });
  const download = async () => {
    const events = [];
    const result = await fetchArtifact(model, 'inference.onnx', 'v6-small-det', (event) => events.push(event));
    assert.deepEqual(new Uint8Array(result), bytes);
    return events.at(-1);
  };

  try {
    console.warn = () => {};
    globalThis.fetch = async () => { requests += 1; return new Response(bytes); };

    delete globalThis.caches;
    assert.equal((await download()).cached, false);

    useCache(failed);
    assert.equal((await download()).cached, false);

    useCache(async () => ({ match: failed, delete: failed, put: failed }));
    assert.equal((await download()).cached, false);

    useCache(async () => ({ match: async () => undefined, put: failed }));
    const memoryOnly = await download();
    assert.equal(memoryOnly.cached, false);
    assert.match(memoryOnly.message, /memory only/);

    let stored;
    useCache(async () => ({ match: async () => undefined, put: async (_url, response) => { stored = response; } }));
    assert.equal((await download()).cached, true);
    assert.deepEqual(new Uint8Array(await stored.arrayBuffer()), bytes);

    const beforeHit = requests;
    useCache(async () => ({ match: async () => new Response(bytes) }));
    assert.equal((await download()).cached, true);
    assert.equal(requests, beforeHit);

    let deleted = false;
    useCache(async () => ({
      match: async () => new Response(new Uint8Array([3, 2, 1])),
      delete: async () => { deleted = true; }, put: async () => {},
    }));
    assert.equal((await download()).cached, true);
    assert.equal(deleted, true);

    useCache(async () => ({ match: async () => new Response(new Uint8Array(4)), delete: failed, put: failed }));
    assert.equal((await download()).cached, false);

    useCache(failed);
    globalThis.fetch = async () => new Response(new Uint8Array([3, 2, 1]));
    await assert.rejects(download(), /SHA-256/);
    globalThis.fetch = async () => new Response(new Uint8Array(4));
    await assert.rejects(download(), /exceeded/);
    globalThis.fetch = async () => { throw new Error('offline'); };
    await assert.rejects(download(), /offline/);
  } finally {
    globalThis.fetch = originalFetch;
    console.warn = originalWarn;
    if (originalCaches) Object.defineProperty(globalThis, 'caches', originalCaches);
    else delete globalThis.caches;
  }
}
