export class FootprintCollectionCache {
  constructor({
    limit = 12,
    maxBytes = 134_217_728,
    normalize = (value) => value,
    estimateBytes = estimateFootprintBytes,
  } = {}) {
    this.limit = Math.max(1, Number(limit) || 12);
    this.maxBytes = Math.max(0, Number(maxBytes) || 0);
    this.normalize = normalize;
    this.estimateBytes = estimateBytes;
    this.values = new Map();
    this.inflight = new Map();
    this.sizeBytes = 0;
  }

  async load(key, loader) {
    const cacheKey = String(key || "");
    if (!cacheKey) throw new Error("A footprint cache key is required.");
    if (this.values.has(cacheKey)) {
      const entry = this.values.get(cacheKey);
      this.values.delete(cacheKey);
      this.values.set(cacheKey, entry);
      return entry.value;
    }
    if (this.inflight.has(cacheKey)) return this.inflight.get(cacheKey);
    const request = Promise.resolve()
      .then(loader)
      .then((payload) => {
        const value = this.normalize(payload);
        this.remember(cacheKey, value);
        return value;
      })
      .finally(() => {
        if (this.inflight.get(cacheKey) === request) this.inflight.delete(cacheKey);
      });
    this.inflight.set(cacheKey, request);
    return request;
  }

  remember(key, value) {
    const sizeBytes = Math.max(0, Number(this.estimateBytes(value)) || 0);
    const previous = this.values.get(key);
    if (previous) {
      this.sizeBytes -= previous.sizeBytes;
      this.values.delete(key);
    }
    if (!this.maxBytes || sizeBytes > this.maxBytes) return;
    this.values.set(key, { value, sizeBytes });
    this.sizeBytes += sizeBytes;
    while (this.values.size > this.limit || this.sizeBytes > this.maxBytes) {
      const oldest = this.values.keys().next().value;
      const evicted = this.values.get(oldest);
      this.values.delete(oldest);
      this.sizeBytes -= evicted?.sizeBytes || 0;
    }
  }

  clear() {
    this.values.clear();
    this.inflight.clear();
    this.sizeBytes = 0;
  }
}

export function estimateFootprintBytes(collection) {
  let bytes = 256;
  for (const feature of collection?.features || []) {
    bytes += 192;
    try {
      bytes += JSON.stringify(feature?.properties || {}).length * 2;
    } catch {
      bytes += 1024;
    }
    bytes += coordinateValueCount(feature?.geometry?.coordinates) * 8;
  }
  return bytes;
}

export function loadFootprintCollection(
  api,
  collectionCache,
  url,
  { channel = "incident-footprints:current" } = {},
) {
  return collectionCache.load(
    url,
    () => api.get(url, { channel, cache: false }),
  );
}

function coordinateValueCount(value) {
  if (!Array.isArray(value)) return 0;
  let count = 0;
  const stack = [value];
  while (stack.length) {
    const current = stack.pop();
    for (const item of current) {
      if (Array.isArray(item)) stack.push(item);
      else if (typeof item === "number") count += 1;
    }
  }
  return count;
}
