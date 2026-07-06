export class ApiClient {
  constructor({ cacheLimit = 18, cacheByteLimit = 32 * 1024 * 1024 } = {}) {
    this.cacheLimit = cacheLimit;
    this.cacheByteLimit = cacheByteLimit;
    this.cacheBytesUsed = 0;
    this.cache = new Map();
    this.controllers = new Map();
    this.inflight = new Map();
  }

  async get(url, { channel = url, cache = false } = {}) {
    this.abort(channel);
    if (cache && this.cache.has(url)) {
      const entry = this.cache.get(url);
      this.cache.delete(url);
      this.cache.set(url, entry);
      return entry.value;
    }

    const existing = this.inflight.get(url);
    if (existing && !existing.controller.signal.aborted) {
      this.controllers.set(channel, existing.controller);
      try {
        const value = await existing.promise;
        if (cache) this.remember(url, value);
        return value;
      } finally {
        if (this.controllers.get(channel) === existing.controller) this.controllers.delete(channel);
      }
    }
    if (existing) this.inflight.delete(url);

    const controller = new AbortController();
    this.controllers.set(channel, controller);
    const request = (async () => {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        const body = await response.text();
        throw new HttpError(response.status, response.statusText, body);
      }
      return response.json();
    })();
    const entry = { promise: request, controller };
    this.inflight.set(url, entry);
    try {
      const value = await request;
      if (cache) this.remember(url, value);
      return value;
    } finally {
      if (this.inflight.get(url) === entry) this.inflight.delete(url);
      if (this.controllers.get(channel) === controller) this.controllers.delete(channel);
    }
  }

  async post(url, body, { channel = url } = {}) {
    this.abort(channel);
    const controller = new AbortController();
    this.controllers.set(channel, controller);
    try {
      const response = await fetch(url, {
        method: "POST",
        signal: controller.signal,
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const responseBody = await response.text();
        throw new HttpError(response.status, response.statusText, responseBody);
      }
      return await response.json();
    } finally {
      if (this.controllers.get(channel) === controller) this.controllers.delete(channel);
    }
  }

  preload(url) {
    if (this.cache.has(url)) return;
    this.get(url, { channel: `prefetch:${url}`, cache: true }).catch(() => {});
  }

  abort(channel) {
    const controller = this.controllers.get(channel);
    if (controller) {
      controller.abort();
      for (const [url, entry] of this.inflight) {
        if (entry.controller === controller) this.inflight.delete(url);
      }
    }
    this.controllers.delete(channel);
  }

  abortPrefix(prefix) {
    for (const channel of [...this.controllers.keys()]) {
      if (String(channel).startsWith(prefix)) this.abort(channel);
    }
  }

  abortAll() {
    for (const controller of this.controllers.values()) controller.abort();
    this.controllers.clear();
    this.inflight.clear();
  }

  forget(url) {
    const entry = this.cache.get(url);
    if (entry) this.cacheBytesUsed = Math.max(0, this.cacheBytesUsed - entry.bytes);
    this.cache.delete(url);
  }

  remember(url, value) {
    let bytes;
    try {
      bytes = new TextEncoder().encode(JSON.stringify(value)).byteLength;
    } catch (_error) {
      return;
    }
    this.forget(url);
    if (bytes > this.cacheByteLimit) return;
    this.cache.set(url, { value, bytes });
    this.cacheBytesUsed += bytes;
    while (this.cache.size > this.cacheLimit || this.cacheBytesUsed > this.cacheByteLimit) {
      const oldest = this.cache.keys().next().value;
      this.forget(oldest);
    }
  }
}

export class HttpError extends Error {
  constructor(status, statusText, body = "") {
    super(`${status} ${statusText}${body ? `: ${body}` : ""}`);
    this.name = "HttpError";
    this.status = Number(status);
  }
}

export function withQuery(path, values = {}) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined && value !== null && String(value) !== "") {
      query.set(key, String(value));
    }
  }
  const suffix = query.toString();
  return suffix ? `${path}?${suffix}` : path;
}

export function isAbortError(error) {
  return error?.name === "AbortError";
}

export function isUnsupportedError(error) {
  return [404, 405, 501].includes(Number(error?.status));
}
