export class ApiClient {
  constructor({ cacheLimit = 18 } = {}) {
    this.cacheLimit = cacheLimit;
    this.cache = new Map();
    this.controllers = new Map();
  }

  async get(url, { channel = url, cache = false } = {}) {
    this.abort(channel);
    if (cache && this.cache.has(url)) {
      const value = this.cache.get(url);
      this.cache.delete(url);
      this.cache.set(url, value);
      return value;
    }

    const controller = new AbortController();
    this.controllers.set(channel, controller);
    try {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        const body = await response.text();
        throw new HttpError(response.status, response.statusText, body);
      }
      const value = await response.json();
      if (cache) this.remember(url, value);
      return value;
    } finally {
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
    if (controller) controller.abort();
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
  }

  forget(url) {
    this.cache.delete(url);
  }

  remember(url, value) {
    this.cache.set(url, value);
    while (this.cache.size > this.cacheLimit) {
      this.cache.delete(this.cache.keys().next().value);
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
