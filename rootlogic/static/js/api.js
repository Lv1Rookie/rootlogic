// The one place that talks to the FastAPI backend.

export const api = async (path, opts = {}) => {
  const r = await fetch(path, {
    headers: {"Content-Type": "application/json"}, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
