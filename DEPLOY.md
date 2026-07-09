# Deploying the dashboard as a static page (e.g. clusterfree.org/burden)

The dashboard runs in **two modes from one `index.html`**:

- **Local dev:** `python3 server.py` serves the page and answers the API. Fast
  native NumPy. The page auto-detects the backend and uses it.
- **Static hosting (public):** with no backend present, the page runs the *same*
  compute code (`webapi.dispatch`) **in the browser** via
  [Pyodide](https://pyodide.org) (Python + NumPy compiled to WebAssembly). No
  server, no serverless functions, no timeouts, no per-request cost.

The mode is chosen automatically: the page probes the relative URL `api/config`.
Served at `/` (the local server) it succeeds → backend mode. Served as static
files under a subpath (`/burden/`) it 404s → Pyodide mode.

## Put it on your existing Vercel / Next.js site

Copy **three files** into your Next.js repo under `public/burden/`:

```
public/burden/index.html
public/burden/ch_simulation.py
public/burden/webapi.py
```

That's it. Vercel serves everything in `public/` as static assets, so the page
is live at:

```
https://clusterfree.org/burden/
```

The page fetches `ch_simulation.py` and `webapi.py` **relative to itself**
(`/burden/ch_simulation.py`, `/burden/webapi.py`), loads them into Pyodide, and
runs the model client-side. Nothing else in your Next.js app is touched.

### Optional: clean `/burden` URL (no trailing slash)

Static index files resolve at `/burden/`. If you want `/burden` to work too, add
a rewrite in `vercel.json` (or `next.config.js` `rewrites`):

```json
{ "rewrites": [{ "source": "/burden", "destination": "/burden/index.html" }] }
```

### How clusterfree.org/model actually does it

The live deployment (`aph-research/clusterfree`, a Next.js site on Vercel) keeps
this repo as the single source of truth and pulls the three files at build time:

- `scripts/fetch-model.mjs` downloads `index.html`, `ch_simulation.py`, and
  `webapi.py` from this repo's raw `main` URLs into `public/model/` (run by a
  `prebuild` npm script, and `public/model/` is git-ignored).
- `next.config.js` has a **rewrite** `/model` -> `/model/index.html` so the page
  lives at the clean URL `clusterfree.org/model` (a rewrite, not a redirect, to
  avoid a loop with Next's trailing-slash stripping).
- `index.html` resolves its sibling assets relative to its own directory (see the
  `BASE` constant), so it works served at `/`, `/model`, or `/model/`.

To ship a change: edit here, commit and push to `main`, then redeploy the
clusterfree site (any deploy re-fetches the latest).

## Notes / caveats

- **First load** pulls Pyodide + NumPy from the jsDelivr CDN (~a few MB), then
  it's browser-cached. Subsequent loads are quick. Each simulation runs in a
  couple of seconds in WASM; the Sensitivity tornado uses a smaller sample size
  in-browser (3000 vs 6000 patients) to stay responsive.
- **Content-Security-Policy:** if the host sets a strict CSP, `script-src` must
  allow `https://cdn.plot.ly` (Plotly) and `https://cdn.jsdelivr.net` (Pyodide),
  and `connect-src` must allow `https://cdn.jsdelivr.net` (Pyodide fetches its
  WASM/packages). WASM also needs `'unsafe-eval'` or `'wasm-unsafe-eval'` in
  `script-src`. On clusterfree these are scoped to `/model` in `src/middleware.ts`.
  Without them the browser blocks the charts and/or the Python runtime.
- **Pyodide version** is pinned in `index.html` (`PYODIDE_BASE`, currently
  v0.26.4). Bump it there if you want a newer runtime.
- The CSV export is generated in-browser (a Blob download) in static mode, and
  as a normal file download in local-server mode. Same button either way.

## Other hosting options (not needed for the above)

- **Managed Python host (Render/Railway/Fly):** deploy `server.py` after
  switching its bind to `0.0.0.0` and `os.environ["PORT"]`, and adding a
  `requirements.txt` with `numpy`. Keeps the client/server split; costs money to
  stay always-on and has cold starts on free tiers.
- **Vercel Python Serverless Functions:** put `webapi.dispatch` behind
  `api/*.py` handlers. Works, but watch the Hobby-tier 10s timeout on the
  sensitivity sweep and cold-start NumPy imports. The static/Pyodide route above
  avoids both.
