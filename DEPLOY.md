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

### Keeping the deployed copy in sync

The three files are the *entire* deployable bundle. When you change the model or
UI in this repo, re-copy them into `public/burden/`. A simple script or a small
CI step in the Next.js repo can `curl` them from this repo's raw GitHub URLs so
you never hand-copy.

## Notes / caveats

- **First load** pulls Pyodide + NumPy from the jsDelivr CDN (~a few MB), then
  it's browser-cached. Subsequent loads are quick. Each simulation runs in a
  couple of seconds in WASM; the Sensitivity tornado uses a smaller sample size
  in-browser (3000 vs 6000 patients) to stay responsive.
- **Content-Security-Policy:** if clusterfree.org sets a strict CSP, allow
  `cdn.jsdelivr.net` in `script-src` (and WASM: add `'wasm-unsafe-eval'` to
  `script-src`). Without this the browser will block Pyodide.
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
