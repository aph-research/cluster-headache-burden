"""
Local web server for the cluster-headache dashboard.

A zero-dependency stdlib HTTP server (only numpy, via ch_simulation) that serves
index.html and exposes the shared compute layer (webapi.dispatch) as a small JSON
/ CSV API. This is for LOCAL development; the public build (clusterfree.org/burden)
runs the same webapi.dispatch in-browser via Pyodide, no server needed.

  - GET /                        -> index.html
  - GET /api/config              -> defaults + slider metadata
  - GET /api/simulate?lever=..   -> run the sim, JSON results
  - GET /api/sensitivity?..      -> tornado JSON
  - GET /api/cost_effectiveness  -> cost-effectiveness JSON
  - GET /api/export[_counterfactual] -> CSV download

Usage:
    python3 server.py            # then open http://localhost:8000
    python3 server.py 8080       # custom port
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from webapi import dispatch

HERE = Path(__file__).parent
INDEX = HERE / "index.html"

# path -> endpoint name (JSON responses)
JSON_ROUTES = {
    "/api/config": "config",
    "/api/simulate": "simulate",
    "/api/sensitivity": "sensitivity",
    "/api/sensitivity_ce": "sensitivity_ce",
    "/api/cost_effectiveness": "cost_effectiveness",
    "/api/cost_effectiveness_funnel": "cost_effectiveness_funnel",
}
# path -> (endpoint name, download filename) (CSV responses)
CSV_ROUTES = {
    "/api/export": ("export", "ch_simulations.csv"),
    "/api/export_counterfactual": ("export_counterfactual", "ch_counterfactual.csv"),
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/demo":  # temporary: cost-effectiveness layout comparison
            self._send(200, (HERE / "ce_layouts_demo.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif u.path in JSON_ROUTES:
            try:
                self._json(dispatch(JSON_ROUTES[u.path], params))
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, code=400)
        elif u.path in CSV_ROUTES:
            try:
                endpoint, fname = CSV_ROUTES[u.path]
                body = dispatch(endpoint, params).encode()
                self._send(200, body, "text/csv; charset=utf-8",
                           {"Content-Disposition": f'attachment; filename="{fname}"'})
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, code=400)
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *a):  # quieter console
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CH simulation UI -> http://localhost:{port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
