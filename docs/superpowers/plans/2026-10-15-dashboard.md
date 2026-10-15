# Splash `/dashboard` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auto-refreshing performance dashboard at `GET /dashboard`, showing engine health, an active-phase throughput speedometer, caching, memory, requests, Neural Engine usage, and model information.

**Architecture:** One self-contained HTML page (inline CSS + inline JS, mirroring `chat.html`) served by the existing stdlib HTTP server. The page polls the existing `GET /status` JSON every 2 s and renders it; the only JavaScript dependency is Chart.js, vendored as a committed minified file like the chat page's assets. No backend, `/status`, or inference-path changes.

**Tech Stack:** Python 3.12–3.14 stdlib (`http.server`), HTML/CSS/ES2020, Chart.js 4.5.1 (MIT, UMD, vendored).

**Spec:** `docs/superpowers/specs/2026-10-14-dashboard-design.md`

## Global Constraints

- No new Python dependencies; stdlib only.
- No build step, no bundler, no npm: the page is static files read into the server process at import time, exactly as `chat.html` is.
- Zero runtime network dependencies: every asset is vendored in the repo.
- Exactly one vendored JS library: Chart.js **4.5.1**, downloaded from
  `https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js`
  (208,522 bytes; sha256, base64: `SERKgtTty1vsDxll+qzd4Y2cF9swY9BCq62i9wXJ9Uo=`),
  committed as `server/dashboard-assets/chart-4.5.1.min.js`. License text goes in
  `THIRD_PARTY_NOTICES` following the markdown-it/Highlight.js entries.
- `--no-webui` disables the dashboard along with the chat page.
- The page is public (no key); all data comes from `/status`, which is
  key-protected whenever `--api-key` is set. An unauthenticated visitor must see
  no data.
- Charts must re-theme live on `prefers-color-scheme` changes.
- The `/status` document is partial by contract: any absent section or field
  renders as an n/a chip — never `undefined`, never a crash.
- No changes to the `/status` schema, the Prometheus exporter, or any
  inference path.
- Visual language: the chat page's CSS custom-property system (light +
  `prefers-color-scheme: dark` override) and the `-apple-system` font stack.
- Tests run from the repo root: `python3 -m unittest <module> -v` (CI runs
  `make check-python`).

## File Structure

| File | Responsibility |
| --- | --- |
| `server/dashboard.html` | The page: layout CSS, the poll controller, formatting helpers, panel renderers, Chart.js gauges/sparklines |
| `server/dashboard-assets/chart-4.5.1.min.js` | Vendored Chart.js UMD (the page's only asset) |
| `server/server.py` | Serves the page + asset under the `webui` gate; the public-route set in `parse_request` |
| `dev/tools/package.py` | Packages the page + asset folder with the release |
| `THIRD_PARTY_NOTICES` | Chart.js license |
| `dev/tests/server/test_server_access.py` | Route, auth, webui-gate, asset-exposure tests |
| `dev/tests/tools/test_package.py`, `dev/tests/tools/test_packaged_server.py` | Packaging assertions |
| `DEVELOPMENT.md`, `README.md` | User-facing docs |

The page's JavaScript is one script scope with these extension points, which
later tasks fill in without touching each other's code:

- `state` — `{ key, lastStatus, lastOkAt, lastRestarts, prevCounters, prevPollAt,
  backoff, hidden }` (Tasks 3's `windowedRates` owns `prevCounters`/`prevPollAt`;
  earlier code must not assume they exist)
- `pollOnce()` / `scheduleNext()` — the single poll loop (Task 2)
- `render(status)` — dispatches to `panels.<name>` for each registered panel
- `panels` — object of render functions: `strip` (Task 2), `gauges` (Task 3),
  `cache`, `memory` (Task 4), `requests`, `ane`, `model` (Task 5). Each renders
  into its own `<section id="...">` in the static HTML.
- `fmt = { bytes, num, ms, pct, clock }` — number formatting (Task 2)
- `chip(node, level, text)` with levels `ok|warn|bad|muted`; `na(node)` (Task 2)
- `palette()` — theme colors read from the CSS variables (Task 2)
- `charts` — array of every Chart.js instance; `rethemeAll()` rebuilds them on
  theme change (Task 2 owns the mechanism; Tasks 3–5 register into it)

## Review Focus

Failure modes the spec implies but the automated tests cannot exercise (the
page's JS has no test harness in this repo, as with the chat page). Each is
pinned by a manual-verification step in the owning task:

1. **Partial status document** (a section absent while the engine restarts) →
   the panel shows n/a chips, nothing throws. — Task 2, step 3
2. **Key-protected server** (`--api-key`) → 401 flips up the key banner, no
   data is visible without the key, a wrong key stays empty. — Task 2, step 3
3. **Tab hidden for a long time, then restored** → the jumped poll window
   yields a sane rate, no spike. — Task 2, step 3
4. **Stale status** (`transport.status_stale` true, or a fetch failure) → the
   "updated Ns ago" chip goes stale-colored, last data is kept, nothing fakes
   liveness. — Task 2, step 3
5. **Engine restart mid-watch** (cumulative counters reset to zero,
   `transport.restarts` increments) → windowed diffs never go negative,
   sparkline buffers clear, a "engine restarted" note appears. — Task 3, step 2

---

### Task 1: Serve the dashboard route — page, vendored asset, wiring, packaging, docs

**Files:**
- Create: `server/dashboard.html` (initial stub: real shell, empty panels)
- Create: `server/dashboard-assets/chart-4.5.1.min.js`
- Modify: `server/server.py` (module constants near `CHAT_ASSETS` ~line 104; public set in `parse_request` ~line 333; `do_GET` ~line 542)
- Modify: `dev/tools/package.py` (`SERVER_FILES` ~line 72, `stage_runtime` ~line 91)
- Modify: `THIRD_PARTY_NOTICES`
- Modify: `dev/tests/server/test_server_access.py`
- Modify: `dev/tests/tools/test_package.py`, `dev/tests/tools/test_packaged_server.py`
- Modify: `DEVELOPMENT.md` (three statements + route table), `README.md` is deferred to Task 5

**Interfaces:**
- Produces: `server.DASHBOARD_HTML` (bytes), `server.DASHBOARD_ASSETS` (dict, key `/dashboard-assets/chart-4.5.1.min.js` → bytes); routes `GET/HEAD /dashboard` and the asset, public, 404 under `--no-webui`.
- Consumes: nothing from later tasks.

- [ ] **Step 1: Create the files the import needs**

Create `server/dashboard-assets/` and `server/dashboard.html` (a stub for now:
doctype, `<title>Splash Dashboard</title>`, the favicon link,
`<script src="/dashboard-assets/chart-4.5.1.min.js"></script>`, a minimal
`<style>` with the chat page's CSS variables, a `<header>` with the brand, and
an empty `<main>` containing a "Loading" placeholder). This file is replaced
in place by Tasks 2–5; only the shell exists yet.

- [ ] **Step 2: Vendor Chart.js 4.5.1 with a checksum gate**

```bash
mkdir -p server/dashboard-assets
curl -fsSL -o server/dashboard-assets/chart-4.5.1.min.js \
  https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js
python3 - <<'EOF'
import base64, hashlib, pathlib
data = pathlib.Path("server/dashboard-assets/chart-4.5.1.min.js").read_bytes()
assert len(data) == 208522, len(data)
assert base64.b64encode(hashlib.sha256(data).digest()).decode() == "SERKgtTty1vsDxll+qzd4Y2cF9swY9BCq62i9wXJ9Uo="
print("chart.js 4.5.1 verified")
EOF
```

Expected: `chart.js 4.5.1 verified`. If the checksum differs, stop and re-download — never commit a mismatching vendor file. Also download `https://cdn.jsdelivr.net/npm/chart.js@4.5.1/LICENSE.md` to a temp location for Step 5's notice (do not commit it; `THIRD_PARTY_NOTICES` is the license record).

- [ ] **Step 3: Write the failing route tests**

Add to `dev/tests/server/test_server_access.py` (follow the shapes of
`test_public_probes_and_optional_webui` and
`test_chat_assets_are_public_but_do_not_expose_other_files`):

```python
    def test_dashboard_page_is_public_but_gated_like_the_chat_page(self):
        disabled = self.harness(api_key="test-server-key", webui=False)
        for method in ("GET", "HEAD"):
            self.assertEqual(disabled.request(method, "/dashboard")[0], 404)
            self.assertEqual(
                disabled.request(
                    method, "/dashboard-assets/chart-4.5.1.min.js"
                )[0],
                404,
            )
        default = self.harness()
        status, content_type, html = default.request("GET", "/dashboard")
        self.assertEqual((status, content_type), (200, "text/html; charset=utf-8"))
        self.assertIn(b"Splash Dashboard", html)
        self.assertIn(b"/dashboard-assets/chart-4.5.1.min.js", html)
        protected = self.harness(api_key="test-server-key")
        status, _, html = protected.request("GET", "/dashboard")
        self.assertEqual(status, 200)
        self.assertNotIn(b"test-server-key", html)
        self.assertEqual(protected.request("HEAD", "/dashboard")[0], 200)

    def test_dashboard_assets_are_public_but_do_not_expose_other_files(self):
        harness = self.harness(api_key="test-server-key")
        for route in server.DASHBOARD_ASSETS:
            with self.subTest(route=route):
                status, content_type, payload = harness.request("GET", route)
                self.assertEqual(status, 200)
                self.assertEqual(content_type, "text/javascript; charset=utf-8")
                self.assertEqual(
                    payload, (server.ROOT / "server" / route[1:]).read_bytes()
                )
                self.assertEqual(harness.request("HEAD", route)[0], 200)
        key = {"Authorization": "Bearer test-server-key"}
        for path in ("/dashboard-assets/../server.py", "/dashboard-assets/missing.js"):
            self.assertEqual(harness.request("GET", path, headers=key)[0], 404)
```

Extend two existing tests:
- In `test_public_probes_and_optional_webui`, add `"/dashboard"` and
  `*server.DASHBOARD_ASSETS` to the `webui=False` 404 loop.
- In `test_routes_and_auth_use_one_normalized_path`, add:
  `self.assertEqual(harness.request("GET", "/./dashboard")[0], 200)` and
  `self.assertEqual(harness.request("GET", "/dashboard/../dashboard")[0], 200)`.

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python3 -m unittest dev.tests.server.test_server_access -v`
Expected: FAIL — `AttributeError: module 'server.server' has no attribute 'DASHBOARD_ASSETS'` (import-level, since the test references it); the route tests do not run.

- [ ] **Step 5: Wire the route in `server/server.py`**

Three edits, mirroring the chat page:

```python
# After the CHAT_ASSETS block (~line 113):
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html").read_bytes()
DASHBOARD_ASSETS = {
    f"/dashboard-assets/{name}": (
        Path(__file__).parent / "dashboard-assets" / name
    ).read_bytes()
    for name in ("chart-4.5.1.min.js",)
}
```

```python
# In parse_request (~line 333) — the public-vs-authenticated split:
            public = self.command == "OPTIONS" or (
                self.command in ("GET", "HEAD")
                and (
                    self.route
                    in (
                        "/",
                        "/index.html",
                        "/favicon.ico",
                        "/health",
                        "/ready",
                        "/dashboard",
                    )
                    or self.route in CHAT_ASSETS
                    or self.route in DASHBOARD_ASSETS
                )
            )
```

```python
# In do_GET (~line 542) — the webui gate:
        if (
            path in ("/", "/index.html", "/favicon.ico", "/dashboard")
            or path in CHAT_ASSETS
            or path in DASHBOARD_ASSETS
        ):
            if not self.server.webui:
                self._safe_error(APIError(404, "not found", "not_found"))
            elif path == "/favicon.ico":
                self._send(200, FAVICON_SVG, "image/svg+xml")
            elif path in CHAT_ASSETS:
                mime = "text/css" if path.endswith(".css") else "text/javascript"
                self._send(200, CHAT_ASSETS[path], f"{mime}; charset=utf-8")
            elif path in DASHBOARD_ASSETS:
                self._send(200, DASHBOARD_ASSETS[path], "text/javascript; charset=utf-8")
            elif path == "/dashboard":
                self._send(200, DASHBOARD_HTML, "text/html; charset=utf-8")
            else:
                self._send(200, CHAT_HTML, "text/html; charset=utf-8")
            return
```

- [ ] **Step 6: Run the route tests to verify they pass**

Run: `python3 -m unittest dev.tests.server.test_server_access -v`
Expected: PASS, including the two new tests and the two extended ones.

- [ ] **Step 7: Add the failing packaging expectations**

In `dev/tests/tools/test_package.py`: in `test_build_needs_no_hub_access_and_never_ships_credentials` and `test_release_has_only_runtime_files_and_excludes_credentials`, add `("server/dashboard-assets", package.DASHBOARD_ASSET_FILES),` to the fixture-folder loop; in `test_release_has_only_runtime_files...` change the staged-`server` assertion to `{*package.SERVER_FILES, "chat-assets", "dashboard-assets"}` and add a `dashboard-assets` content check mirroring the `chat-assets` one (names equal `set(package.DASHBOARD_ASSET_FILES)`).
In `dev/tests/tools/test_packaged_server.py`: add the same `("server/dashboard-assets", package.DASHBOARD_ASSET_FILES),` entry to `stage_release`'s copy loop.

- [ ] **Step 8: Make the packaging real**

In `dev/tools/package.py`: add `"dashboard.html",` to `SERVER_FILES` (after `"chat.html"`), add the `DASHBOARD_ASSET_FILES = ("chart-4.5.1.min.js",)` tuple next to `CHAT_ASSET_FILES`, and add `("server/dashboard-assets", DASHBOARD_ASSET_FILES),` to `stage_runtime`'s folder loop (after the chat-assets entry).
In `THIRD_PARTY_NOTICES`: add a Chart.js entry modeled on the markdown-it one — name/version/repo URL, the sentence "The dashboard uses the vendored browser bundle in `server/dashboard-assets/chart-4.5.1.min.js`.", and the MIT license text from the downloaded `LICENSE.md` (Chart.js is MIT; copyright line as it appears there).

- [ ] **Step 9: Run the packaging tests to verify they pass**

Run: `python3 -m unittest dev.tests.tools.test_package dev.tests.tools.test_packaged_server -v`
Expected: PASS.

- [ ] **Step 10: Fix the user-facing docs' auth-surface statements**

In `DEVELOPMENT.md`, three existing statements become false once `/dashboard` is public; update each (locate by these anchor phrases, preserving the surrounding prose):
- the `--api-key` row: "on every route but the chat page, `/health` and `/ready`" → name the dashboard alongside the chat page
- the `--no-webui` row: "Disable the chat page." → "Disable the chat page and the dashboard."
- the Network-and-authentication prose: "only the chat page, … stay public" → include the dashboard
Add a route-table row after the `/status`, `/metrics` row (~line 596):
`| GET /dashboard | The dashboard page (its vendored asset at /dashboard-assets/…); public, and gone with --no-webui |`
The detailed "Dashboard" section is written in Task 5, once the panels exist.

- [ ] **Step 11: Run the full Python suite**

Run: `make check-python` (or `python3 -m unittest discover -s dev/tests -t .` from the repo root)
Expected: PASS — nothing else regressed.

- [ ] **Step 12: Commit**

```bash
git add server/dashboard.html server/dashboard-assets server/server.py \
  dev/tools/package.py THIRD_PARTY_NOTICES \
  dev/tests/server/test_server_access.py dev/tests/tools/test_package.py \
  dev/tests/tools/test_packaged_server.py DEVELOPMENT.md
git commit -m "serve: dashboard route, vendored Chart.js, and packaging

Adds GET /dashboard (public like the chat page, gated by --no-webui)
and its single vendored asset under /dashboard-assets. The page is a
shell for now; the panels land in follow-up commits. Chart.js 4.5.1
(MIT) is vendored and listed in THIRD_PARTY_NOTICES."
```

---

### Task 2: Page core — poll controller, key UX, status strip, formatting, defensive rendering

**Files:**
- Modify: `server/dashboard.html` (the whole page body; the Task 1 stub becomes the real skeleton)

**Interfaces:**
- Consumes: `server.DASHBOARD_HTML` + the `/status` JSON (spec data-contract table); the `/dashboard` route and asset from Task 1.
- Produces (the extension points the File Structure block lists): `state`, `pollOnce()`, `scheduleNext()`, `render(status)`, `panels.strip`, `fmt.{bytes,num,ms,pct,clock}`, `chip(node, level, text)`, `na(node)`, `palette()`, `charts[]`, `rethemeAll()`. Tasks 3–5 register `panels.gauges` / `panels.cache` / `panels.memory` / `panels.requests` / `panels.ane` / `panels.model` and register their charts in `charts`.

- [ ] **Step 1: Write the page core**

Replace the stub with the real shell and controller:

- **HTML**: a `<header>` (brand, `#model-name`, `#key-form` hidden by default, `#strip` chip row, `#updated` "updated Ns ago" chip) and a `<main class="grid">` with the six `<section class="card" id="gauges|cache|memory|requests|ane|model">` blocks — each with its heading and a placeholder body the owning task replaces. Add the CSS: the chat page's variable set plus `--ok`, `--warn`, `--bad` semantic colors (light + dark values), card/chip/placeholder styles, the responsive grid (two columns, stacking below ~900 px).
- **`pollOnce()`**: `fetch('/status', {headers: key ? {Authorization: 'Bearer ' + key} : {}})`; on **401** show `#key-form` and stop; on **2xx** parse and `render(status)`, reset backoff, record `lastOkAt`; on any failure bump the backoff (2 s → 5 s → 10 s, capped). `scheduleNext()` picks the interval: 2 s when `!document.hidden`, 10 s when hidden, the backoff value after failures; a single `setTimeout` chain (no overlapping in-flight polls — ignore a response whose request is superseded). A 1 s ticker updates `#updated`'s age and the stale chip (stale = `now - lastOkAt > 10 s`, or `status.transport.status_stale`). Until the first successful `/status` arrives — or while `status.ready === false` — the strip shows a "waiting for engine" chip; panels keep their placeholders and never render fake zeros.
- **`#key-form`**: a password input (in-memory only — never persisted, the chat page's convention) + apply button; on apply set `state.key`, hide the form, `pollOnce()`.
- **`render(status)`**: defensive by construction — every panel reads through `status && status.<section>` and missing fields become `na()` chips; dispatches to each registered `panels.*`; updates `#model-name` from `status.instance.model` and the strip via `panels.strip`: chips for `ready` (ok/bad), `thermal_state` (nominal ok, fair warn, serious/critical bad), `memory_pressure` (normal ok, warning warn, critical bad), `metal.healthy` (+`failure_reason` in a title tooltip), `transport` (ready ok / recovering warn / stopped bad), and `loop.max_tick_ms` as a number chip.
- **`fmt`**: `bytes` (B/KiB/MiB/GiB, binary, one decimal), `num` (1.2K/1.2M), `ms` (123 ms / 1.2 s), `pct` (0.85 → 85%), `clock` (seconds → 23m 05s / 1h 23m).
- **Theme**: `palette()` reads `--text/--muted/--link/--border/--ok/--warn/--bad` via `getComputedStyle`; a `matchMedia('(prefers-color-scheme: dark)')` listener calls `rethemeAll()` (rebuilds each chart in `charts` with the new palette).

- [ ] **Step 2: Syntax-check the page JavaScript**

Extract the inline script and parse it so a syntax error fails the step (the repo has no JS harness; a parse check is the floor):

```bash
python3 - <<'EOF'
import pathlib, re, shutil, subprocess, sys, tempfile, os
html = pathlib.Path("server/dashboard.html").read_text()
js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
if not shutil.which("node"):
    print("node unavailable — skipping parse check"); sys.exit(0)
with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
    f.write(js); path = f.name
result = subprocess.run(["node", "--check", path], capture_output=True)
os.unlink(path)
sys.exit(result.returncode)
EOF
```

Expected: exit 0 (or the printed skip when node is absent).

- [ ] **Step 3: Manual verification against a live engine (Review Focus 1–4)**

Start the server from the checkout on a free port with the model already
downloaded on this Mac (e.g. `./splash serve --model mlx-community/Qwen3.8-27B-4bit
--port 8100` from the repo root — the engine binary must be built; `make` first if
not), and in a browser:

1. Open `http://127.0.0.1:8100/dashboard`: the strip shows real values, the "updated Ns ago" chip ticks, panels show n/a placeholders (no panels are built yet), no console errors.
2. **Partial document (Focus 1):** in the DevTools console, take the live status and call `render` on it with whole sections removed (e.g. `delete s.cache; delete s.memory_governor; render(s)`) → the affected panels show n/a chips, no exception.
3. **Key-protected (Focus 2):** start a second instance with `--api-key test-dash` on another port → the dashboard shows only the key banner and no data; a wrong key keeps it empty; the correct key fills the page.
4. **Hidden tab (Focus 3):** hide the tab ≥ 30 s, restore it → the next poll window is long but renders a sane "updated" age and no rate spike (the gauges task will see the same window; the controller must not treat it as an error).
5. **Stale (Focus 4):** in the console `render({...state.lastStatus, transport: {...state.lastStatus.transport, status_stale: true}})` → the stale chip shows; also confirm a killed server (or `--port` where nothing listens) produces the error/backoff state, not a fake "live".

- [ ] **Step 4: Commit**

```bash
git add server/dashboard.html
git commit -m "serve: dashboard page core — poll controller, key UX, status strip

The page polls /status every 2 s (10 s hidden, capped backoff on
failure), shows the engine's ready/thermal/pressure/Metal/transport
state, and renders any partial document as n/a chips. Charts and
panels land in the next commits."
```

---

### Task 3: Speedometer, draft-acceptance gauge, TTFT/ITL sparklines

**Files:**
- Modify: `server/dashboard.html` (the `#gauges` section body + the Chart.js code)

**Interfaces:**
- Consumes: `render` dispatch, `state`, `palette()`, `charts`, `fmt`, `na` from Task 2.
- Produces: `panels.gauges(status)`; `makeGauge(canvas, { label, unit, mode })` → `{ set(value, max) }`; `makeSparkline(canvas, colorKey)` → `{ push(v) }`; `windowedRates(status)` → `{ prefill, decode }` tok/s for the current poll window (0 when idle); `niceMax(x)` → 1/2/5·10ⁿ rounding.

- [ ] **Step 1: Write the gauge code**

The `#gauges` card gets: the main speedometer (a Chart.js doughnut, 240° arc — `rotation: 240, circumference: 240` — value segment in the accent color over a track, plus an inline custom plugin drawing the needle from the center at the value fraction and the value+unit in the arc's mouth), a smaller draft-acceptance doughnut (0–100 %), two sparkline canvases (TTFT, ITL), and two lifetime-average tiles (`prefill/decode tokens/s since engine start`, from `metrics.*_tokens_per_second`).

- **`windowedRates(status)`** — the owner's active-phase decision, made mechanical:

```js
function windowedRates(status) {
  const m = (status && status.metrics) || {};
  const t = (status && status.transport) || {};
  const now = Date.now();
  const out = { prefill: 0, decode: 0 };
  if (state.prevCounters) {
    const dt = (now - state.prevPollAt) / 1000;
    const restarted = t.restarts !== state.lastRestarts;
    if (dt > 0 && dt < 120) {
      const dPre = (m.prefill_input_tokens ?? 0) - state.prevCounters.prefill;
      const dDec = (m.decode_output_tokens ?? 0) - state.prevCounters.decode;
      if (!restarted && dPre >= 0 && dDec >= 0) {
        out.prefill = dPre / dt;
        out.decode = dDec / dt;
      }
    }
  }
  state.prevCounters = {
    prefill: m.prefill_input_tokens ?? 0,
    decode: m.decode_output_tokens ?? 0,
  };
  state.prevPollAt = now;
  state.lastRestarts = t.restarts ?? state.lastRestarts;
  return out;
}
```

(the caller stores the last 3 windows in a ring and the needle shows their mean; on a detected restart the rings clear and the "engine restarted" note appears).

- **`panels.gauges(status)`**: rates via `windowedRates`; active phase from `scheduler.decoding` / `scheduler.prefilling` (decode wins when both are nonzero; both zero → idle: needle 0, label "idle"); the needle tracks the active phase's rate; gauge scale = `niceMax` of the values seen in that mode since the last restart, with a floor (e.g. 10 tok/s) and an eased scale change on mode switch. Acceptance gauge: `metrics.draft_acceptance_rate` (n/a when `drafted_tokens` is 0/absent), with drafted/accepted totals and tok/cycle (`1 + accepted/cycles`) beside it. Sparklines: push `metrics.ttft_ms.p50/p95` and `metrics.itl_ms.p50/p95` (n/a-aware), 150-sample rings, `y.max = niceMax(current ring)` , `chart.update('none')`.

- [ ] **Step 2: Verify against the live engine (Review Focus 5)**

Against the Task 2 server:
1. Idle: needle at 0, label "idle". Fire a long prompt at the API (`curl -s .../v1/chat/completions` with a big prompt) → the needle climbs in **prefill** mode, flips to **decode** while the response streams, and settles back to 0; the scale eases rather than snapping; the lifetime tiles and acceptance gauge move; the TTFT/ITL sparklines grow.
2. **Restart guard (Focus 5):** in the console, feed `render` a document whose `metrics` counters are smaller than the last seen and `transport.restarts` incremented by 1 → `windowedRates` yields 0 (no negative spike), the rings clear, and the "engine restarted" note appears; then feed a normal document and confirm the needles resume.
3. `node --check` the page JS as in Task 2, step 2.

- [ ] **Step 3: Commit**

```bash
git add server/dashboard.html
git commit -m "serve: dashboard speedometer, acceptance gauge, and sparklines

One speedometer tracks the engine's active phase — prefill tok/s while
prefilling, decode tok/s while decoding — from windowed diffs of the
engine's counters, plus the draft-acceptance gauge and TTFT/ITL
history. Engine restarts reset the windows and note the restart."
```

---

### Task 4: Cache and memory panels

**Files:**
- Modify: `server/dashboard.html` (the `#cache` and `#memory` section bodies + their renderers)

**Interfaces:**
- Consumes: `render` dispatch, `state`, `palette()`, `fmt`, `na` from Task 2.
- Produces: `panels.cache(status)`, `panels.memory(status)`.

- [ ] **Step 1: Write the panels**

`#cache`: a hit-rate donut (`cache.hit_rate`, with hits/cold-misses counts), counters for `reused_tokens` and `deduplicated_state_publications` (spec: the deduplicated-tokens figure), a stacked bar for KV pages (active / cached / free of allocated, with `allocated_bytes` and `reclaimable_bytes` in GiB), recurrent state (`state.entries`, `state.bytes` GiB, checkpoint entries/bytes), SSD disk cache (`used_bytes`/`capacity_bytes` bar, `volume_available_bytes`, `write_budget_bytes`, the `writes_gated` state as a chip, `read_bytes`/`written_bytes` totals), and four small tiles for the process caches (`tokenizer_cache`: entries, hits, reused tokens; `image_cache`, `grammar_cache`, `response_store`: entries + bytes).

`#memory`: a bar of `memory_governor.charged_bytes` against `limit_bytes` with `headroom_bytes` labeled, `memory_actual.current_bytes`/`peak_bytes`, `host_available_bytes`, `denied_reservations` (warning chip when > 0), and the weights row (`weights.released/restoring/restores/restore_failures` — "released"/"in memory"/"restoring" chip + idle-release seconds).

- [ ] **Step 2: Verify against the live engine**

Cross-check the rendered values against `curl -s http://127.0.0.1:8100/status` for the same moment (spot: hit rate, KV GiB, state GiB, disk used/capacity, charged/limit GiB). Then exercise a cache event: send the same prompt twice (second should show a higher hit rate / reused tokens) and confirm the counters move. Partial-document behavior: `render` with `kv`/`state`/`disk` removed → n/a chips only. `node --check` as before.

- [ ] **Step 3: Commit**

```bash
git add server/dashboard.html
git commit -m "serve: dashboard cache and memory panels

KV pages, recurrent state, SSD cache, and the tokenizer/image/grammar/
response caches; Metal usage against the governor with weights-release
state. Every figure reads straight from the status document."
```

---

### Task 5: Requests, Neural Engine, model card — plus docs and final verification

**Files:**
- Modify: `server/dashboard.html` (the `#requests`, `#ane`, `#model` section bodies + their renderers)
- Modify: `DEVELOPMENT.md` (the "Dashboard" section), `README.md` (quick-start line)

**Interfaces:**
- Consumes: everything from Tasks 2–4.
- Produces: `panels.requests(status)`, `panels.ane(status)`, `panels.model(status)`.

- [ ] **Step 1: Write the panels**

`#requests`: the four lifecycle counters (`requests.submitted/completed/cancelled/failed`), the active-requests table (one row per `active_requests[]`: id, phase as a colored chip — queued / waiting_* / prefill / decode —, priority, prompt tokens with a progress bar of `prompt_processed`/`prompt_tokens`, `generated_tokens` of `max_new_tokens`, `age_ms` ticking), scheduler counts (queued / prefilling / decoding / waiting_*), and the decode batch-width mini-bar (`decode_batches_by_width.b1..b4`).

`#ane` (the spec's derived figures, all guarded against zero denominators): the state chip (`split` ok / `off` muted / `stopped` bad, with `reason` as the caption), the channel share as a small dial or bar (`share` %), "chunks of ≥ `minimum_rows` rows", the cumulative counters (`split_commands`, `evaluations`, `ane_ms` via `fmt.ms`), and the derived values: ANE share of prefill time = `ane_ms / model_timing.prefill.total_wall_ms` (bar), mean ms/evaluation, evaluations per command, rerun rate (warning-colored when > 0).

`#model`: `instance.model` (prominent), `host:port`, pid, short instance id, uptime (`now - started_at` via `fmt.clock`), `maximum_context_tokens` (`fmt.num`), `input_modalities` chips, KV dtype (`identity.cache.dtype`) + block tokens, the device row (`memory_plan.device`: name, GPU cores, physical memory), `chat_template.later_system` (handle both the scalar and the `{name: value}` dict shapes), and `schema_version` in the corner.

- [ ] **Step 2: Finish the docs**

`DEVELOPMENT.md`: add a short "Dashboard" subsection (where the page lives, that it polls `/status`, that `--no-webui` hides it and `--api-key` gates its data via the in-page key entry, that its figures are the status document's). `README.md` quick start: one sentence after the "Open http://127.0.0.1:8000" line — the chat page at `/`, the performance dashboard at `/dashboard`.

- [ ] **Step 3: Full manual verification pass**

Against the live engine: every panel populated and consistent with `/status`; send a request with an image (or a tool-heavy one) and watch requests/scheduler/ANE move together; toggle the OS appearance and confirm all charts re-theme; reload the page on a `--api-key` instance and re-enter the key; confirm the `--no-webui` instance 404s the page (curl) and that `/` still serves the chat page everywhere.

- [ ] **Step 4: Run the full Python suite**

Run: `make check-python`
Expected: PASS (the page is static; the suite covers the route and packaging).

- [ ] **Step 5: Commit**

```bash
git add server/dashboard.html DEVELOPMENT.md README.md
git commit -m "serve: dashboard requests, Neural Engine, and model panels

Completes the dashboard: the request queue, the ANE split's usage and
derived prefill-time share, and the model card. Documents the page in
DEVELOPMENT.md and the README quick start."
```
