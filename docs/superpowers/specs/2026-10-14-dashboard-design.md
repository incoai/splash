# Splash `/dashboard` — Design

Date: 2026-10-14
Status: awaiting owner approval

## Intent and why

Splash is a local inference engine. Today the user sees engine state only in four places.

- per-request lines on the server console
- the raw `/status` JSON
- the Prometheus text of `/metrics`
- external Prometheus/Grafana installs

When the owner asked "is there an admin panel?" the honest answer was no.

This adds a self-contained web dashboard at `GET /dashboard`. The existing HTTP server serves it. A user can leave it open in a browser tab. The page then shows: engine health, throughput (speedometer-style), caching behavior, request queue, Neural Engine (ANE) usage, model information. The page makes the engine legible without any external tooling.

## Goals

1. Auto-refresh: poll `GET /status` every 2 s. No user action.
2. One main speedometer for engine throughput. It tracks the active phase: prefill tok/s while the engine prefills, decode tok/s while it decodes. The needle falls to zero when the engine idles. Plus a draft-acceptance gauge for the speculative decoder.
3. Rolling TTFT and ITL sparklines (p50/p95).
4. Cache metrics: KV cache, recurrent state, SSD disk cache, the small process caches (tokenizer, image, grammar, response store).
5. ANE panel: split state, channel share, min chunk, cumulative counters, derived utilization, reruns. (Explicit owner request.)
6. Live request view: lifecycle counters, active-request table with phases, scheduler queues, decode batch widths.
7. Model card: identity, device, context, modalities, KV dtype, build.
8. Status strip: ready, thermal, memory pressure, Metal, engine transport.
9. Works with and without `--api-key`. Works with and without `--no-webui`.
10. Consistent with the chat page visual language: system font, the existing light/dark CSS variable system, GitHub-ish neutrals.
11. Zero runtime network dependencies. The repo vendors every asset.
12. Ships in packaged releases, exactly as `chat.html` does.
13. Read-only: the dashboard shows data. It changes no engine behavior and no request handling.

## Non-goals (v1)

- Control actions (a "release weights" button, pause, etc.). The API exists (`POST /v1/memory/release`). A control surface is a later version.
- History persistence or multi-server aggregation. Sparklines are in-memory ring buffers in the page. The page stores nothing.
- Replacing Prometheus: `/metrics` stays the scraping path. The dashboard is the human path.
- Per-request drill-down: per-request token/latency detail already rides in the API responses.
- Mobile-first layout: the page must degrade gracefully when stacked. It targets a desktop tab.
- New backend fields: the dashboard consumes the existing `/status` document only. No changes to the status schema, the Prometheus exporter, or any inference path.

## Anti-goals (failures even if it works)

- The engine must not feel the polling load. `/status` answers from the native status document the loop writes each tick, plus cheap Python-side stats. The page must back off on error. It must slow down when the tab is hidden. A parked tab never hammers a busy server.
- No new information exposure. The page is public, exactly like the chat page. Every number it shows comes from `/status`. The server protects that endpoint once the user sets `--api-key`. A key-protected server shows an unauthenticated visitor only a "this server needs an API key" state. Not data.
- No new runtime dependency on the network. The page must render fully from the local server, on a Mac with no internet.
- No new build step: no bundler, no npm, no compile. The server process reads the page as static files at import time, as `chat.html` is.

## Constraints

- Python stdlib HTTP server. No new Python dependencies.
- One vendored JS library. Commit it minified, like `server/chat-assets/*`. Name it with its version. List it in `dev/tools/package.py`. Put its license text in `THIRD_PARTY_NOTICES`.
- The chat page proves the visual convention: `-apple-system` font stack, CSS custom properties with a `prefers-color-scheme` dark override, small hand-rolled layout CSS. The dashboard follows it.
- Routes must pass the existing security machinery: path normalization, origin validation, the public-vs-authenticated split in `server/server.py`.
- Chart theming must follow live `prefers-color-scheme` changes. The page re-themes charts when the OS flips.

## Decisions made with the owner

1. **Path: `/dashboard`.** Owner's choice (over `/admin`, `/stats`).
2. **Gating: the existing `--no-webui` flag.** It disables the dashboard with the chat page. The owner chose sharing over a separate flag.
3. **Libraries are welcome.** Owner: "you can use jquery or other libraries (tailwind/whatever), don't reinvent things insecurely." The design takes that as permission to use established libraries rather than hand-rolled graphics code.
4. **Scope:** status strip, throughput gauges, TTFT/ITL sparklines, cache, requests/scheduler, model card, ANE panel. The owner added the ANE panel after the initial scope.
5. **The speedometer tracks the active phase.** The engine's own throughput fields are lifetime cumulative averages that barely move. The designer asked the owner what the needles should track. Owner's answer: "prefill tokens/sec, then switch to decode tokens/sec." One instrument for the phase the engine is in right now. Not two static dials.

## Design calls (mine, owner may veto)

- **Charting: Chart.js** (MIT, single UMD file). It draws the radial gauges (doughnut with partial circumference) and the sparklines (line charts). Vanilla JS handles polling, DOM, number formatting. No jQuery or CSS framework. The chat page variable system is the design language. The layout is ~100 lines of grid/flex CSS.
- **Polling behavior:** 2 s while the tab is visible. 10 s while `document.hidden`. On a fetch error or HTTP failure, back off 2 s → 5 s → 10 s (capped) until a success. The "last updated N s ago" chip turns stale-colored past 10 s. The page detects engine restarts via a `transport.restarts` increment (and status counters resetting). On restart, sparkline buffers clear. A "engine restarted" note appears.
- **Needle source (mechanics of the owner's phase choice):** the page diffs the cumulative token counters (`metrics.prefill_input_tokens`, `metrics.decode_output_tokens`) between consecutive polls. It divides by the window wall time. It smooths with the mean of the last ~3 windows. The needle shows the rate a user would observe right now. It rests at zero while the engine idles. The page reads the active phase from `scheduler.prefilling`/`scheduler.decoding`. It falls back to the newest valid `metrics.current_*_batch`. A `transport.restarts` increment guards the diffs. The lifetime averages (`metrics.prefill_tokens_per_second`, `decode_tokens_per_second`) remain as small "since engine start" tiles.
- **Gauge scaling:** the scale is the nice-rounded maximum (1/2/5·10ⁿ rounding) of the values seen in the active mode since the last restart. A small floor keeps a light load readable on the dial. The scale eases to the new mode's scale when the phase switches.
- **API key UX:** the page tries `/status` bare. On 401 it shows a key input in the header (type=password, in-memory only, never persisted — the chat page's existing convention). While unauthenticated, all panels show an empty "needs API key" state. Not zero data.
- **Derived ANE figures** (computed client-side from existing fields):
  - ANE share of prefill time = `ane_ffn.ane_ms` ÷ `model_timing.prefill.total_wall_ms`
  - mean ms per evaluation = `ane_ffn.ane_ms` ÷ `ane_ffn.evaluations`
  - evaluations per split command = `evaluations` ÷ `split_commands`
  - rerun rate = `reruns` ÷ `split_commands` (warning-colored when > 0)
- **Number formatting:** tabular numerals. Bytes as GiB/MB. Tokens as K/M. Milliseconds as ms or s. All formatting lives in one small helper module in the page.
- **Layout:** a responsive CSS grid. Status strip full-width. Gauge row (decode, prefill, acceptance, TTFT+ITL sparklines card). Cache + memory row. Requests + ANE row. Model card full-width at the bottom. Below ~900 px it stacks to one column.
- **File shape:** `server/dashboard.html` (inline CSS + inline JS, mirroring `chat.html`) plus `server/dashboard-assets/chart-<ver>.min.js` (vendored). The page is the only new route surface. It reuses `/favicon.ico`.

## Data contract (from `GET /status`)

No field changes. The panels read:

| Panel | Fields |
| --- | --- |
| Status strip | `ready`, `thermal_state`, `memory_pressure`, `metal.healthy` + `failure_reason`, `transport.ready/recovering/stopped/restarts/status_stale/status_age_ms`, `loop.max_tick_ms` |
| Speedometer | `scheduler.prefilling`, `scheduler.decoding` (active phase), `metrics.prefill_input_tokens`, `metrics.decode_output_tokens` (windowed diffs), `metrics.current_prefill_batch`, `metrics.current_decode_batch` (recency fallback, shown as last-batch rates), `metrics.prefill_tokens_per_second`, `metrics.decode_tokens_per_second` (lifetime tiles), `metrics.draft_acceptance_rate`, `metrics.drafted_tokens`, `metrics.accepted_draft_tokens` (acceptance gauge) |
| Sparklines | `metrics.ttft_ms.{p50,p95}`, `metrics.itl_ms.{p50,p95}` (ring buffers client-side) |
| Cache | `cache.{hits,cold_misses,hit_rate,kv_hit_tokens,kv_disk_hit_tokens,reused_tokens,deduplicated_state_publications,lazy_junctions,…}`, `kv.{pages_allocated,pages_active,pages_cache,pages_free,allocated_bytes,reclaimable_bytes,block_tokens}`, `state.{entries,in_use,bytes,allocated_bytes,checkpoint_*}`, `disk.{capacity_bytes,limit_bytes,used_bytes,volume_available_bytes,write_budget_bytes,read_bytes,written_bytes,writes_gated,kv_blocks,kv_bytes,kv_demotions,kv_restores,taken_back,write_behind}` |
| Memory | `memory_actual.{current_bytes,peak_bytes}`, `memory_governor.{limit_bytes,charged_bytes,headroom_bytes,denied_reservations,system_pressure,host_available_bytes}`, `weights.{released,restoring,restores,restore_failures}` |
| Requests | `requests.{submitted,completed,cancelled,failed}`, `active_requests[]` (id, phase, priority, prompt_tokens, prompt_processed, generated_tokens, max_new_tokens, age_ms), `scheduler.{queued,waiting_resources,waiting_prefix,prefilling,decoding,waiting_mask,decode_batches_by_width}`, `admission.*`, `http.{requests,connections}` |
| ANE | `ane_ffn.{state,share,minimum_rows,reason,split_commands,evaluations,ane_ms,reruns}` + `model_timing.prefill.*` for the derived shares |
| Model card | `instance.{model,host,port,pid,id,started_at}`, `maximum_context_tokens`, `input_modalities`, `vision`, `identity.*` (KV dtype, block tokens, model fingerprints), `memory_plan.device` (chip, GPU cores, RAM), `chat_template.later_system` (scalar, or a `{name: value}` dict when the tokenizer has several templates. The renderer handles both shapes), `schema_version` |

Fields that may be absent (engine restarting, optional subsystems) must render as an em-dash or an "n/a" chip. Never as `undefined` or a crash. The page treats the status document as partial by contract.

## Behavior notes

- **Theming:** one palette of CSS variables for light/dark (as `chat.html`). Charts read the same variables via `getComputedStyle` at theme detection. They rebuild on a `prefers-color-scheme` change.
- **Empty engine:** before the first successful `/status` (engine loading), panels show a "waiting for engine" state. The page never shows fake zeros.
- **Restart semantics:** cumulative counters are since-engine-start. `transport.restarts` tells when they reset. Sparkline windows clear on restart.
- **No timers that outlive the page:** all intervals live in one controller object, with a single teardown path. Navigating away must leave no leaked fetches. Abort-on-unload is not required. Intervals must stop when the page is hidden-paused.

## Files

| File | Change |
| --- | --- |
| `server/dashboard.html` | new — the page (inline CSS + JS) |
| `server/dashboard-assets/chart-<version>.min.js` | new — vendored Chart.js UMD |
| `server/server.py` | load `DASHBOARD_HTML` at import. Serve `GET /dashboard` under the `webui` gate. Add the page route and its assets to the public-route set in `parse_request`. That method holds the public-vs-authenticated split (`server.py:333-340`). `verify_request` cannot see the route, so it cannot hold it. Register the asset files |
| `dev/tools/package.py` | add `dashboard.html` to `SERVER_FILES`. Add a `DASHBOARD_ASSET_FILES` tuple staged into `server/dashboard-assets`, mirroring the `chat-assets` pattern |
| `THIRD_PARTY_NOTICES` | Chart.js entry, with license text, following the markdown-it/Highlight.js entries |
| `dev/tests/server/test_server_access.py` | route tests (see Test contract) |
| `dev/tests/tools/test_package.py` | its staged-`server/`-directory assertion is exact (`{*SERVER_FILES, "chat-assets"}`). It gains the dashboard entries |
| `dev/tests/tools/test_packaged_server.py` | its `stage_release` helper mirrors the staged folder list. It must gain the `dashboard-assets` folder or it raises `FileNotFoundError` |
| `DEVELOPMENT.md` | route-table row + a short "Dashboard" section. It updates three existing statements that would otherwise become false. The `--api-key` row ("every route but the chat page, `/health` and `/ready`"). The `--no-webui` row ("Disable the chat page."). The Network-and-authentication prose ("only the chat page, `/health`, `/ready` … stay public"). My call, so the user-facing docs stay true about which routes need a key |
| `README.md` | one line in the quick start |

## Test contract

Follow the existing patterns in `dev/tests/server/test_server_access.py` and `test_status_contract.py`:

1. `GET /dashboard` → 200, `text/html; charset=utf-8`, body is the page. Only while `webui` is on.
2. `--no-webui` → `GET /dashboard` → 404, same as the chat page.
3. With `--api-key` set: `GET /dashboard` (`GET` and `HEAD`) → 200 unauthenticated (public, like `/`). Meanwhile `GET /status` → 401 (already covered).
4. `GET /dashboard-assets/<chart>` → 200 `text/javascript`. A sibling path outside the registered set → 404. Files must not be exposed.
5. Path normalization: `/./dashboard` and `/dashboard/../dashboard` route as the dashboard (existing harness pattern).
6. Packaging: the packaged server ships the dashboard page + asset. The exact assertions live in `test_package.py` (staged `server/` directory) and `test_packaged_server.py` (staged folder list).
7. The page changes no `/status` or `/metrics` contract. Existing tests must pass unchanged.

The page's JavaScript has no automated test in this repo. There is no JS test harness and no build tooling. The chat page's JS is likewise untested. The team verifies it by hand against a live server, including with `--api-key` and with `--no-webui`. It stays deliberately simple: one poll controller, one render function per panel, formatting helpers. The implementation plan must include this manual verification as an explicit step.

## Open items deliberately left to the builder

- Exact Chart.js configuration (arc angles, tick density, animation timing) and the nice-rounding function for gauge scales.
- Grid template details, spacing, the sub-900 px stacking.
- Ring-buffer length (design suggests ~150 samples ≈ 5 min at 2 s).
- The precise "n/a" rendering for each optional field.
