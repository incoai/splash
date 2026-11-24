# Splash coding policy for Hermes

This plugin is opt-in. Qualify it in a dedicated Hermes profile against a
loopback Splash Chat Completions endpoint. Copy the `splash-coding` directory
into that profile's plugins directory and use the profile's usual plugin
settings.

```yaml
plugins:
  enabled:
    - splash-coding
  entries:
    splash-coding:
      settings:
        endpoint: http://127.0.0.1:11502/v1
        route_model: qwen3.8-flash-next-splash
        workspace: /absolute/path/to/the/approved/worktree
        cache_group: approved-coding-workspace
        shared_prefix: true
        bounded_reads: false
```

Use a cache group only within a trust boundary that is authorized for the
same repository. Neither the cache group nor a cache salt is an authorization
boundary. The plugin scopes a salt to the endpoint, model, exact worktree,
Git revision pinned at session start, configured prefix, and ordered tool
definitions. It preserves explicit cache salts from other callers. Splash
still requires exact matching prompt tokens for reuse. Start a new session
after changing the repo snapshot, tools, or workspace instructions.

`system_prefix` is an optional, bounded string of operator-reviewed workspace
instructions. The plugin prepends it to the existing system message without
changing the existing message roles or content. Leave it empty unless there
is stable shared material worth caching. Do not populate it automatically
from repository files or tool results. The plugin neither seeds a cache
entry nor changes the RAM/SSD limits. A full stable prompt head can be seeded
by normal work and reused by subsequent sessions.

Enable `bounded_reads` separately for an experiment. On an unbounded initial
read of a large local regular file inside the approved worktree, the hook
asks for symbol search and a focused read. The next read is allowed as
requested, including expanded ranges or an unchanged retry. The regular
Hermes path controls and tool approvals still apply.

Run the focused tests from the Splash checkout.

```sh
python3 -m unittest discover -s dev/integrations/hermes
```

## Qualification on Apple Silicon

These measurements used a signed Splash 1.7.1-based Coder IQ1_M deployment
on an M5 Max with 96 GiB of engine memory and a 512 GiB persistent cache.
The branch is based on Splash 1.7.2. Model performance numbers below describe
the tested 1.7.1-based runtime.

| Comparison | Metric | Control | Candidate |
| --- | --- | ---: | ---: |
| Shared repo prefix, four agents and two matched rounds | Verified fixes/min, including seed time | 2.54 | 4.19 |
| Shared repo prefix, four agents and two matched rounds | Initial p95 time to first tool call | 69.61 s | 23.31 s |
| Shared repo prefix, four agents and two matched rounds | Held-back task passes | 7/8 | 7/8 |
| Bounded reads with a shared prefix, two large-file tasks | Follow-up p95 TTFT | 3.29 s | 1.63 s |
| Bounded reads with a shared prefix, two large-file tasks | Held-back task passes | 2/2 | 2/2 |

The cache comparison used the same prompt content and tool definitions for
both arms. Each cold agent used its own cache namespace. The shared arm
seeded the common snapshot once per wave. Seven of its eight starts restored
from memory and one restored from SSD. The two-task read comparison changed
only the bounded-read policy. Both used independent graders on temporary
repository copies. These are small workload-specific samples.

A separate 48-line first-window cap was tested with four concurrent agents,
four tasks per wave and reversed order in the second round. Both arms passed
8/8 independent checks, but follow-up p95 TTFT increased from 2.04 s to 4.51 s
and verified fixes/min including seed time decreased from 6.12 to 5.32.
The cap remains outside this integration. The measured TTFT includes engine
admission wait. The benchmark did not include an external queue hop.

The upstream-based branch passed the focused plugin tests, Ruff, and all
488 integrated engine tests, including shared-head checkpoint and
persistent-cache cases. The installed Hermes runtime also exercised the
plugin middleware and read fallback in an isolated profile.
