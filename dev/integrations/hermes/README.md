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
