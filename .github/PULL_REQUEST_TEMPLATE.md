<!--
Keep one logical change per pull request. Read AGENTS.md and
docs/contributing/verification.md before filling this in. Delete sections that
do not apply; do not delete the evidence fields.
-->

## What changed and why

<!-- The smallest contract that changed, and the problem it solves. -->

## Linked issue

<!-- A repository issue, or why there is none. Larger changes need an issue. -->

## Layer and files

<!-- server, runtime/engine, runtime/model, runtime/ops, runtime/metal,
     install, app, dev, or docs. Name the files a reviewer should read first. -->

## Verification

```text
Commands:
Results:
Mac and macOS:
Model and revision:
Checks not run, and why:
```

<!-- Paste first-hand output. "The agent says it passes" is not evidence. -->

## Performance (if applicable)

<!-- ABBA baseline vs candidate, metric, complete workload, spread. State the
     command and where the result is kept. A single run is not evidence. -->

## AI assistance

<!-- Disclosure is welcome, not disqualifying. Name the tool and the scope. -->

- [ ] No AI tools were used.
- [ ] AI tools were used; the tool and scope are described above.

## Review checklist

- [ ] I read `AGENTS.md` and the reference for the layer I changed.
- [ ] The change is one logical unit, or I explained why it cannot be split.
- [ ] I ran the checks in `docs/contributing/verification.md` for this change.
- [ ] I included the commands and results above.
- [ ] I did not commit build outputs, model files, reports, personal notes or
      secrets.
- [ ] I can explain every line and respond to review.
