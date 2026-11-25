# AI-assisted contributions

AI tools are welcome here. This policy makes the human accountable, keeps
review possible, and keeps generated work from hiding risk.

## Rules

1. **Disclose the tool.** Name the model or agent in the pull request. A short
   line is enough. Disclosure is not a reason to reject a change.
2. **A human owns the change.** The human author must be able to explain every
   line, answer review questions, and run the checks. The agent cannot be the
   accountable author.
3. **One logical change per pull request.** Put exploration in a draft. Do not
   mix a refactor, a behavior change and generated assets in one review.
4. **Evidence is first-hand.** Paste the command and the result. "The agent
   says the tests pass" is not evidence.
5. **Generated code and assets are identified.** Say what was generated, why it
   is committed, and what license applies. Do not commit a generated blob to
   avoid a build step that the repository already defines.
6. **Agents do not act as maintainers.** An agent may draft a comment, issue or
   pull request. A human sends it, applies labels, closes it, or merges it.
7. **Keep secrets out.** Never paste tokens, private repository names, or
   environment files. The architecture check rejects Hugging Face tokens in
   workflows for the same reason.

## Size and review

Keep a pull request to one logical change. Above about 400 changed lines or ten
files, explain why it cannot be split. Above about 2,000 lines, link a design
issue or an RFC that agreed on the approach. A release pull request is exempt
because its scope is the release.

For a stacked change, name the base pull request and mark the stack in the
title or body. Reviewers should be able to see the incremental diff; a stack
whose every pull request repeats the same 30,000 lines is not reviewable.

## What review will ask

- Does the change belong in this layer?
- Does it add a special case where one rule should cover every case?
- Which test fails on the old code and passes on the new code?
- What happens on cancellation, replay, eviction, a refused allocation or an
  out-of-memory start?
- Which Mac, model, revision and workload produced the numbers?
- What remains unmeasured or device-specific?

If the answer is "the agent decided", the change is not ready.
