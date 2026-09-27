# RFC 193 — P-core work-interval hint + host-gap hiding

Upstream issue: https://github.com/incoai/splash/issues/193

Source: bonsai `CBv2EngineWorkInterval.swift` (`AudioWorkIntervalCreate`
P-core hint, `MLXFAST_ENGINE_WORK_INTERVAL`, prompt-forward renew).

## Problem

Host gap between verify-block build and GPU submit shows up as idle
(`Engine.cpp`, `Scheduler.*`, `NativeRuntime.*`, `Bootstrap.mm`).
Bonsai reports 11.4→3.1 ms verify build / 2.0→0.6 ms idle from a
scheduling hint alone; Splash lists host-gap hiding as out-of-scope
(`PLAN.md:26`) with no measurement.

## Proposal

1. `runtime/engine/Rfc193WorkInterval.hpp` declares the flag + env var
   (`SPLASH_ENGINE_WORK_INTERVAL`).
2. Follow-up scopes an opt-in P-core / QoS hint to the
   prompt-forward-submit + verify-build window only; renew on
   `promptForwardBegan/Submitted`.

## Contracts

- Scheduling hint only; no numerics / protocol / memory-plan change;
  must not starve UI/audio.

## Validation

- `benchmark-decode-profile` verify-build ms + GPU idle ms B1–B4
  (alternating, medians ≥31); ABBA neutral required. Easiest to revert.
