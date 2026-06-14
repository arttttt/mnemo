# Phase 3 — Background consolidation

**Goal:** a background worker improves memory off the hot path with a small local model — **concurrent from the
start** — and it never decides staleness on its own (flag‑only).

Each step is **Why** (the requirement and reasoning) · **What** (exactly what to build) · **Done when** · **Depends on**.

---

### 3.1 On‑demand generator

**Why.** The "reasonable LLM" decision: a small model (≈4B, e.g. a Qwen3‑4B / Gemma‑class instruct model) may run,
but **only in the background and only transiently** — loaded for a consolidation run, unloaded after — so the system
keeps ~0 RAM at idle. And because small models are unreliable at free‑form structured output, we require **guided
decoding** (grammar / constrained JSON) so what comes back is always valid and applicable.

**What.** A local generator behind a port that loads on a trigger and unloads after the run, producing schema‑valid
output via grammar/guided decoding. The inference engine is chosen for the concurrency need (see 3.2).

**Done when.**
- The model loads on a trigger and unloads after the run.
- Output is always schema‑valid.
- RAM returns to baseline after a run.

---

### 3.2 Concurrent consolidation engine

**Why.** With 10+ agents, memory accumulates fast, so consolidation must be **built concurrent from the start** — a
worker pool over batches, not one serial pass — and it must **never block** the hot write/read path. (This is a
deliberate decision: an earlier note assumed a single serial pass; we changed it.)

**What.** Process candidate batches in parallel with backpressure, entirely off the hot path. If the model must serve
parallel requests, use an inference server that batches concurrent requests rather than one that serializes them.

**Done when.**
- Multiple batches process in parallel with backpressure.
- The hot path (store reads/writes) is unaffected during a run.
- Tested.

**Depends on:** 3.1.

---

### 3.3 Triggers

**Why.** Consolidation should run automatically at sensible moments and — per the no‑LLM‑on‑write rule — **never** on
the write path.

**What.** Trigger by volume (every N new records), on idle, and on manual request.

**Done when.**
- Each trigger path starts a run.
- It never fires on the hot write path.
- Tested.

**Depends on:** 3.2.

---

### 3.4 Operations: merge / summarize / insights / contradiction‑flag

**Why.** The background pass is where the "smart" work lives: merging the near‑duplicates we deliberately did **not**
suppress on write, compressing clusters of small notes, and extracting reusable lessons. Most important, per our
agreement, it **flags** contradictions for review and **never auto‑invalidates** anything — the system does not
decide staleness for the user; currency changes only on an explicit signal.

**What.** Implement near‑dup merge, cluster summarization, insight extraction, and contradiction **flagging** — all
via guided output, each idempotent and failure‑isolated (a failed batch never corrupts data and is simply retried).

**Done when.**
- Each operation works via guided output and is idempotent and failure‑isolated.
- Contradictions are flagged for review, never auto‑marked stale.
- Tested.

**Depends on:** 3.1–3.3.

---

### 3.5 Semantic links (background)

**Why.** The associative and contradiction links that need *judgement* (unlike the deterministic `supersedes` /
`derived_from` from 1.8) belong here, where the model has the context of neighbouring memories — and, again, without
burdening the coding agent. Inferred links must be clearly marked and offered, not silently applied.

**What.** From nearest‑neighbour candidates, propose typed links (`related_to` / `contradicts`) tagged
`provenance=llm`, surfaced for review rather than auto‑applied.

**Done when.**
- Proposals are stored as suggestions (`provenance=llm`) and surfaced.
- They are not auto‑applied.
- Tested.

**Depends on:** 1.8, 3.1.

---

### 3.6 Degradation without the generator

**Why.** On a resource‑tight machine the whole consolidation layer must be optional — memory still works fully
without it (and without ever needing a big model).

**What.** A "generator off" mode where consolidation is skipped cleanly.

**Done when.**
- Store / search / delete all work with the generator off.
- Consolidation is skipped without error.
- Tested.

---

**Phase done when:** consolidation is concurrent and off the hot path; the generator's RAM is transient; nothing is
auto‑invalidated; the system runs with the generator disabled.
