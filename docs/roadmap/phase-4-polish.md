# Phase 4 — Polish & optional upgrades

**Goal:** make it pleasant and robust to install and operate; add quality upgrades that are opt‑in.

Step format: **Why** (the requirement) · **What** (exactly what to do) · **Done when** (verifiable).

---

### 4.1 Optional reranker + entity extraction (opt‑in)
**Why:** extra models/quality upgrades must be justified and opt‑in, never default (axiom 6).
**What:** a pluggable reranker and a local entity extractor, both off by default.
**Done when:** enabling them measurably improves retrieval on a small benchmark; the default build is unchanged; tests.

### 4.2 Export / import
**Why:** portability and no lock‑in (NFR‑18).
**What:** dump all memory to a portable file and restore it into a fresh store.
**Done when:** export then import into an empty store reproduces the data exactly.

### 4.3 Packaging + per‑client install guides
**Why:** install must be easy and reproducible (NFR‑16/17).
**What:** a published install path and per‑client setup docs.
**Done when:** a clean machine installs and connects following the guide alone.

### 4.4 Air‑gapped mode
**Why:** must run fully offline after install (NFR‑3).
**What:** a pre‑seeded model cache + offline flags.
**Done when:** the system runs with no network after a one‑time setup.

### 4.5 Tasks (optional)
**Why:** tasks linked to memories are useful (FR‑12, optional).
**What:** minimal create / list / complete for tasks, linkable to memories and surfaced in `recall`.
**Done when:** tasks persist, link to memories, and appear in `recall`.

---

**Phase 4 done when:** every shipped item has its own test/guide and the default build stays lean.
