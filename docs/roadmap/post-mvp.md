# Post‑MVP (after the initial scope)

These are committed for later. They are deliberately **not** in the initial scope, but the MVP is designed so
they add cleanly (e.g. the schema is forward‑compatible).

Format: **Why** (the requirement) · **What** (exactly what to do).

---

### Full bi‑temporal validity model
**Why:** answer "what was true / what did we know as of date X" and support retro‑corrections, without deleting
history — the only rigorous way to handle changing facts over time.
**What:** add transaction‑time (`created_at` / `expired_at`) and valid‑time (`valid_from` / `valid_to`) on records,
with point‑in‑time queries. Done **in full** (no half‑measures); the schema is already forward‑compatible.

### `importance` reintroduced
**Why:** prioritise and age memory in retrieval.
**What:** blend importance + recency into ranking, add decay, and optionally auto‑score importance in consolidation.

### Revision tooling for flagged contradictions
**Why:** the human/agent decides what is stale — the system never auto‑invalidates; they need a way to act on flags.
**What:** a review/resolve flow over the contradictions the background worker flagged.

### `session` scope
**Why:** some context is only relevant within a session.
**What:** a `session` scope value + the matching search filter.

### `recall(project)` — background‑precomputed digest (basic recall **shipped**)
**Why:** one call that returns a concise "here's where you are" (rules + what matters now) instead of forcing
several searches.
**What (shipped):** `recall(query, project)` — a local LLM synthesizes a grounded answer from the project's
memories on demand (the one opt‑in LLM read tool; the write path stays LLM‑free; the generator is transient).
**What remains post‑MVP:** a background worker that **precomputes** a project digest off the hot path so
`recall` just reads it (no per‑call generation, lower latency), plus recall‑latency/quality tuning.

### Transient embedder (idle‑unload)
**Why:** the design principle is "heavy things are transient" and the README promises "~1 GB while active,
~0 when idle" — but only the **generator** (loaded just for a consolidation window) and the **whole service**
(idle‑exit when the last connector leaves) actually unload. The **embedder stays resident for the service's
life**: its ONNX session holds **~0.7 GB** (mmap'd weights paged in by inference + the ORT CPU arena's
high‑water of activations), and a single 2048‑token encode bumps that to **~1.5 GB and never releases**
(measured). So while an agent is connected but **quiet** (not writing), the service holds the full embedder
footprint instead of falling toward idle — the resting plateau does **not** drop on its own. (ORT arena tuning
— `enable_cpu_mem_arena=False` or per‑run shrinkage — only lowers the long‑input *peak* ~1.5→1.0 GB; it does
not return the resting plateau, which is the loaded model itself.) *Observability: the runtime lifecycle logs now
report **current** RSS next to the peak, so a load/unload's real resident delta — and whether a free actually lands —
is visible; previously they logged only the monotonic peak, which by construction never shows a release.*
**What:** make the embedder transient like the generator — **lazily load** the ONNX session on first encode and
**unload** it (drop the session → the OS reclaims the weights + arena) after an idle grace, reloading on demand.
Embedding is already off the hot path (the async worker), so the reload cost is tolerable; a grace timer avoids
load/unload thrash. Natural home: the async embedding scheduler (it already has an idle notion). Closes the
"connected but quiet" gap so idle RAM falls to **baseline**, not only to ~0 when *every* agent disconnects.
Optionally pair with `enable_cpu_mem_arena=False` to also cap the active long‑input peak.

### MLX runtime for the embedder (Apple‑Silicon speed)
**Why:** on Apple Silicon, MLX (unified‑memory, GPU) is typically faster than the embedder's
current ONNX CPU path, and the embedder is the resident hot component. But the embedder is
`pplx-embed` (custom architecture `bidirectional_pplx_qwen3`), and **stock/generic MLX cannot
load it** — the same custom‑arch reason it runs on ONNX rather than stock MLX. So the speed‑up
is gated on **porting the architecture**, not just converting weights.
**What:** implement `bidirectional_pplx_qwen3` in MLX (the attention/embedding modules + the
bidirectional pooling head), convert the 0.6b‑int8 weights to MLX format, and add it behind the
existing embedder port as a **config‑selected alternative backend**, keeping ONNX as the
default/fallback. Bench MLX vs ONNX on the embedder retrieval task for speed **and** quality
before switching the default. Note (2026‑06): MLX for the *generators* was deliberately **not**
pursued — the official Gemma 4 QAT GGUFs (near‑lossless Q4, `UD‑Q4_K_XL`) on llama.cpp/Metal
were prioritised because quantization quality matters more than the MLX speed gain there; the
embedder is the remaining MLX opportunity.

### Fail‑fast on an embedder/store dimension mismatch
**Why:** the store bakes its embedding dimension at first write (`CHECK(vec_length(embedding) == N)`); the
`dim` passed to the repository is only used to create a *fresh* schema and is **not** reconciled with an
existing store. If the two disagree the only signal is a cryptic deep error — a `CHECK` violation on write or
sqlite‑vec's `Vector dimension mismatch` on query. The common trigger (a stale resident service on the old
dimension after a reindex) is already closed (0.2.4 restarts the service after reindex); what remains is the
manual case — switching `MNEMO_EMBEDDER` to a different‑dimension model **without** running `mnemo reindex`.
Low priority, but a power‑user footgun that fails opaquely.
**What:** a defensive check that compares `embedder.dim` against the store's baked dimension and **fails fast**
with an actionable message ("store is dim 1024, embedder is dim 384 — run `mnemo reindex`, or pin
`MNEMO_EMBEDDER` to a 1024‑dim model"). **Do NOT auto‑reindex** (a heavy, data‑touching op — the user decides).
Add a `current_dim()` read to the repository port (symmetric to `set_dimension`) and run the check at
**service start** (next to the migration hook) — crucially **NOT** in the repository constructor /
`build_container`, or it would break `mnemo reindex` itself, which must open a mismatched store to fix it via
`set_dimension`. Optionally extend the check to the CLI `store`/`search` commands (but never `reindex`) to
cover the direct‑CLI path too. From dogfooding FEEDBACK item 2.

### Edit / re‑key an existing memory (evolve identical content under a new topic_key)
**Why:** `remember` evolves a memory by reusing a `topic_key` with **changed** content (the topic_key upsert
supersedes the prior). But re‑storing **identical** content under a *new* topic_key hits the exact‑dup guard
first — it is now a **loud error** (it used to be a silent drop), but there is still no way to re‑key or edit an
existing memory in place. From the 0.2.14 review.
**What:** a deliberate edit/re‑key op (an explicit MCP tool, or a `topic_key`‑precedence flag on `remember`)
that moves identical content onto a new topic_key / chain — or edits an active memory's metadata — without
forking a duplicate. Settle the semantics (re‑key vs. attach vs. supersede‑with‑same‑content) when the
ops/management surface is designed; it pairs with the `get`/`neighbors` + management‑surface items below.

### Language‑aware memory‑size caps
**Why:** the per‑memory cap is in **tokens** (per type — `rule` 128, others 512), which is far from char‑fair
across languages. Measured on the pplx tokenizer (chars/token): English 4.97, Russian 2.74, Chinese 1.64,
Hindi 1.08 (Devanagari — the most token‑hungry). So at 128 tokens the real budget is ~636 English chars but only
~350 Russian / ~210 Chinese / ~137 Hindi — a non‑English memory gets far less actual content for the same cap.
**What:** consider a per‑language / per‑script **correction factor** on the effective cap so the char/meaning
budget is comparable. Open: cheap language/script detection on the write path; source factors from the
chars‑per‑token table; and whether to correct at all — tokens already track semantic density (a CJK/Devanagari
char carries more meaning), so this may only matter for the most token‑hungry scripts.

---

## Retrieval & recall surface — problems found by dogfooding

Using mnemo on its own memory surfaced one consistent signal: the write/storage side is mature, but the
**read/recall** surface lags ("dump the corpus" more than "precise answer"). These are committed as post‑MVP
research/improvements. Sources are the tagged feedback memories [[feedback/mcp-retrieval-ux]] and
[[feedback/type-discipline]], re‑validated live on the SQLite + `sqlite-vec` + bge‑small path.

### `search` relevance score — RESOLVED by REMOVAL
**Why:** the `search` `score` was a reciprocal‑rank‑fusion value (k=60, ≈`1/(60+rank)` per channel) — every hit in
a narrow ~0.016–0.033 band, **not** a similarity / confidence. A consuming agent misreads it as relevance
confidence; in dogfooding it caused a *false* diagnosis ("weak embedder") when the embedder was fine. Opaque
**and** misleadable.
**What (done):** the field is **removed from the search response**, not relabeled. The alternatives were weighed
and rejected:
- *Rename `score` → `rrf`*: honest, but leaves the agent with no usable relevance number — and it does not need one.
- *A "real" similarity/confidence scalar*: a HYBRID (dense + lexical/FTS) hit has no honest single relevance number —
  a cosine value under‑represents a lexical‑leg hit, re‑creating the same misread.
- *An internal relevance floor / refusal on `search`*: rejected. The consumer is a capable LLM agent that READS each
  hit's content, judges relevance itself, and controls breadth via its own `limit`. A miscalibrated floor would DROP
  the real answer (unrecoverable — strictly worse than the noise the agent filters out), and the threshold has no
  production ground truth to calibrate against. (A floor is meaningless for `browse` — there is no query.)
The RRF value stays **internal** (it only orders the hits — the list order conveys the ranking). Retrieval‑quality
leverage is the **bank** (consolidation: dedup + staleness) and a **feedback loop**, not an exposed score. Refusal /
faithfulness gates remain a RECALL (generator) concern, not a `search` one. Lexical **corroboration** (counting query
terms a hit carries) was likewise weighed as an exposed/gating signal and **rejected for `search`** for the same
reason — the agent already reads the hits — and kept, if anywhere, as a RECALL input‑gate lever (not a priority).

### Bank consolidation — deferred; a read‑only audit, never a background auto‑writer
**Why:** consolidation (near‑dup dedup + staleness/supersede) is the real retrieval leverage — the score item above
points here — but the open question from the latest review is *how it runs*, not just *what it does*. A worker that
**rewrites** the bank unattended is a standing failure point: a bad merge or a wrong supersede is **persistent
poison** (the generation‑vs‑structure axis), and at single‑user scale there is **no feedback signal** to catch it.
**Decision (2026‑06):** if/when pursued, consolidation is **read‑only and propose‑only** — a `mnemo audit` (built on
the management surface below) that **surfaces** near‑dup clusters (SemDeDup: embed → cluster → cosine, LLM‑free) and
staleness/supersede candidates (recency + topic, optional NLI) for the **human/agent to apply**, through the existing
propose‑not‑apply seam (`ProposedMemory` / `Plan` / `Operation`) and **supersede‑not‑delete**. Never an unattended
writer; **no abstractive merge** (extractive / classification only). **Deferred for now** — marginal at personal
scale until there is real usage feedback to calibrate against; the lever that unblocks it first is a **feedback
signal**, not more retrieval knobs. This qualifies the auto‑apply framing in *Retrieval robustness to write‑time
mis‑typing* below: re‑typing too is **propose‑first**, not unattended auto‑apply, under the same stance.

### Graph navigation at the MCP surface — `get` **shipped**, `neighbors` deferred
**Why:** memory is densely linked (`[[topic_key]]` wikilinks), but there is no way to *traverse* relationships
from the tools — reaching a linked memory means another semantic search and hope.
**What (`get` shipped):** `get(id | topic_key)` — an exact point lookup returning the full record plus its
**supersede chain** (walked along the `supersedes` pointers, light entries, paged by `chain_limit`/`chain_after`).
A `[[wikilink]]` is a `topic_key`, so `get` dereferences it. See [05-mcp-api.md](../05-mcp-api.md).
**What remains (`neighbors`):** `neighbors(id)` (typed edges out/in, single hop — not multi‑hop inference). A typed
`links` table existed once but was **removed** (write‑only — no reader), so this **reintroduces** the typed edge
store (and the `supersedes` column may fold into it) only when there is finally a consumer. Deterministic
typed‑edge graph, not a knowledge graph.

### Query‑less browse / list mode — **shipped**
**Why:** `search` requires a `query`, so "all `type=decision` in this project, newest first" can't be expressed
without inventing a query that itself biases ranking. Retrieving a *category* (e.g. `tags=["feedback"]`) shouldn't
need a semantic guess.
**What:** a separate `browse` tool — a pure filter (type / tags / scope / `created_after`) ordered by recency,
no relevance ranking and no `score`, distinct from semantic `search`. Built on a `retrieve(Retrieval)` store
contract where a request with no text/vector is the browse path (no embedding). See [05-mcp-api.md](../05-mcp-api.md).

### Lexical leg: stop‑word filtering / query sanitizer
**Why:** `_match_query` (the SQLite store) splits a query into every `\w+` token and OR‑joins them
(`"a" OR "b" OR …`), and FTS5's default `unicode61` tokenizer does not strip stop‑words. So "how do we handle
auth errors" pulls in any row containing a filler word ("do", "we") via the lexical leg, which then earns an RRF
contribution — noise that, at a small `limit`, can displace the right answer (a specific instance of the
ranking‑quality wart below).
**What:** a reusable **query sanitizer** rather than inlined rules at the call site — a `StopWordsPort` provider
(built‑in per‑language sets, later file/config‑extensible) plus a `QuerySanitizer` (lowercase, drop stop‑words and
too‑short tokens), injected via the container. `_match_query` would call `sanitizer.tokens(text)` and keep only
the FTS quoting/joining; the same sanitizer is reusable anywhere query text is processed. Consider AND / phrase
matching for multi‑token queries as a separate lever.

### Ranking quality on the hybrid path
**Why:** a generic "status/next‑steps" memory floats to the top of unrelated queries (observed again on the
SQLite + bge‑small path: a storage‑engine query returned the status note above the ADR). At a small `limit` this
displaces the right answer.
**What:** investigate the cause (document length, or one over‑broad catch‑all memory dominating an RRF channel)
and tune — per‑channel weighting, length normalisation, or down‑weighting catch‑all memories.

### Retrieval robustness to write‑time mis‑typing
**Why:** a normative memory mis‑stored as `type=learning` instead of `rule` is silently invisible to the precise
`type=rule` query ([[feedback/type-discipline]]). The root cause is a write‑time data error, not a retrieval bug,
and a query‑side workaround ("also search `type=learning`") would only pollute the access pattern.
**What:** fix at the source via a **re‑type op in the background consolidation worker**: under the determinism
axiom, re‑typing is a cheap, reindex‑free, reversible filter‑facet correction — a candidate for auto‑apply with
`provenance=llm` and a before→after log, rather than flag‑only (which would create review noise for every typo).
Settle the policy when the worker's op set is designed — and against the **propose‑first / no‑unattended‑writes**
stance in *Bank consolidation* above (re‑typing earns auto‑apply only once the read‑only audit proves it safe).

### Management surface — audit & correct memory *through the server* (dereference read **shipped**)
**Why:** auditing and correcting memory (find stale entries, supersede, re‑type, delete) must go through the
**real service** — the one owner of the store — but the MCP read surface couldn't support the AUDIT half: no fetch
by `id`/`topic_key`, and `search`/`browse` exposed neither `topic_key` nor `status`. So a sweep over the memories
forced a **direct read of the SQLite file** — a side‑channel around the store's owner (observed in a real audit).
**What (dereference read shipped):** the per‑handle audit READ is now expressible through the server — `get(id |
topic_key)` (the record + its supersede chain, reaching a superseded version by id or via a topic_key's chain), and
`topic_key`/`status` surfaced on the hits. This closes the direct‑SQLite side‑channel for *dereferencing* a known
memory, and is the read substrate a future read‑only `mnemo audit` (see *Bank consolidation* above) would sit on.
**What remains:** (1) a category‑level **superseded sweep** — listing replaced/stale memories you don't know in
advance (a `browse` status facet) — was intentionally **not** shipped: its only consumer is the deferred audit pass,
so it would run ahead of demand. (2) the *correcting* write ops (supersede / re‑type / in‑place edit) through the
service — see the re‑type item above and *Edit / re‑key*. Management writes must go through the service too, like
`remember` / `delete` already do.

### Deferred indefinitely
**Why:** out of the local, single‑user, lightweight scope mnemo targets.
**What (not doing unless the scope changes):** knowledge graph / multi‑hop traversal; web dashboard; document/PDF
ingestion; multi‑user / RBAC / cloud sync.
