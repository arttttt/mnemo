# 12 — Testing

Testing is a first‑class requirement (NFR‑23..25), not an afterthought. Two distinct kinds, kept in separate suites.

## Layers → test kind

| Layer | Kind | How |
|---|---|---|
| `domain` (entities, rules) | **unit** | pure, no I/O — hashing, normalization, entity invariants |
| `application` (use cases) | **unit** | drive use cases with in‑memory/fake ports; assert behavior (dedup, scoping) |
| `adapters` (store / embedder / mcp / cli) | **integration** | exercise each adapter against its real boundary |
| `infrastructure` (container / config) | **integration** | wiring builds a working use case from config |

## Layout

```
tests/
├── unit/
│   ├── test_domain.py
│   └── test_use_cases.py
└── integration/
    ├── test_in_memory_repository.py   # persistence round-trip
    ├── test_container.py              # composition root wires a working use case
    ├── test_cli.py                    # CLI commands end-to-end
    ├── test_mcp_server.py             # MCP tools registered + callable
    └── test_fastembed.py             # real embedder (marked heavy, opt-in)
```

## Principles

- **Fast & offline by default:** tests use the `hash` embedder; unit tests inject an in‑memory fake repository,
  integration tests run against a temp‑file SQLite store — no network, no heavy deps.
- **Heavy/networked tests are opt‑in:** the real embedder (fastembed) lives behind `@pytest.mark.heavy` and
  are skipped unless explicitly requested.
- **Ports enable isolation:** because use cases depend on ports, unit tests inject fakes — no MCP/DB needed to test core logic.
- **One behavior per test;** integration tests assert the adapter honors its port contract.
- **Determinism:** the `hash` embedder is deterministic, so semantic‑ranking assertions are stable in CI.

## Running

```bash
uv run pytest                 # unit + offline integration (default)
uv run pytest tests/unit      # only unit
uv run pytest -m heavy        # real-backend integration (downloads models)
```

## What "done" means per phase
- A phase is not complete until its new domain/use‑case logic has unit tests and its new adapters have
  integration tests (see [10-roadmap.md](10-roadmap.md) Definition of Done).
