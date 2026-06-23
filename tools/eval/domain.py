#!/usr/bin/env python3
"""Project-fact domain eval (п3) — the REAL go/no-go readout for mnemo. NOT BUILT YET.

LoCoMo (eval.locomo) only validates the machinery against a public standard; it is
conversational, not mnemo's domain. This eval measures mnemo on its actual job — project-fact
memory — and is where reranker / fusion / gate decisions are finally settled.

Plan (see docs/FEEDBACK "Benchmark harness"): an in-repo, versioned fixture built from mnemo's
own dogfooded bank + a hand-authored question set, each question tagged with gold source memory
id(s) and a gold answer (or REFUSE). Three deliberate slices:
  - answerable  — the answer is in the bank; gold = source id(s) + expected answer.
  - irrelevant  — nothing answers it; gold = REFUSE (the abstention slice).
  - superseded  — a newer version exists; gold = the CURRENT value, not the stale one.
Reuses the same Tier-1 metrics as eval.locomo (Recall@k / MRR + Any/Complete + the abstention
curve) on the isolated harness, so it slots onto tools.eval.core with a domain loader.
"""
import sys


def main() -> None:
    sys.exit("tools.eval.domain (п3) is not built yet — see the module docstring for the plan.")


if __name__ == "__main__":
    main()
