# Rerun of PauliPropagation.jl, 24 November 2026

The records of `rerun_pp.sh` at commit `78a3249` (run as `1bd5c71`, the same files before a history rewrite) on the same Vicena computer as the [first run](../results-2026-11-24-vicena/README.md): `pp-1t.jsonl` and `pp-16t.jsonl`, in three variants (a plain call, a workspace sized up front, `MultiPauliSum` zones). Archive SHA-256 `7337dbb613252c57bbf1c1ddf665915152aea6c7088eb9850ef675c7c7739433`. The other tools were not rerun. `python ../conservative.py` combines both runs into the README's table.

Faults to know before using them:

- **Shared machine.** Other work ran on the computer for at least part of the rerun and may have slowed it.
- **Memory kills.** Thirteen of the 26 whole-circuit runs at one thread ended "killed (likely out of memory)", including two it had solved before. These are not losses of that tool and are not used as such.
- **Single runs.** `rebuild_s` includes Julia compilation, so do not use it.

A [paired run on an idle machine](../paired.py) will replace both runs.

Return to the [comparison harness](../README.md).
