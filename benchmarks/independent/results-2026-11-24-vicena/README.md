# Scoreboard on a Vicena computer, 23–24 November 2026

The [scoreboard](../README.md) run once with one thread per tool and once with sixteen, on a Vicena computer: an AMD EPYC 7502P container with a 16-CPU quota and 32 GiB of memory, Linux, Python 3.12.10. Quantum Circuit Simulator was built from commit `c5127f0` of the `kernel-fusion` branch; every other version is in each run's `machine.json`.

| Run | Tables | Raw records |
| --- | --- | --- |
| One thread per tool | [scoreboard-1t/scoreboard.md](scoreboard-1t/scoreboard.md) | `scoreboard-1t/*.jsonl` |
| Sixteen threads per tool | [scoreboard-16t/scoreboard.md](scoreboard-16t/scoreboard.md) | `scoreboard-16t/*.jsonl` |

`python summarize_scoreboard.py results-2026-11-24-vicena/scoreboard-1t` regenerates each table byte for byte from its records. The tables were produced by the summary of commit `1304e2d`, which counts a workload that Quantum Circuit Simulator refused and another tool finished as a loss.

## Caveats

- **quimb used more than one core.** It starts helper processes of its own, so its one-thread figures had more of the machine than the label says. This favours quimb.
- **Leftover helper processes.** When a quimb worker timed out, the harness of this run killed the worker but not quimb's helpers. Twenty of them stayed alive, blocked on a pipe; a sample showed them using no CPU, and they were killed before the sixteen-thread run finished. Workers now run in their own process groups (`isolated.py`), which prevents this.
- **Failures without a reason.** Several PauliPropagation.jl rows are `failed` with an empty error: Julia exited without a result or a message, most likely killed for memory on the deep circuits. The harness now records the exit status.
- **A shared host.** The container shares its physical machine; figures can vary between runs, and small timings carry more of that variation.

Return to the [comparison harness](../README.md).
