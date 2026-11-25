"""The README's table: our slowest time against any other simulator's fastest.

    python conservative.py

Reads the run of 23-24 November 2026 and the rerun of 24 November, at 1 and 16
threads, for one more evaluation of a prepared circuit. UNPROVEN workloads are
left out: the other tool was killed for memory in the rerun.
"""
import json
from collections import defaultdict
from pathlib import Path

import summarize_scoreboard as summary

HERE = Path(__file__).resolve().parent
RUN, RERUN = HERE / "results-2026-11-24-vicena", HERE / "results-2026-11-24-rerun"
UNPROVEN = {"B_noisy_6x6_L5", "D_noisy1q_6x6_L5"}


def main():
    measure = summary.warm("warm_value_s")
    ours, theirs = defaultdict(list), defaultdict(list)
    for threads in ("1t", "16t"):
        mine, others, _, order, _ = summary.load(str(RUN / f"scoreboard-{threads}"))
        for line in (RERUN / f"pp-{threads}.jsonl").read_text().splitlines():
            r = json.loads(line)
            others[r["workload"]].append(("PauliPropagation.jl", r["parts_mode"], r))
        for w in order:
            us, them = summary.best(mine[w], measure), summary.best(others[w], measure)
            if us:
                ours[w].append(us[0])
            if them:
                theirs[w].append((them[0], them[1]))
    print("| Workload | Ours, slowest | Fastest other | Lead |\n| --- | ---: | ---: | ---: |")
    for w in ours:
        if theirs[w] and w not in UNPROVEN:
            us, (them, tool) = max(ours[w]), min(theirs[w])
            print(f"| {w} | {summary.f(us)} | {tool} {summary.f(them)} | {them / us:.3g}× |")


if __name__ == "__main__":
    main()
