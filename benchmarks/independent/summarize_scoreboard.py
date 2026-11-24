"""The scoreboard tables, from the records scoreboard.sh writes.

    python summarize_scoreboard.py <out-dir> > scoreboard.md

Each competitor's figure is its fastest configuration: whole circuits, one
circuit cut to the light cone, or one circuit per observable term, with
PauliPropagation.jl and pauli-prop at a 1e-12 coefficient threshold.
Quantum Circuit Simulator's figure is its fastest of exact and 1e-12, native
and graph. Errors are against its exact value at the first parameter vector.
"""
import json, sys
from collections import defaultdict

OUT = sys.argv[1]
TARGET = 10.0
NAMES = {"pauli_prop": "pauli-prop", "spd": "spd", "qulacs": "Qulacs", "aer": "Aer", "lightning": "Lightning", "quimb": "quimb",
         "PauliPropagation.jl tuned": "PauliPropagation.jl"}


def rows(name):
    try:
        return [json.loads(l) for l in open(f"{OUT}/{name}")]
    except FileNotFoundError:
        return []


def ok(r, key):
    return r.get(key) is not None and not r.get("error")


def single(r, tool):
    """Setup of this circuit plus one typical evaluation. Once-per-process
    costs (imports, library loading, Julia's compilation) are left out: every
    tool's setup is a second setup of the same circuit in the same process."""
    if not ok(r, "warm_value_s") or r.get("rebuild_s") is None:
        return None
    return r["rebuild_s"] + r["warm_value_s"]


def f(x):
    if x is None:
        return "—"
    return f"{x * 1e6:.3g} µs" if x < 1e-3 else f"{x * 1e3:.3g} ms" if x < 1 else f"{x:.3g} s"


def ratio(ours, theirs):
    if ours is None or theirs is None:
        return "—", None
    r = theirs / ours
    text = f"{r:.3g}×" if r >= 1 else f"{1 / r:.3g}× slower"
    return (text if r >= TARGET else f"**{text}**"), r


# entries[workload] = list of (tool, setup, record)
ours, theirs = defaultdict(list), defaultdict(list)
exact = {}
for r in rows("qcs.jsonl"):
    graph = r["tool"] == "pauli_cones_graph"
    fell_back = graph and not (r.get("diagnostics") or {}).get("backend", "graph").endswith("graph")
    if r["tool"] == "pauli_cones" and ok(r, "value"):
        exact[r["workload"]] = r["value"]
    if not fell_back:
        ours[r["workload"]].append(("graph" if graph else "native", "exact", r))
for r in rows("qcs_1e-12.jsonl"):
    graph = r.get("tool", "").endswith("_graph")
    if graph and r.get("backend") not in (None, "graph"):
        continue
    ours[r["workload"]].append(("graph" if graph else "native", "1e-12", r))
for r in rows("pp.jsonl"):
    theirs[r["workload"]].append(("PauliPropagation.jl", r.get("parts_mode", "whole"), r))
for r in rows("others.jsonl"):
    parts = r.get("cones", "none")
    theirs[r["workload"]].append((NAMES.get(r["tool"], r["tool"]), "whole" if parts == "none" else parts, r))
order = list(dict.fromkeys(r["workload"] for r in rows("qcs.jsonl")))


def err(w, r):
    if w not in exact or r.get("value") is None:
        return "no ref"
    return f"{abs(r['value'] - exact[w]):.0e}"


def best(entries, measure):
    xs = [(measure(r, tool), tool, setup, r) for tool, setup, r in entries]
    xs = [x for x in xs if x[0] is not None]
    return min(xs, key=lambda x: x[0]) if xs else None


warm = lambda key: (lambda r, tool: r[key] if ok(r, key) else None)
machine = json.load(open(f"{OUT}/machine.json"))
v = machine["versions"]
print(f"# Scoreboard: {machine['cpu']}, {machine['logical_cpus']} logical CPUs, {machine['memory_gib']} GiB\n")
print(f"{machine['threads']} thread(s) per tool; {machine['system']}; Python {machine['python']}. Versions: "
      + ", ".join(f"{k} {x}" for k, x in v.items()) + ".\n")
print("Each figure is the median of 5 fresh parameter vectors in one fresh process with a time limit. Other tools run "
      "as their authors recommend: PauliPropagation.jl with `VectorPauliSum`, its `Performance` kernels and "
      "`min_abs_coeff = 1e-12`; pauli-prop with `atol = 1e-12`; sparse Pauli dynamics (spd) with its "
      "Clifford parts moved into the Pauli frame once, commuting rotations merged and threshold 1e-12. Every other tool also runs on circuits cut to the "
      "observable's light cone, as one circuit (cone) or one per observable term (split), and counts its fastest "
      "setup. Quantum Circuit Simulator (QCS) runs uncut circuits, exactly and at the same 1e-12 threshold. "
      f"Ratios below {TARGET:.0f}× are in bold. *Repeated* is one more evaluation of a prepared circuit, as in an "
      "optimiser or a sweep; *single* adds the setup of the circuit, QCS graph compilation "
      "included and once-per-process costs such as imports and Julia's compilation excluded.\n")

table, counts = [], defaultdict(int)
for w in order:
    o_rep, t_rep = best(ours[w], warm("warm_value_s")), best(theirs[w], warm("warm_value_s"))
    o_one = best(ours[w], lambda r, tool: single(r, "QCS"))
    t_one = best(theirs[w], single)
    rep_text, rep = ratio(o_rep[0] if o_rep else None, t_rep[0] if t_rep else None)
    one_text, one = ratio(o_one[0] if o_one else None, t_one[0] if t_one else None)
    for name, x, us, them in (("repeated", rep, o_rep, t_rep), ("single", one, o_one, t_one)):
        if x is not None:
            counts[name, x >= TARGET] += 1
        elif us is not None:
            counts[name, "only"] += 1
        elif them is not None:
            counts[name, "lost"] += 1
        else:
            counts[name, "none"] += 1
    table.append(f"| {w} | " + (f"{f(o_rep[0])} ({o_rep[1]}, {o_rep[2]}, {err(w, o_rep[3])})" if o_rep else "refused")
                 + " | " + (f"{t_rep[1]} {f(t_rep[0])} ({t_rep[2]}, {err(w, t_rep[3])})" if t_rep else "none finished")
                 + f" | {rep_text} | " + (f"{f(o_one[0])}" if o_one else "—") + " | "
                 + (f"{t_one[1]} {f(t_one[0])}" if t_one else "—") + f" | {one_text} |")

print("## Summary\n")
for name in ("repeated", "single"):
    print(f"- **{name.capitalize()}:** {counts[name, True]} workloads at {TARGET:.0f}× or more, "
          f"{counts[name, False]} below, {counts[name, 'only']} that only QCS finished, "
          f"{counts[name, 'lost']} that QCS refused and another tool finished, "
          f"{counts[name, 'none']} that no tool finished.")
print("\n## Values\n")
print("| Workload | QCS repeated (backend, mode, error) | Fastest other, repeated (setup, error) | QCS lead | "
      "QCS single | Fastest other, single | QCS lead |")
print("|---|---|---|---|---|---|---|")
print("\n".join(table))

print("\n## Gradients of every shared parameter, repeated\n")
print("No other tool returns gradients through noise channels; those rows show QCS alone.\n")
print("| Workload | QCS (backend, mode) | Fastest other (setup) | QCS lead |")
print("|---|---|---|---|")
for w in order:
    o, t = best(ours[w], warm("warm_grad_s")), best(theirs[w], warm("warm_grad_s"))
    if o is None and t is None:
        continue
    print(f"| {w} | " + (f"{f(o[0])} ({o[1]}, {o[2]})" if o else "refused") + " | "
          + (f"{t[1]} {f(t[0])} ({t[2]})" if t else "none") + f" | {ratio(o[0] if o else None, t[0] if t else None)[0]} |")

print("\n## Every configuration, repeated evaluation\n")
setups = sorted({(tool, setup) for w in order for tool, setup, _ in theirs[w]})
mine = sorted({(b, m) for w in order for b, m, _ in ours[w]})
print("| Workload | " + " | ".join(f"QCS {b} {m}" for b, m in mine) + " | "
      + " | ".join(f"{t} {s}" for t, s in setups) + " |")
print("|---|" + "---|" * (len(mine) + len(setups)))


def cell(r):
    if r is None:
        return "—"
    if ok(r, "warm_value_s"):
        return f(r["warm_value_s"])
    e = r.get("error", "")
    return "timeout" if "timeout" in e else "refused" if ("Overflow" in e or "cap" in e) else "failed"


for w in order:
    a = {(b, m): r for b, m, r in ours[w]}
    b = {(t, s): r for t, s, r in theirs[w]}
    print(f"| {w} | " + " | ".join(cell(a.get(k)) for k in mine) + " | " + " | ".join(cell(b.get(k)) for k in setups) + " |")
