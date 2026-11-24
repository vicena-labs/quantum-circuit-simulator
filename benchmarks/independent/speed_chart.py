"""The README's speed chart, and its data for the Vicena website, from a scoreboard run.

    python speed_chart.py results-2026-11-24-vicena/scoreboard-1t

For each chosen workload: Quantum Circuit Simulator's fastest time for one more
evaluation, against the fastest of the seven other simulators in its best
setup, exactly as summarize_scoreboard.py picks them. Writes
docs/images/speed-{light,dark}.png and speed-chart.json beside the records.
Needs matplotlib; uses Poppins, Vicena's typeface, when it is installed.
"""
import importlib.util, json, math, sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("summary", HERE / "summarize_scoreboard.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

LIMIT = 120.0  # each run's time limit, in seconds
# Workload, plain name, the detail an expert needs. Strongest results first, then
# the two small circuits under a tenfold lead, then a circuit QCS refuses.
ROWS = [
    ("K_eagle_mz_S5_t1pi8", "IBM's 127-qubit utility experiment", "Kicked Ising, 5 steps, θh = π/8, exact"),
    ("K_eagle_mz_S5_t4pi8", "Same experiment at θh = π/2", "A Clifford angle, where Pauli tools do best"),
    ("B_noisy_6x6_L5", "Noisy circuit, 36 qubits, 5 layers", "6 × 6 grid, two-qubit depolarizing noise"),
    ("A_vqe_4x5_L4", "Noiseless circuit, 20 qubits, 4 layers", "Ising energy on a 4 × 5 grid"),
    ("B_noisy_32x32_L3", "Noisy circuit, 1,024 qubits", "32 × 32 grid, 3 layers, mean magnetisation"),
    ("B_noisy_20x20_L3", "Noisy circuit, 400 qubits", "20 × 20 grid, 3 layers"),
    ("B_noisy_2x5_L4", "Noisy circuit, 10 qubits", "2 × 5 grid, 4 layers"),
    ("A_vqe_3x4_L2", "Small circuit, 12 qubits", "2 layers; call overhead dominates"),
    ("C_centre_10x10_L4", "One qubit's value, 100 qubits", "4 layers, centre Z"),
    ("C_centre_10x10_L8", "One qubit's value, 100 qubits, 8 layers", "Too deep for exact Pauli propagation"),
]
SETUP = {"whole": "whole circuit", "cone": "light-cone cut", "split": "one circuit per term"}
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#1a1a19", "muted": "#6b6a66", "grid": "#e4e3df",
              "ours": "#9a6a00", "other": "#256abf"},
    "dark": {"surface": "#1a1a19", "ink": "#f3f2ee", "muted": "#a3a29c", "grid": "#33322f",
             "ours": "#c98500", "other": "#3987e5"},
}


def main(run):
    ours, theirs, _, _, machine = summary.load(run)
    measure = summary.warm("warm_value_s")
    rows = []
    for workload, name, detail in ROWS:
        us, them = summary.best(ours[workload], measure), summary.best(theirs[workload], measure)
        rows.append(dict(workload=workload, name=name, detail=detail,
                         qcs_s=us[0] if us else None,
                         other_s=them[0] if them else None,
                         other_tool=them[1] if them else None,
                         other_setup=them[2] if them else None))
    data = dict(source=f"{Path(run).name} of {Path(run).parent.name}", cpu=machine["cpu"],
                threads=machine["threads"], limit_s=LIMIT, versions=machine["versions"], rows=rows)
    (Path(run).parent / "speed-chart.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    for folder in (Path.home() / "Library" / "Fonts", Path("/Library/Fonts"), Path("/usr/share/fonts"),
                   Path.home() / ".local" / "share" / "fonts"):
        for file in folder.glob("**/Poppins-*.ttf") if folder.is_dir() else []:
            font_manager.fontManager.addfont(str(file))
    font = "Poppins" if any("Poppins" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    for theme_name, theme in THEMES.items():
        draw(rows, machine, theme, font, HERE.parents[1] / "docs" / "images" / f"speed-{theme_name}.png")


def label(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds:.3g} s"


def outcome(row):
    """The right-hand verdict, and whether it is a loss."""
    us, them = row["qcs_s"], row["other_s"]
    if us is None:
        return f"QCS refuses; {row['other_tool']} {label(them)}", True
    if them is None:
        return "No other simulator finished in 2 min", False
    ratio = them / us
    if ratio >= 1000:  # two significant figures: the timings carry no more
        ratio = round(ratio, -int(math.log10(ratio)) + 1)
    return f"{ratio:,.0f}× faster" if ratio >= 10 else f"{ratio:.1f}× faster", False


def draw(rows, machine, theme, font, path):
    plt.rcParams["font.family"] = [font, "DejaVu Sans"]  # DejaVu for glyphs Poppins lacks, such as θ
    n = len(rows)
    height = 1.7 + 0.62 * n
    fig = plt.figure(figsize=(15.5, height), dpi=150, facecolor=theme["surface"])
    top = 1 - 1.45 / height
    ax = fig.add_axes([0.27, 0.6 / height, 0.40, top - 0.6 / height], facecolor=theme["surface"])
    ax.set_xscale("log")
    ax.set_xlim(5e-6, 400)
    ax.set_ylim(n - 0.5, -0.5)
    ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, LIMIT]
    ax.set_xticks(ticks, ["10 µs", "100 µs", "1 ms", "10 ms", "100 ms", "1 s", "10 s", "2-min limit"])
    ax.minorticks_off()
    ax.tick_params(axis="x", colors=theme["muted"], labelsize=9, length=0, pad=6)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for t in ticks[:-1]:
        ax.axvline(t, color=theme["grid"], lw=0.8, zorder=0)
    ax.axvline(LIMIT, color=theme["muted"], lw=1, ls=(0, (3, 3)), zorder=0)
    for i, row in enumerate(rows):
        us, them = row["qcs_s"], row["other_s"]
        left, right = us if us is not None else LIMIT * 1.9, them if them is not None else LIMIT * 1.9
        ax.plot([min(left, right), max(left, right)], [i, i], color=theme["muted"], lw=1.5, alpha=0.45, zorder=1,
                solid_capstyle="round")
        if them is None:  # did not finish: an open marker past the limit
            ax.annotate("", xy=(LIMIT * 2.7, i), xytext=(LIMIT * 1.25, i),
                        arrowprops=dict(arrowstyle="-|>", color=theme["other"], lw=1.5), zorder=2)
            ax.scatter([LIMIT * 1.25], [i], s=70, facecolor=theme["surface"], edgecolor=theme["other"], lw=2, zorder=3)
        else:
            ax.scatter([them], [i], s=70, color=theme["other"], edgecolor=theme["surface"], lw=2, zorder=3)
        if us is None:  # refused: an open marker past the limit
            ax.scatter([LIMIT * 1.9], [i], s=70, facecolor=theme["surface"], edgecolor=theme["ours"], lw=2, zorder=3)
        else:
            ax.scatter([us], [i], s=80, color=theme["ours"], edgecolor=theme["surface"], lw=2, zorder=4)
        y = ax.transData.transform((1, i))[1] / fig.dpi / height
        gap = 0.05 / height
        fig.text(0.02, y + gap, row["name"], color=theme["ink"], fontsize=10.5, weight="semibold", va="bottom")
        fig.text(0.02, y - gap, row["detail"], color=theme["muted"], fontsize=8.8, va="top")
        verdict, lost = outcome(row)
        fig.text(0.69, y + gap, verdict, color=theme["ink"], fontsize=10.5,
                 weight="normal" if lost else "semibold", va="bottom")
        against = (f"QCS {label(us)} · {row['other_tool']} {label(them)}, {SETUP[row['other_setup']]}"
                   if us is not None and them is not None else
                   f"QCS {label(us)}" if us is not None else "Exact mode exceeds its 1-million-term cap")
        fig.text(0.69, y - gap, against, color=theme["muted"], fontsize=8.5, va="top")
    title_y = 1 - 0.3 / height
    fig.text(0.02, title_y, "Time for one more evaluation of a prepared circuit (lower is faster)",
             color=theme["ink"], fontsize=14.5, weight="semibold", va="top")
    fig.text(0.02, title_y - 0.36 / height,
             f"One CPU thread per tool on a Vicena computer ({machine['cpu'].split(' 32-Core')[0]}). "
             "Blue is the fastest of seven other simulators: PauliPropagation.jl, IBM pauli-prop, sparse Pauli dynamics, "
             "quimb, Qiskit Aer, Qulacs and PennyLane Lightning,\neach set up as its authors recommend, at a 1e-12 "
             "coefficient threshold, on whole circuits and on circuits cut to the light cone. "
             "Data and method: benchmarks/independent.", color=theme["muted"], fontsize=8.8, va="top", linespacing=1.5)
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=theme["ours"], label="Quantum Circuit Simulator"),
               Line2D([], [], ls="", marker="o", ms=8, color=theme["other"], label="Fastest other simulator")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5,
              labelcolor=theme["ink"], handletextpad=0.4, columnspacing=1.6, borderaxespad=0.2)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
