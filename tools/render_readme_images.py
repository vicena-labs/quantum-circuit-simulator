"""Render the README's banner, chip figures and 3D chip model from the bundled data.

    python tools/render_readme_images.py

Writes light and dark variants to docs/images/, and IBM Fez's 3D model
(ibm_fez.stl, from `cad.write`), which GitHub shows in an interactive viewer. Qubits are coloured by their
calibrated T1 (one blue ramp, darker is longer in light mode, brighter in
dark mode); couplers are drawn as neutral lines. Needs matplotlib, and
dwave-networkx for the D-Wave panel.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from quantum_circuit_simulator import cad, devices

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#1a1a19", "muted": "#6b6a66", "edge": "#c9c8c3",
              "ramp": LinearSegmentedColormap.from_list("t1", RAMP[1:])},
    "dark": {"surface": "#1a1a19", "ink": "#f3f2ee", "muted": "#a3a29c", "edge": "#4a4945",
             "ramp": LinearSegmentedColormap.from_list("t1", RAMP[::-1][1:])},
}
FONT = {"family": "DejaVu Sans"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        banner(theme, OUT / f"banner-{name}.png")
        chips(theme, OUT / f"chips-{name}.png")
        model(theme, OUT / f"chip-3d-{name}.png")
    cad.write(devices.get("ibm_fez"), OUT / "ibm_fez.stl")


def banner(theme, path):
    """The title beside IBM Fez's 156 qubits."""
    fig = plt.figure(figsize=(16, 4.6), dpi=150, facecolor=theme["surface"])
    ax = fig.add_axes([0.55, 0.06, 0.43, 0.88])
    draw(ax, devices.get("ibm_fez"), theme, size=40, width=1.5)
    fig.text(0.035, 0.66, "Quantum Circuit Simulator", color=theme["ink"], fontsize=40, weight="bold", **FONT)
    fig.text(0.037, 0.50, "by Vicena Labs", color=theme["muted"], fontsize=22, **FONT)
    fig.text(0.037, 0.30, "Noisy quantum circuits on real chips' calibrations,\n"
                          "exact where possible, bounded where not.",
             color=theme["ink"], fontsize=17, linespacing=1.5, **FONT)
    fig.text(0.037, 0.10, "Right: IBM Fez's 156 qubits, coloured by T1 from its 26 Feb 2025 calibration",
             color=theme["muted"], fontsize=11, **FONT)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def chips(theme, path):
    """Three real chips side by side, each with its qubit count and source."""
    panels = [("ibm_fez", "IBM Fez · Heron r2", "156 qubits, coloured by T1", 30, None),
              ("google_willow_pink", "Google Willow", "105 qubits, coloured by T1", 34, None),
              ("dwave-advantage2", "D-Wave Advantage2 · Zephyr", "4,800 qubits; one corner shown", 22, 0.16)]
    fig = plt.figure(figsize=(16, 6.2), dpi=150, facecolor=theme["surface"])
    lefts, widths = [0.02, 0.40, 0.70], [0.34, 0.27, 0.28]
    for left, width, (name, title, subtitle, size, corner) in zip(lefts, widths, panels):
        fig.text(left, 0.93, title, color=theme["ink"], fontsize=15, weight="bold", **FONT)
        fig.text(left, 0.885, subtitle, color=theme["muted"], fontsize=11, **FONT)
        ax = fig.add_axes([left, 0.10, width, 0.74])
        draw(ax, devices.get(name), theme, size=size, width=1.1 if corner is None else 0.6, corner=corner)
    legend(fig, theme)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


# The exploded view's materials: dark silicon and package, silver wiring, blue qubits.
STACK = {"dark": {"surface": "#0b0b0c", "slab": "#1d1e21", "edge": "#3a3c41", "metal": "#c9ced6",
                  "qubit": "#6aa8f0", "coupler": "#e3e6eb", "hole": "#050506", "ink": "#f3f2ee", "muted": "#9a9993"},
         "light": {"surface": "#fcfcfb", "slab": "#2a2b2f", "edge": "#4a4c52", "metal": "#d6dae1",
                   "qubit": "#6aa8f0", "coupler": "#eef0f3", "hole": "#0b0b0c", "ink": "#1a1a19", "muted": "#6b6a66"}}
GAP = 2600.0      # vertical spacing between exploded layers, micrometres
SLAB = 220.0      # each layer's thickness as drawn
RAISE = 90.0      # metal relief as drawn; real films are far thinner


def model(theme, path):
    """An exploded view of IBM Fez's layout: package, wiring, readout, then qubits and couplers.

    The qubit layer is the package's own CAD layout (cad.layout); the readout
    resonators, wiring and package beneath it are drawn from the same qubit
    positions to show how such a chip is built, not from IBM's design.
    """
    look = STACK["dark" if theme["surface"] == THEMES["dark"]["surface"] else "light"]
    shapes = cad.layout(devices.get("ibm_fez"))
    outline = next(points for layer, points in shapes["polygons"] if layer == cad.LAYERS["outline"])
    x0, y0 = outline[0]
    x1, y1 = outline[2]
    qubits = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for layer, pts in shapes["polygons"]
              if layer == cad.LAYERS["qubit"] for a, b in [(pts[0], pts[2])]][::2]
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=look["surface"])
    ax = fig.add_axes([-0.06, -0.08, 0.92, 1.16], projection="3d", facecolor=look["surface"],
                      computed_zorder=False)
    layers = [("Package", _package(x0, y0, x1, y1)), ("Wiring", _wiring(qubits, x0, x1)),
              ("Readout", _readout(qubits)), ("Qubits and couplers", _qubits(shapes))]
    label_heights = []
    for level, (name, features) in enumerate(layers):
        z = level * GAP
        margin = 500.0 if name == "Package" else 0.0
        _block(ax, [(x0 - margin, y0 - margin), (x1 + margin, y0 - margin), (x1 + margin, y1 + margin),
                    (x0 - margin, y1 + margin)], z, z + SLAB, look["slab"], look["edge"])
        for kind, footprint in features:
            if kind == "hole":
                _face(ax, footprint, z + SLAB + 1, look["hole"])
            else:
                _block(ax, footprint, z + SLAB, z + SLAB + RAISE, look[kind], _shade(look[kind], 0.6))
        label_heights.append((name, z + SLAB))
    ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(-0.45 * GAP, len(layers) * GAP))
    ax.set_box_aspect((x1 - x0, y1 - y0, (len(layers) + 0.45) * GAP * 1.5))
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()
    for name, z in label_heights:  # a label at the right edge of each layer
        _label(fig, ax, (x1, (y0 + y1) / 2, z), name, look)
    fig.text(0.04, 0.93, "IBM Fez, layer by layer", color=look["ink"], fontsize=20, weight="bold", **FONT)
    fig.text(0.04, 0.895, "156 transmon qubits and 176 couplers from cad.layout; readout, wiring and package "
                          "drawn from the same positions; heights exaggerated", color=look["muted"], fontsize=11, **FONT)
    fig.savefig(path, facecolor=look["surface"])
    plt.close(fig)


def _qubits(shapes):
    """Transmon crosses and coupler traces from the CAD layout."""
    out = []
    for layer, footprint, _, _ in cad.solids(shapes):
        if layer == cad.LAYERS["qubit"]:
            out.append(("qubit", footprint))
        elif layer == cad.LAYERS["coupler"]:
            out.append(("coupler", footprint))
    return out


def _readout(qubits, turns=5, pitch=34.0, width=12.0, span=300.0):
    """A meandering readout resonator beneath each qubit."""
    out = []
    for x, y in qubits:
        left = x - span / 2
        for k in range(turns):
            yy = y - turns * pitch / 2 + k * pitch
            out.append(("metal", cad._rect(left, yy, left + span, yy + width)))
            side = left + span - width if k % 2 == 0 else left
            out.append(("metal", cad._rect(side, yy, side + width, yy + pitch + width)))
    return out


def _wiring(qubits, x0, x1, width=60.0):
    """One control line per row of qubits, run to the chip's edge."""
    rows = sorted({round(y) for _, y in qubits})
    out = []
    for y in rows:
        for offset in (-260.0, 260.0):
            out.append(("metal", cad._rect(x0 + 150, y + offset, x1 - 150, y + offset + width)))
    return out


def _package(x0, y0, x1, y1, pitch=1300.0, radius=180.0):
    """A grid of through-vias in the package, and pads around its rim."""
    out = []
    for x in np.arange(x0 + pitch, x1 - pitch / 2, pitch):
        for y in np.arange(y0 + pitch, y1 - pitch / 2, pitch):
            out.append(("hole", cad._circle(x, y, radius, sides=18)))
    for x in np.arange(x0 - 300, x1 + 300, 160.0):
        out += [("metal", cad._rect(x, y0 - 420, x + 70, y0 - 330)), ("metal", cad._rect(x, y1 + 330, x + 70, y1 + 420))]
    return out


def _block(ax, footprint, bottom, top, colour, side):
    """A solid: side faces first, then the top, drawn in that order."""
    walls = [[(xa, ya, bottom), (xb, yb, bottom), (xb, yb, top), (xa, ya, top)]
             for (xa, ya), (xb, yb) in zip(footprint, footprint[1:] + footprint[:1])]
    ax.add_collection3d(Poly3DCollection(walls, facecolors=side, edgecolors="none"))
    _face(ax, footprint, top, colour)


def _face(ax, footprint, z, colour):
    ax.add_collection3d(Poly3DCollection([[(x, y, z) for x, y in footprint]], facecolors=colour, edgecolors="none"))


def _label(fig, ax, point, text, look):
    """A leader line from a layer's edge to its name at the right of the figure."""
    from mpl_toolkits.mplot3d import proj3d
    x, y, _ = proj3d.proj_transform(*point, ax.get_proj())
    fx, fy = fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    fig.add_artist(plt.Line2D([fx + 0.01, 0.86], [fy, fy], color=look["muted"], linewidth=0.8))
    fig.add_artist(plt.Line2D([0.86], [fy], marker="o", markersize=4, color=look["ink"]))
    fig.text(0.87, fy, text, color=look["ink"], fontsize=14, va="center", **FONT)


def _shade(colour, factor=0.72):
    """A darker side face, so blocks read as solids."""
    rgb = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)]) * factor
    return "#" + "".join(f"{int(v):02x}" for v in rgb)


def legend(fig, theme):
    """The colour key: one ramp from short to long T1; D-Wave publishes no T1."""
    ax = fig.add_axes([0.02, 0.035, 0.16, 0.022])
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=theme["ramp"])
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.185, 0.04, "T1, short to long (5th to 95th percentile of each chip; D-Wave publishes no T1)",
             color=theme["muted"], fontsize=10, va="center", **FONT)


def draw(ax, device, theme, size, width, corner=None):
    xy = np.asarray(device.positions(), dtype=float)
    xy[:, 1] *= -1  # rows run downwards, as vendors draw their chips
    edges = np.asarray(device.edges)
    keep = np.ones(len(xy), bool)
    if corner is not None:  # a dense chip: show the top-left fraction of its area
        span = xy.max(0) - xy.min(0)
        keep = (xy[:, 0] <= xy[:, 0].min() + corner * span[0]) & (xy[:, 1] >= xy[:, 1].max() - corner * span[1])
        edges = edges[keep[edges].all(1)]
    ax.add_collection(LineCollection(xy[edges], colors=theme["edge"], linewidths=width, zorder=1))
    t1 = _t1(device)
    if t1 is None:
        colours = np.tile(theme["ramp"](0.75), (len(xy), 1))
    else:
        lo, hi = np.percentile(t1, [5, 95])
        colours = theme["ramp"](np.clip((t1 - lo) / (hi - lo), 0, 1))
    ax.scatter(xy[keep, 0], xy[keep, 1], s=size, c=colours[keep], linewidths=0, zorder=2)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.03)
    ax.axis("off")


def _t1(device):
    overrides = device.to_dict().get("qubit_overrides") or []
    values = {o["qubit"]: o["t1"] for o in overrides if o.get("t1")}
    if len(values) < device.n_qubits // 2:
        return None
    default = np.median(list(values.values()))
    return np.array([values.get(q, default) for q in range(device.n_qubits)])


if __name__ == "__main__":
    main()
