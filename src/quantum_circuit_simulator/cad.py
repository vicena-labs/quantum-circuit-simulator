"""Chip layouts as CAD files: GDSII for fabrication tools, DXF for mechanical CAD, SVG for documents,
and STL for a 3D model.

Every device has qubit coordinates (real chips from their published maps,
catalogue layouts from their lattices, custom chips from a force-directed
layout), so any chip in the package, measured or designed, can be drawn:

    from quantum_circuit_simulator import cad, devices
    cad.write(devices.get("ibm_fez"), "fez.gds")          # format from the extension
    cad.write(devices.get("google_willow_pink"), "willow.svg")
    cad.write(devices.get("ibm_fez"), "fez.stl")          # 3D: GitHub and slicers open it directly

Superconducting chips are drawn as cross-shaped (Xmon-style) transmons joined
by straight coupler traces, with a readout pad beside each qubit; trapped-ion,
neutral-atom and annealer chips are drawn as sites joined by their couplings.
Layers: 1 qubits, 2 couplers, 3 readout pads, 10 labels, 0 chip outline.
The STL model is the outline as a silicon slab with every qubit, coupler and
readout pad raised on it; its heights are exaggerated so the circuit can be
seen (`relief`), since real metal films are a fraction of a micrometre thick.

This is a placement layout generated from connectivity, in micrometres at a
chosen pitch, for visualising, documenting and starting a design. It is not a
fabrication-ready mask: real transmon geometry, resonators, flux lines,
airbridges and ground-plane cuts come from electromagnetic design, which
dedicated tools such as Qiskit Metal or KQCircuits perform.
"""
from __future__ import annotations

import math
import struct
import time
from pathlib import Path

LAYERS = {"outline": 0, "qubit": 1, "coupler": 2, "readout": 3, "label": 10}


def layout(device, *, pitch=800.0, arm=220.0, width=30.0, trace=10.0, site=60.0, margin=600.0):
    """Shapes of a chip in micrometres: {"polygons": [(layer, [(x, y), ...])], "paths": [...], "labels": [...]}.

    pitch is the distance between neighbouring lattice points; arm and width size each
    transmon cross; trace is the coupler width; site the radius of non-superconducting sites.
    """
    xy = [(x * pitch, y * pitch) for x, y in device.positions()]
    transmon = device.family == "superconducting"
    polygons, paths, labels = [], [], []
    for q, (x, y) in enumerate(xy):
        if transmon:
            polygons.append((LAYERS["qubit"], _rect(x - arm, y - width / 2, x + arm, y + width / 2)))
            polygons.append((LAYERS["qubit"], _rect(x - width / 2, y - arm, x + width / 2, y + arm)))
            polygons.append((LAYERS["readout"], _rect(x + arm * 0.55, y + arm * 0.55, x + arm * 0.95, y + arm * 0.95)))
        else:
            polygons.append((LAYERS["qubit"], _circle(x, y, site)))
        labels.append((LAYERS["label"], (x - arm * 0.9, y - arm * 0.9), f"Q{q}"))
    reach = arm if transmon else site
    for a, b in device.edges:
        (xa, ya), (xb, yb) = xy[a], xy[b]
        length = math.hypot(xb - xa, yb - ya)
        if length <= 2 * reach:
            continue
        ux, uy = (xb - xa) / length, (yb - ya) / length
        # From the edge of one qubit to the edge of the other.
        paths.append((LAYERS["coupler"], [(xa + ux * reach, ya + uy * reach), (xb - ux * reach, yb - uy * reach)], trace))
    xs = [p[0] for p in xy] or [0.0]
    ys = [p[1] for p in xy] or [0.0]
    pad = margin + reach
    polygons.append((LAYERS["outline"], _rect(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)))
    return {"polygons": polygons, "paths": paths, "labels": labels,
            "size_um": (max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)}


def write(device, path, *, fmt=None, **options):
    """Write a chip layout to `path`; fmt "gds", "dxf", "svg" or "stl", or taken from the extension."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    shapes = layout(device, **options)
    writer = {"gds": _gds, "gdsii": _gds, "dxf": _dxf, "svg": _svg, "stl": _stl}.get(fmt)
    if writer is None:
        raise ValueError("format must be gds, dxf, svg or stl")
    data = writer(device, shapes)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return {"path": str(path), "format": fmt, "qubits": device.n_qubits, "couplers": len(device.edges),
            "size_um": [round(v, 3) for v in shapes["size_um"]], "layers": LAYERS,
            "note": "placement layout from connectivity, not a fabrication-ready mask"}


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _circle(x, y, r, sides=24):
    return [(x + r * math.cos(2 * math.pi * k / sides), y + r * math.sin(2 * math.pi * k / sides)) for k in range(sides)]


# ---------------------------------------------------------------- GDSII
# Database unit 1 nm; user unit 1 um.
DB_PER_UM = 1000


def _gds(device, shapes):
    out = bytearray()
    now = time.gmtime()
    stamp = [now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec] * 2
    out += _record(0x0002, struct.pack(">h", 600))                      # HEADER, version 6
    out += _record(0x0102, struct.pack(">12h", *stamp))                 # BGNLIB
    out += _record(0x0206, _string("QCSIM"))                            # LIBNAME
    out += _record(0x0305, _real8(1 / DB_PER_UM) + _real8(1e-9))        # UNITS: user units per db unit, metres per db unit
    out += _record(0x0502, struct.pack(">12h", *stamp))                 # BGNSTR
    out += _record(0x0606, _string(_cell_name(device.name)))            # STRNAME
    for layer, points in shapes["polygons"]:
        closed = points + [points[0]]
        out += _record(0x0800) + _record(0x0D02, struct.pack(">h", layer)) + _record(0x0E02, struct.pack(">h", 0))
        out += _record(0x1003, _xy(closed)) + _record(0x1100)          # BOUNDARY, LAYER, DATATYPE, XY, ENDEL
    for layer, points, width in shapes["paths"]:
        out += _record(0x0900) + _record(0x0D02, struct.pack(">h", layer)) + _record(0x0E02, struct.pack(">h", 0))
        out += _record(0x0F03, struct.pack(">i", round(width * DB_PER_UM)))
        out += _record(0x1003, _xy(points)) + _record(0x1100)          # PATH ... WIDTH, XY, ENDEL
    for layer, (x, y), text in shapes["labels"]:
        out += _record(0x0C00) + _record(0x0D02, struct.pack(">h", layer)) + _record(0x1602, struct.pack(">h", 0))
        out += _record(0x1003, _xy([(x, y)])) + _record(0x1906, _string(text)) + _record(0x1100)
    out += _record(0x0700) + _record(0x0400)                             # ENDSTR, ENDLIB
    return bytes(out)


def _record(kind, payload=b""):
    return struct.pack(">HH", 4 + len(payload), kind) + payload


def _string(text):
    data = text.encode("ascii", "replace")
    return data + b"\0" if len(data) % 2 else data


def _xy(points):
    return b"".join(struct.pack(">ii", round(x * DB_PER_UM), round(y * DB_PER_UM)) for x, y in points)


def _real8(value):
    """GDSII eight-byte real: sign, excess-64 base-16 exponent, 56-bit mantissa."""
    if value == 0:
        return b"\0" * 8
    sign = 0x80 if value < 0 else 0
    value = abs(value)
    exponent = 64
    while value >= 1:
        value /= 16
        exponent += 1
    while value < 1 / 16:
        value *= 16
        exponent -= 1
    mantissa = round(value * (1 << 56))
    if mantissa >= 1 << 56:   # rounding carried into the next hexadecimal digit
        mantissa >>= 4
        exponent += 1
    return bytes([sign | exponent]) + mantissa.to_bytes(7, "big")


def _cell_name(name):
    clean = "".join(c if c.isalnum() or c in "_-$" else "_" for c in name)
    return clean[:32] or "CHIP"


# ---------------------------------------------------------------- DXF (R12, ASCII)
def _dxf(device, shapes):
    lines = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "13", "0", "ENDSEC",   # 13 = micrometres
             "0", "SECTION", "2", "ENTITIES"]
    names = {v: k.upper() for k, v in LAYERS.items()}
    for layer, points in shapes["polygons"]:
        lines += ["0", "POLYLINE", "8", names[layer], "66", "1", "70", "1"]
        for x, y in points:
            lines += ["0", "VERTEX", "8", names[layer], "10", f"{x:.4f}", "20", f"{y:.4f}"]
        lines += ["0", "SEQEND"]
    for layer, ((x0, y0), (x1, y1)), width in shapes["paths"]:
        lines += ["0", "POLYLINE", "8", names[layer], "66", "1", "70", "0", "40", f"{width:.4f}", "41", f"{width:.4f}"]
        for x, y in ((x0, y0), (x1, y1)):
            lines += ["0", "VERTEX", "8", names[layer], "10", f"{x:.4f}", "20", f"{y:.4f}"]
        lines += ["0", "SEQEND"]
    for layer, (x, y), text in shapes["labels"]:
        lines += ["0", "TEXT", "8", names[layer], "10", f"{x:.4f}", "20", f"{y:.4f}", "40", "60", "1", text]
    lines += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- SVG
COLOURS = {0: "#c8ccd4", 1: "#3b6fd8", 2: "#e0892b", 3: "#4aa564", 10: "#20242c"}


def _svg(device, shapes):
    outline = next(p for layer, p in shapes["polygons"] if layer == LAYERS["outline"])
    x0, y0 = outline[0]
    x1, y1 = outline[2]
    w, h = x1 - x0, y1 - y0
    # SVG's y axis points down; flip so the drawing matches the coordinates.
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {-y1:.1f} {w:.1f} {h:.1f}" '
             f'width="{min(1600, w / 10):.0f}" height="{min(1600, w / 10) * h / w:.0f}">',
             f"<title>{device.name}: {device.n_qubits} qubits, {len(device.edges)} couplers</title>"]
    for layer, points in sorted(shapes["polygons"], key=lambda s: s[0]):
        style = 'fill="none" stroke="#c8ccd4" stroke-width="20"' if layer == 0 else f'fill="{COLOURS[layer]}"'
        parts.append(f'<polygon points="{" ".join(f"{x:.1f},{-y:.1f}" for x, y in points)}" {style}/>')
    for layer, ((xa, ya), (xb, yb)), width in shapes["paths"]:
        parts.append(f'<line x1="{xa:.1f}" y1="{-ya:.1f}" x2="{xb:.1f}" y2="{-yb:.1f}" stroke="{COLOURS[layer]}" '
                     f'stroke-width="{max(width, 12):.1f}"/>')
    for layer, (x, y), text in shapes["labels"]:
        parts.append(f'<text x="{x:.1f}" y="{-y:.1f}" font-size="90" font-family="sans-serif" '
                     f'fill="{COLOURS[layer]}">{text}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- STL
# Heights in micrometres: a substrate slab, then each layer raised on it. The
# relief is exaggerated for viewing; real films are well under a micrometre.
SUBSTRATE_UM = 250.0
RELIEF_UM = {LAYERS["qubit"]: 40.0, LAYERS["coupler"]: 25.0, LAYERS["readout"]: 30.0}


def solids(shapes):
    """(layer, footprint, bottom, top) of every 3D block: the slab, then each shape on it."""
    out = []
    for layer, points in shapes["polygons"]:
        if layer == LAYERS["outline"]:
            out.append((layer, points, 0.0, SUBSTRATE_UM))
        else:
            out.append((layer, points, SUBSTRATE_UM, SUBSTRATE_UM + RELIEF_UM[layer]))
    for layer, ((xa, ya), (xb, yb)), width in shapes["paths"]:
        length = math.hypot(xb - xa, yb - ya)
        nx, ny = -(yb - ya) / length * width / 2, (xb - xa) / length * width / 2
        footprint = [(xa + nx, ya + ny), (xa - nx, ya - ny), (xb - nx, yb - ny), (xb + nx, yb + ny)]
        out.append((layer, footprint, SUBSTRATE_UM, SUBSTRATE_UM + RELIEF_UM[layer]))
    return out


def _stl(device, shapes):
    """Binary STL of the extruded layout, in micrometres."""
    triangles = [t for _, footprint, bottom, top in solids(shapes) for t in _prism(footprint, bottom, top)]
    header = f"{device.name} chip by quantum-circuit-simulator, micrometres".encode()[:80].ljust(80, b" ")
    out = bytearray(header + struct.pack("<I", len(triangles)))
    for a, b, c in triangles:
        out += struct.pack("<12fH", *_normal(a, b, c), *a, *b, *c, 0)
    return bytes(out)


def _prism(footprint, bottom, top):
    """Triangles of a convex footprint extruded from `bottom` to `top`, wound outwards."""
    points = footprint if _area(footprint) > 0 else footprint[::-1]
    low = [(x, y, bottom) for x, y in points]
    high = [(x, y, top) for x, y in points]
    faces = []
    for k in range(1, len(points) - 1):
        faces.append((high[0], high[k], high[k + 1]))
        faces.append((low[0], low[k + 1], low[k]))
    for k in range(len(points)):
        j = (k + 1) % len(points)
        faces += [(low[k], low[j], high[j]), (low[k], high[j], high[k])]
    return faces


def _area(points):
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])) / 2


def _normal(a, b, c):
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    norm = math.sqrt(sum(x * x for x in n)) or 1.0
    return [x / norm for x in n]
