"""Test compositions for the GST showcase feasibility test (digital DiffCSP only).

A: GeTe-Sb2Te3 pseudo-binary (GeTe)_m(Sb2Te3)_n, one formula unit per cell (<= 20 atoms).
B: Ge2Sb2Te5 (9-atom host) + one dopant X, interstitial (host + X, 10 atoms) and
   substitutional (one host atom replaced by X, 9 atoms) on a chemically plausible site.
"""
from __future__ import annotations

GST_SERIES = {  # name: (m GeTe, n Sb2Te3)
    "GeSb2Te4": (1, 1),
    "Ge2Sb2Te5": (2, 1),
    "Ge3Sb2Te6": (3, 1),
    "GeSb4Te7": (1, 2),
    "Ge4Sb2Te7": (4, 1),
    "Ge3Sb4Te9": (3, 2),
    "GeSb6Te10": (1, 3),
    "Ge5Sb2Te8": (5, 1),   # high-GeTe side, added 2026-10-19 (15 / 17 / 19 atoms, not in MP)
    "Ge6Sb2Te9": (6, 1),
    "Ge7Sb2Te10": (7, 1),
}

ENDPOINTS = {"GeTe": (1, 0), "Sb2Te3": (0, 1)}  # end members, group "gst_end" (mixing energy = 0 reference)

HOST = {"Ge": 2, "Sb": 2, "Te": 5}
DOPANTS = ["C", "N", "B", "Si", "Ti", "Sc"]
# substituted host element per dopant (test choice; the full study can enumerate all three)
SUB_SITE = {"C": "Ge", "N": "Te", "B": "Sb", "Si": "Ge", "Ti": "Ge", "Sc": "Sb"}


def gst_counts(m: int, n: int) -> dict[str, int]:
    return {el: k for el, k in {"Ge": m, "Sb": 2 * n, "Te": m + 3 * n}.items() if k}


def all_compositions() -> dict[str, dict]:
    """name -> {'counts': {el: n}, 'group': ..., 'dopant': ..., 'mode': ...}"""
    out = {}
    for name, (m, n) in GST_SERIES.items():
        out[name] = {"counts": gst_counts(m, n), "group": "gst_series", "dopant": None, "mode": None}
    for name, (m, n) in ENDPOINTS.items():
        out[name] = {"counts": gst_counts(m, n), "group": "gst_end", "dopant": None, "mode": None}
    for x in DOPANTS:
        c = dict(HOST)
        c[x] = 1
        out[f"GST+{x}_int"] = {"counts": c, "group": "doped", "dopant": x, "mode": "interstitial"}
        c = dict(HOST)
        c[SUB_SITE[x]] -= 1
        c[x] = 1
        out[f"GST+{x}_sub{SUB_SITE[x]}"] = {"counts": c, "group": "doped", "dopant": x,
                                             "mode": f"substitution@{SUB_SITE[x]}"}
    return out
