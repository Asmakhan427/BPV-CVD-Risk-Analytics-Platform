"""
Shared, colorblind-safe color tokens used across every static (matplotlib/
seaborn) and interactive (Plotly) visualization in the platform, so all
figures read as one consistent visual system.

Categorical hues are an 8-slot, CVD-validated palette used in fixed order
(never re-cycled by rank). Risk/severity levels use the reserved status
palette (good / warning / critical) since BPV clusters carry an inherent
clinical severity ordering rather than an arbitrary identity.
"""

# Fixed-order categorical palette (validated: adjacent-pair CVD ΔE >= 8 in
# both light and dark modes). Use slots in order; never cycle by rank.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Status / severity palette (fixed — never reused for generic series)
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# BPV cluster severity mapping (ordinal clinical meaning -> status color)
CLUSTER_COLORS = {
    "Low BPV": STATUS["good"],
    "Medium BPV": STATUS["warning"],
    "High BPV": STATUS["critical"],
}
CLUSTER_ORDER = ["Low BPV", "Medium BPV", "High BPV"]

# Sequential single-hue ramp (blue), light -> dark, for magnitude encodings
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Diverging pair (blue <-> red) with neutral gray midpoint
DIVERGING = ["#0d366b", "#256abf", "#6da7ec", "#f0efec", "#ec8f8e", "#d9504f", "#8a2222"]
DIVERGING_MIDPOINT = "#f0efec"

# Chart chrome / ink (light mode defaults; dashboard renders light theme)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Convenience: risk binary (event / no event)
RISK_COLORS = {"CV Event": STATUS["critical"], "No Event": STATUS["good"]}

# Algorithm comparison colors (fixed order, categorical)
ALGORITHM_COLORS = {
    "K-Means": CATEGORICAL[0],
    "PAM": CATEGORICAL[1],
    "Ward": CATEGORICAL[2],
    "EM (GMM)": CATEGORICAL[6],
}

PLOTLY_TEMPLATE = "plotly_white"


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a '#rrggbb' hex string to a Plotly-safe 'rgba(r,g,b,a)' string.

    Plotly's graph_objects color validators (e.g. indicator.gauge.step.color)
    reject 8-digit '#rrggbbaa' hex, so alpha must be expressed via rgba().
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
