#!/usr/bin/env python3
"""
Generate poster-ready PNG charts for "One City, Unequal Recovery".

Run from the project root:
    python3 Model/generate_poster.py

Outputs:
    Model/poster_output/*.png

Requirements:
    pip install pandas numpy matplotlib
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------
# Paths and visual style
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Model" / "data" / "processed"
TABLES = ROOT / "Model" / "output" / "tables"
OUT = ROOT / "Model" / "poster_output"
OUT.mkdir(parents=True, exist_ok=True)

# Choose "white" for a white poster background or "transparent" for PNG transparency.
BACKGROUND_MODE = "white"

SHAPES_FILE = ROOT / "apps" / "website" / "public" / "data" / "lines" / "tube-shapes.json"

# Light-theme palette: dark text, subtle gridlines, and clear chart colours.
BG = "white"
PANEL = "white"
FG = "#17212B"
MUTED = "#586574"
GRID = "#D9E0E7"
RED = "#B33A3A"
ORANGE = "#D97736"
YELLOW = "#D5A928"
LIGHT_GREEN = "#70A66A"
GREEN = "#287A50"
BLUE = "#3979A8"
PURPLE = "#8064A8"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.labelcolor": FG,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 10,
    "figure.dpi": 180,
    "savefig.dpi": 300,
})

DAY_ORDER = ["Mon-Thu", "Friday", "Saturday", "Sunday"]
DAY_LABELS = ["Mon–Thu", "Friday", "Saturday", "Sunday"]
DAY_COLORS = {
    "Mon-Thu": BLUE,
    "Friday": ORANGE,
    "Saturday": LIGHT_GREEN,
    "Sunday": PURPLE,
}
CLUSTER_COLORS = {"0": BLUE, "1": ORANGE, 0: BLUE, 1: ORANGE}

# Network-level passenger volume as a percentage of the 2019 baseline.
# 2019 is indexed to 100 for every day type.
NETWORK_RECOVERY = {
    "Mon-Thu": {2019: 100.0, 2022: 76.0, 2025: 82.5},
    "Friday":  {2019: 100.0, 2022: 71.8, 2025: 75.2},
    "Saturday":{2019: 100.0, 2022: 89.5, 2025: 95.5},
    "Sunday":  {2019: 100.0, 2022: 89.0, 2025: 95.1},
}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def save_fig(fig, filename):
    path = OUT / filename
    transparent = BACKGROUND_MODE.lower() == "transparent"
    if transparent:
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.patch.set_alpha(0)
    fig.savefig(
        path, dpi=300, bbox_inches="tight", pad_inches=0.22,
        transparent=transparent, facecolor="white" if not transparent else "none"
    )
    plt.close(fig)
    print(f"Saved: {path} ({'transparent' if transparent else 'white background'})")


def style_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def add_source(fig, text="Source: Transport for London (TfL) Open Data; authors' analysis"):
    fig.text(0.01, 0.012, text, color=MUTED, fontsize=8, ha="left")


def check_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


# ---------------------------------------------------------------------
# 01. Network recovery trend
# ---------------------------------------------------------------------

def chart_01_recovery_trend():
    years = [2019, 2022, 2025]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for day in DAY_ORDER:
        vals = [NETWORK_RECOVERY[day][y] for y in years]
        ax.plot(
            years, vals, marker="o", markersize=8, linewidth=2.8,
            color=DAY_COLORS[day], label=day.replace("Mon-Thu", "Mon–Thu")
        )
        for x, y in zip(years, vals):
            ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        color=FG, fontsize=9)

    ax.axhline(100, color=MUTED, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xticks(years, ["2019\nBaseline", "2022", "2025"])
    ax.set_ylabel("Passenger volume (% of 2019)")
    ax.set_title("01  |  The network recovered at different speeds")
    ax.set_ylim(60, 110)
    style_axis(ax)
    ax.legend(frameon=False, ncol=4, loc="lower right", labelcolor=FG)
    add_source(fig, "Source: TfL Annual Counts (2019, 2022, 2025); network totals indexed to 2019 = 100")
    save_fig(fig, "01_recovery_trend.png")


# ---------------------------------------------------------------------
# 02. Day-type recovery comparison
# ---------------------------------------------------------------------

def chart_02_day_type_recovery():
    days = DAY_ORDER
    vals_2022 = [NETWORK_RECOVERY[d][2022] for d in days]
    vals_2025 = [NETWORK_RECOVERY[d][2025] for d in days]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(days))
    width = 0.34
    b1 = ax.bar(x - width / 2, vals_2022, width, color="#64798C", label="2022")
    b2 = ax.bar(x + width / 2, vals_2025, width, color=LIGHT_GREEN, label="2025")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%",
                    ha="center", va="bottom", color=FG, fontsize=9)

    ax.axhline(100, color=MUTED, linestyle="--", linewidth=1.2, label="2019 baseline")
    ax.set_xticks(x, DAY_LABELS)
    ax.set_ylabel("Passenger volume (% of 2019)")
    ax.set_ylim(0, 112)
    ax.set_title("01  |  Weekends are closest to the pre-pandemic baseline")
    style_axis(ax)
    ax.legend(frameon=False, ncol=3, labelcolor=FG)
    add_source(fig, "Source: TfL Annual Counts (2019, 2022, 2025); network totals indexed to 2019 = 100")
    save_fig(fig, "01_day_type_recovery.png")


# ---------------------------------------------------------------------
# 03. Average 15-minute station profiles by cluster
# ---------------------------------------------------------------------

def time_to_minutes(label):
    try:
        # NUMBAT time labels are formatted as HHMM-HHMM, e.g. 0500-0515.
        start = str(label).split("-")[0]
        if ":" in start:
            hh, mm = start.split(":")
        else:
            hh, mm = start[:2], start[2:4]
        return int(hh) * 60 + int(mm)
    except Exception:
        return np.nan


def chart_03_cluster_profiles():
    df = read_csv(TABLES / "cluster_average_profiles.csv")
    check_columns(df, ["Day_Type", "Cluster", "Time", "Flow_Share"], "cluster_average_profiles.csv")

    df = df.copy()
    df["Cluster"] = df["Cluster"].astype(str)
    df["Minute"] = df["Time"].map(time_to_minutes)
    df = df.dropna(subset=["Minute", "Flow_Share"])

    day_types = [d for d in DAY_ORDER if d in df["Day_Type"].unique()]
    if not day_types:
        day_types = sorted(df["Day_Type"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, day in zip(axes, day_types):
        part = df[df["Day_Type"] == day]
        for cluster in sorted(part["Cluster"].unique()):
            series = part[part["Cluster"] == cluster].sort_values("Minute")
            ax.plot(series["Minute"] / 60, series["Flow_Share"] * 100,
                    color=CLUSTER_COLORS.get(cluster, MUTED), linewidth=2.2,
                    label=f"Cluster {cluster}")
        ax.set_title(day.replace("Mon-Thu", "Mon–Thu"))
        ax.set_ylabel("Share of daily flow (%)")
        ax.set_xlim(5, 24)
        ax.set_xticks([6, 9, 12, 15, 18, 21, 24],
                      ["06:00", "09:00", "12:00", "15:00", "18:00", "21:00", "24:00"])
        style_axis(ax)
        ax.legend(frameon=False, labelcolor=FG, fontsize=9)

    for ax in axes[len(day_types):]:
        ax.set_visible(False)

    fig.suptitle("02  |  Two station profiles show different daily rhythms",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.supxlabel("Time of day")
    add_source(fig, "Source: TfL NUMBAT 15-minute entry/exit profiles; 2019 station clusters")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_fig(fig, "02_cluster_profiles.png")


# ---------------------------------------------------------------------
# 04. Cluster-level recovery comparison
# ---------------------------------------------------------------------

def chart_04_cluster_recovery():
    df = read_csv(TABLES / "cluster_recovery_comparison.csv")
    check_columns(df, ["Day_Type", "Year", "0", "1"], "cluster_recovery_comparison.csv")
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].isin([2019, 2022, 2025])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.ravel()
    for ax, day in zip(axes, DAY_ORDER):
        part = df[df["Day_Type"] == day].sort_values("Year")
        if part.empty:
            ax.set_visible(False)
            continue
        ax.plot(part["Year"], part["0"], marker="o", linewidth=2.4,
                color=BLUE, label="Cluster 0")
        ax.plot(part["Year"], part["1"], marker="o", linewidth=2.4,
                color=ORANGE, label="Cluster 1")
        ax.axhline(100, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(day.replace("Mon-Thu", "Mon–Thu"))
        ax.set_xticks([2019, 2022, 2025])
        ax.set_ylabel("Recovery index (% of 2019)")
        style_axis(ax)
        ax.legend(frameon=False, labelcolor=FG, fontsize=9)

    fig.suptitle("03  |  Recovery differences between clusters are small",
                 fontsize=16, fontweight="bold", y=0.99)
    add_source(fig, "Source: TfL NUMBAT and annual counts; cluster-level recovery index")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_fig(fig, "03_cluster_recovery_comparison.png")


# ---------------------------------------------------------------------
# 05. Cluster transition matrix
# ---------------------------------------------------------------------

def chart_05_cluster_transitions():
    df = read_csv(TABLES / "part2_cluster_transitions.csv")
    check_columns(df, ["2019_Cluster", "0", "1"], "part2_cluster_transitions.csv")
    matrix = df.set_index("2019_Cluster")[["0", "1"]].astype(float)
    matrix.index = [f"2019 Cluster {x}" for x in matrix.index]
    matrix.columns = ["2025 Cluster 0", "2025 Cluster 1"]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    im = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("04  |  Most stations retained their original profile")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iloc[i, j])
            ax.text(j, i, f"{value}\n({value / matrix.values.sum() * 100:.1f}%)",
                    ha="center", va="center",
                    color="white" if value > matrix.values.max() * 0.45 else "#17212B",
                    fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of stations", color=FG)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=MUTED)
    add_source(fig, "Source: authors' K-means station-profile classification; K=2")
    save_fig(fig, "04_cluster_transition_matrix.png")


# ---------------------------------------------------------------------
# Geographic map helpers (light theme, larger maps, TfL line colours)
# ---------------------------------------------------------------------

LINE_COLORS = {
    "bakerloo": "#9A5700",
    "central": "#D71920",
    "circle": "#C49A00",              # darker gold for white backgrounds
    "district": "#007A3D",
    "hammersmith-city": "#D66B8A",   # darker rose for contrast
    "jubilee": "#747D85",
    "metropolitan": "#8A004F",
    "northern": "#202020",
    "piccadilly": "#174A9C",
    "victoria": "#007EAD",
    "waterloo-city": "#438F79",
}

LINE_LABELS = {
    "bakerloo": "Bakerloo",
    "central": "Central",
    "circle": "Circle",
    "district": "District",
    "hammersmith-city": "Hammersmith & City",
    "jubilee": "Jubilee",
    "metropolitan": "Metropolitan",
    "northern": "Northern",
    "piccadilly": "Piccadilly",
    "victoria": "Victoria",
    "waterloo-city": "Waterloo & City",
}


def extract_segments(obj):
    """Recursively find separate arrays of [latitude, longitude] pairs."""
    found = []
    if isinstance(obj, list):
        is_coordinate_sequence = (
            len(obj) >= 2
            and all(
                isinstance(p, (list, tuple)) and len(p) >= 2
                and isinstance(p[0], (int, float))
                and isinstance(p[1], (int, float))
                for p in obj
            )
        )
        if is_coordinate_sequence:
            found.append(obj)
        else:
            for item in obj:
                found.extend(extract_segments(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            found.extend(extract_segments(value))
    return found


def normalise_line_key(name):
    return str(name).lower().replace("_", "-").replace(" ", "-").replace("&", "and")


def draw_tube_lines(ax, longitude_factor):
    """Draw route geometry beneath station markers and return legend handles."""
    if not SHAPES_FILE.exists():
        print(f"WARNING: tube-shapes.json not found: {SHAPES_FILE}")
        return []

    try:
        with open(SHAPES_FILE, "r", encoding="utf-8") as f:
            shapes = json.load(f)
    except Exception as exc:
        print(f"WARNING: Could not read tube-shapes.json: {exc}")
        return []

    handles = []
    for line_name, geometry in shapes.items():
        key = normalise_line_key(line_name)
        # Support both 'hammersmith-city' and 'hammersmith-and-city' naming.
        if key == "hammersmith-and-city":
            key = "hammersmith-city"
        if key == "waterloo-and-city":
            key = "waterloo-city"
        if key not in LINE_COLORS:
            continue

        color = LINE_COLORS[key]
        for segment in extract_segments(geometry):
            arr = np.asarray(segment, dtype=float)
            lat = arr[:, 0]
            lon = arr[:, 1]
            valid = (
                np.isfinite(lat) & np.isfinite(lon)
                & (lon >= -0.7) & (lon <= 0.35)
                & (lat >= 51.2) & (lat <= 51.8)
            )
            # Avoid joining distant points after geographic clipping.
            if valid.sum() < 2:
                continue
            indices = np.where(valid)[0]
            breaks = np.where(np.diff(indices) > 1)[0] + 1
            for chunk in np.split(indices, breaks):
                if len(chunk) < 2:
                    continue
                ax.plot(
                    lon[chunk] * longitude_factor,
                    lat[chunk],
                    color=color,
                    linewidth=2.0,
                    alpha=0.82,
                    solid_capstyle="round",
                    zorder=1,
                )

        handles.append(
            Line2D([0], [0], color=color, linewidth=3.2,
                   label=LINE_LABELS[key])
        )
    return handles


def add_station_legend(ax):
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3979A8",
               markeredgecolor="#24313D", markersize=7, label="Cluster 0"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#D97736",
               markeredgecolor="#24313D", markersize=7, label="Cluster 1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#59636E", markersize=8, label="Not comparable"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#008C95", markeredgewidth=1.8, markersize=11,
               label="Elizabeth line interchange"),
    ]
    ax.legend(
        handles=handles, loc="upper left", frameon=True,
        facecolor="white", edgecolor="#CBD3DC", framealpha=0.96,
        labelcolor=FG, fontsize=10,
    )


def map_base(ax, df, longitude_factor):
    line_handles = draw_tube_lines(ax, longitude_factor)

    comparable = df["Comparison_Status"].astype(str).str.strip().str.lower().eq("comparable")
    cluster0 = df["Cluster"].astype(str).eq("0")
    cluster1 = df["Cluster"].astype(str).eq("1")

    # Station marker shape = cluster; marker colour = flow/recovery value.
    for mask, marker in [(comparable & cluster0, "o"), (comparable & cluster1, "s")]:
        part = df[mask & df["_map_value"].notna()]
        if not part.empty:
            ax.scatter(
                part["_X"], part["Latitude"], marker=marker,
                s=72, c=part["_map_value"], cmap=ax._map_cmap,
                norm=ax._map_norm, edgecolors="#FFFFFF",
                linewidths=0.85, alpha=0.98, zorder=4,
            )

    # Stations with incomparable data are hollow and do not receive a value colour.
    part = df[~comparable]
    if not part.empty:
        ax.scatter(
            part["_X"], part["Latitude"], marker="o", s=76,
            facecolors="white", edgecolors="#697581",
            linewidths=1.6, zorder=5,
        )

    # Outer teal ring identifies an Elizabeth line interchange.
    eliz = df["Elizabeth_Line_Interchange"].fillna(False).astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    part = df[eliz]
    if not part.empty:
        ax.scatter(
            part["_X"], part["Latitude"], s=112, marker="o",
            facecolors="none", edgecolors="#008C95",
            linewidths=1.7, zorder=6,
        )

    # Unclassified stations: hollow diamond.
    unclassified = df["Cluster"].isna() | df["Cluster"].astype(str).isin(["nan", "None", ""])
    part = df[unclassified]
    if not part.empty:
        ax.scatter(
            part["_X"], part["Latitude"], marker="D", s=58,
            facecolors="white", edgecolors="#4B5966",
            linewidths=1.2, zorder=7,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("", color=FG)
    ax.set_ylabel("", color=FG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_station_legend(ax)
    return line_handles


def chart_map(df, year):
    df = df.copy()
    for col in ["Longitude", "Latitude", "Flow_2019", "Flow_2022", "Flow_2025"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Longitude", "Latitude", "Flow_2019", "Flow_2022", "Flow_2025"])
    df = df[
        df["Longitude"].between(-0.7, 0.35)
        & df["Latitude"].between(51.2, 51.8)
    ].copy()

    # Correct longitude scale at London's latitude.
    longitude_factor = np.cos(np.radians(df["Latitude"].mean()))
    df["_X"] = df["Longitude"] * longitude_factor

    if year == 2019:
        values = df["Flow_2019"].clip(lower=1)
        title = "06  |  2019 baseline passenger flow"
        subtitle = "Annualised station entries and exits before the pandemic"
        cbar_label = "Annualised entries + exits (passengers, log scale)"
        cmap = "viridis"
        norm = LogNorm(
            vmin=max(1, values.quantile(0.02)),
            vmax=max(values.quantile(0.98), values.quantile(0.02) + 1),
        )
        filename = "06_map_2019.png"
    elif year == 2022:
        values = df["Flow_2022"] / df["Flow_2019"] * 100
        title = "07  |  Passenger recovery in 2022"
        subtitle = "Station passenger flow as a percentage of the 2019 baseline"
        cbar_label = "2022 flow as % of 2019"
        cmap = "RdYlGn"
        norm = TwoSlopeNorm(vmin=0, vcenter=100, vmax=150)
        filename = "07_map_2022.png"
    else:
        values = df["Flow_2025"] / df["Flow_2019"] * 100
        title = "08  |  Passenger recovery in 2025"
        subtitle = "Station passenger flow as a percentage of the 2019 baseline"
        cbar_label = "2025 flow as % of 2019"
        cmap = "RdYlGn"
        norm = TwoSlopeNorm(vmin=0, vcenter=100, vmax=150)
        filename = "08_map_2025.png"

    df["_map_value"] = values.replace([np.inf, -np.inf], np.nan)
    comparable = df["Comparison_Status"].astype(str).str.strip().str.lower().eq("comparable")
    df.loc[~comparable, "_map_value"] = np.nan

    # Large canvas gives the central routes and station markers more room.
    fig, ax = plt.subplots(figsize=(16, 13), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax._map_cmap = cmap
    ax._map_norm = norm

    line_handles = map_base(ax, df, longitude_factor)

    # Consistent geographic bounds with a little breathing room.
    x_min, x_max = df["_X"].min(), df["_X"].max()
    y_min, y_max = df["Latitude"].min(), df["Latitude"].max()
    x_pad = (x_max - x_min) * 0.055
    y_pad = (y_max - y_min) * 0.055
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.025, shrink=0.82)
    cbar.set_label(cbar_label, color=FG, fontsize=12)
    cbar.ax.tick_params(colors=MUTED, labelsize=10)
    cbar.outline.set_edgecolor(GRID)

    ax.set_title(title, loc="left", pad=22, color=FG, fontsize=25, fontweight="bold")
    ax.text(
        0, 1.012, subtitle, transform=ax.transAxes,
        ha="left", va="bottom", color=MUTED, fontsize=13,
    )

    # TfL line legend sits below the map; no station names are annotated.
    if line_handles:
        fig.legend(
            handles=line_handles, loc="lower center",
            bbox_to_anchor=(0.5, 0.025), ncol=6,
            frameon=False, fontsize=11, labelcolor=FG,
            handlelength=2.4, columnspacing=1.8,
        )

    fig.text(
        0.5, 0.075,
        "Station shape = 2019 cluster  |  Hollow circle = not comparable  |  Teal ring = Elizabeth line interchange",
        ha="center", color=MUTED, fontsize=10,
    )
    fig.text(
        0.01, 0.008,
        "Source: TfL Annual Counts; station flow = annualised entries + exits. No station names displayed.",
        ha="left", color=MUTED, fontsize=8,
    )
    fig.subplots_adjust(left=0.035, right=0.91, top=0.91, bottom=0.17)
    save_fig(fig, filename)


# ---------------------------------------------------------------------
# 09. Weakest stations by Friday recovery
# ---------------------------------------------------------------------

def chart_09_weakest_stations(df):
    check_columns(df, ["Station", "Recovery_Friday", "Include_in_Ranking"], "part3_final_map.csv")
    part = df.copy()
    include = part["Include_in_Ranking"].astype(str).str.lower().isin(["true", "1", "yes"])
    part = part[include].copy()
    part["Recovery_Friday"] = pd.to_numeric(part["Recovery_Friday"], errors="coerce")
    part = part.dropna(subset=["Recovery_Friday"])
    part = part.nsmallest(10, "Recovery_Friday").sort_values("Recovery_Friday")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bars = ax.barh(part["Station"], part["Recovery_Friday"],
                   color=[RED if v < 60 else ORANGE for v in part["Recovery_Friday"]])
    ax.axvline(100, color=MUTED, linestyle="--", linewidth=1.2, label="2019 baseline")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.0f}%",
                va="center", color=FG, fontsize=9)
    ax.set_xlim(0, max(115, part["Recovery_Friday"].max() + 18))
    ax.set_xlabel("Friday passenger volume (% of 2019)")
    ax.set_title("08  |  Lowest Friday Recovery Stations")
    style_axis(ax, "x")
    ax.legend(frameon=False, labelcolor=FG)
    add_source(fig, "Source: TfL Annual Counts; comparable stations included in ranking only")
    save_fig(fig, "08_weakest_stations.png")


# ---------------------------------------------------------------------
# 10. Distribution of station recovery
# ---------------------------------------------------------------------

def chart_10_recovery_distribution(df):
    check_columns(df, ["Recovery_Index", "Comparison_Status"], "part3_final_map.csv")
    part = df.copy()
    comparable = part["Comparison_Status"].astype(str).str.lower().eq("comparable")
    part = part[comparable]
    part["Recovery_Index"] = pd.to_numeric(part["Recovery_Index"], errors="coerce")
    values = part["Recovery_Index"].dropna()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.hist(values, bins=np.arange(20, 221, 10), color=BLUE,
            edgecolor="white", alpha=0.95)
    ax.axvline(100, color=YELLOW, linestyle="--", linewidth=2,
               label="2019 baseline (100%)")
    median = values.median()
    ax.axvline(median, color=ORANGE, linestyle="-", linewidth=1.8,
               label=f"Median: {median:.1f}%")
    ax.set_xlabel("Station recovery index (% of 2019)")
    ax.set_ylabel("Number of stations")
    ax.set_title("09  |  Recovery Distribution")
    style_axis(ax)
    ax.legend(frameon=False, labelcolor=FG)
    add_source(fig, "Source: TfL Annual Counts; incomparable stations excluded. Recovery index = 2025 / 2019 × 100.")
    save_fig(fig, "09_recovery_distribution.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("\nGenerating poster graphics...")
    chart_01_recovery_trend()
    chart_02_day_type_recovery()
    chart_03_cluster_profiles()
    chart_04_cluster_recovery()
    chart_05_cluster_transitions()

    map_df = read_csv(TABLES / "part3_final_map.csv")
    check_columns(map_df, [
        "Station", "Flow_2019", "Flow_2022", "Flow_2025",
        "Longitude", "Latitude", "Cluster",
        "Elizabeth_Line_Interchange", "Comparison_Status",
    ], "part3_final_map.csv")

    chart_map(map_df, 2019)
    chart_map(map_df, 2022)
    chart_map(map_df, 2025)
    chart_09_weakest_stations(map_df)
    chart_10_recovery_distribution(map_df)

    print("\nAll done.")
    print(f"PNG files are in: {OUT}")
    print("If the map PNGs show station dots without line routes, check that tube-shapes.json exists at:")
    print(SHAPES_FILE)


if __name__ == "__main__":
    main()
