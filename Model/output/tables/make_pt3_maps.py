
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D


# --------------------------------------------------
# 0. Visual settings
# --------------------------------------------------

# Choose "white" or "transparent"
BACKGROUND_MODE = "white"

if BACKGROUND_MODE not in ["white", "transparent"]:
    raise ValueError("BACKGROUND_MODE must be 'white' or 'transparent'")

if BACKGROUND_MODE == "white":
    FIGURE_BG = "#FFFFFF"
    AXES_BG = "#FFFFFF"
    TITLE_COLOR = "#17212B"
    TEXT_COLOR = "#4B5563"
    TICK_COLOR = "#374151"
    LEGEND_BG = "#FFFFFF"
    LEGEND_EDGE = "#D1D5DB"
    STATION_EDGE = "#4F4F4F"
    INCOMPARABLE_EDGE = "#64748B"
else:
    FIGURE_BG = "none"
    AXES_BG = "none"
    TITLE_COLOR = "#17212B"
    TEXT_COLOR = "#4B5563"
    TICK_COLOR = "#374151"
    LEGEND_BG = "none"
    LEGEND_EDGE = "none"
    STATION_EDGE = "#FFFFFF"
    INCOMPARABLE_EDGE = "#64748B"


# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "part3_final_map.csv"

PROJECT_ROOT = SCRIPT_DIR.parents[3]
PROJECT_DIR = PROJECT_ROOT / "do-starter-playground"

OUTPUT_DIR = (
    PROJECT_DIR
    / "Model"
    / "poster_output"
)

# TfL route geometry
LINES_PATH = (
    PROJECT_DIR
    / "apps"
    / "website"
    / "public"
    / "data"
    / "lines"
    / "tube-shapes.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. TfL line colours
# Adapted for light backgrounds
# --------------------------------------------------

LINE_COLORS = {
    "bakerloo": "#A85B08",
    "central": "#D71920",
    "circle": "#C49A00",
    "district": "#007A3D",
    "hammersmith-city": "#C76B8A",
    "jubilee": "#777F87",
    "metropolitan": "#8E174F",
    "northern": "#30343B",
    "piccadilly": "#174A8B",
    "victoria": "#007EAD",
    "waterloo-city": "#398C79",
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


# --------------------------------------------------
# 3. Read station data
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

required_columns = [
    "Station",
    "Longitude",
    "Latitude",
    "Flow_2019",
    "Flow_2022",
    "Flow_2025",
    "Comparison_Status",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

for col in [
    "Longitude",
    "Latitude",
    "Flow_2019",
    "Flow_2022",
    "Flow_2025",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["Longitude", "Latitude", "Flow_2019"]
).copy()

# Keep stations around Greater London.
df = df[
    df["Longitude"].between(-0.7, 0.35)
    & df["Latitude"].between(51.2, 51.8)
].copy()

df["Recovery_2022"] = (
    df["Flow_2022"] / df["Flow_2019"] * 100
)

df["Recovery_2025"] = (
    df["Flow_2025"] / df["Flow_2019"] * 100
)

df["Is_Comparable"] = (
    df["Comparison_Status"]
    .fillna("")
    .str.strip()
    .str.lower()
    .eq("comparable")
)

print("Stations loaded:", len(df))
print("Comparable stations:", df["Is_Comparable"].sum())


# --------------------------------------------------
# 4. Read and prepare Tube line geometry
# --------------------------------------------------

if not LINES_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find tube-shapes.json:\n{LINES_PATH}"
    )

with open(LINES_PATH, "r", encoding="utf-8") as f:
    line_shapes = json.load(f)


def extract_segments(obj):
    """
    Extract separate route segments from nested JSON.

    Each coordinate in tube-shapes.json is:
    [latitude, longitude]

    Keep separate route segments separate.
    """

    segments = []

    if isinstance(obj, dict):
        for value in obj.values():
            segments.extend(extract_segments(value))

    elif isinstance(obj, list):

        if (
            len(obj) > 0
            and isinstance(obj[0], (list, tuple))
            and len(obj[0]) >= 2
            and isinstance(obj[0][0], (int, float))
            and isinstance(obj[0][1], (int, float))
        ):
            valid_points = []

            for point in obj:
                if (
                    isinstance(point, (list, tuple))
                    and len(point) >= 2
                    and isinstance(point[0], (int, float))
                    and isinstance(point[1], (int, float))
                ):
                    lat, lon = point[0], point[1]

                    if (
                        -0.7 <= lon <= 0.35
                        and 51.2 <= lat <= 51.8
                    ):
                        valid_points.append((lat, lon))

            if len(valid_points) >= 2:
                segments.append(valid_points)

        else:
            for item in obj:
                segments.extend(extract_segments(item))

    return segments


# --------------------------------------------------
# 5. Shared geographic layout
# --------------------------------------------------

mean_latitude = df["Latitude"].mean()
longitude_factor = np.cos(np.radians(mean_latitude))

df["X"] = df["Longitude"] * longitude_factor
df["Y"] = df["Latitude"]

x_min = df["X"].min()
x_max = df["X"].max()
y_min = df["Y"].min()
y_max = df["Y"].max()

x_padding = (x_max - x_min) * 0.07
y_padding = (y_max - y_min) * 0.07

MAP_EXTENT = (
    x_min - x_padding,
    x_max + x_padding,
    y_min - y_padding,
    y_max + y_padding,
)


# --------------------------------------------------
# 6. Draw Tube lines
# --------------------------------------------------

def draw_tube_lines(ax):

    line_handles = []

    for line_name, line_color in LINE_COLORS.items():

        if line_name not in line_shapes:
            print(
                f"Warning: {line_name} not found in tube-shapes.json"
            )
            continue

        segments = extract_segments(line_shapes[line_name])

        for segment in segments:

            latitudes = [point[0] for point in segment]
            longitudes = [point[1] for point in segment]

            x_values = np.array(longitudes) * longitude_factor
            y_values = np.array(latitudes)

            ax.plot(
                x_values,
                y_values,
                color=line_color,
                linewidth=1.9,
                alpha=0.90,
                solid_capstyle="round",
                zorder=1,
            )

        line_handles.append(
            Line2D(
                [0],
                [0],
                color=line_color,
                linewidth=3,
                label=LINE_LABELS[line_name],
            )
        )

    return line_handles


# --------------------------------------------------
# 7. Base map
# --------------------------------------------------

def create_base_map(title, subtitle):

    fig, ax = plt.subplots(
        figsize=(18, 14),
        dpi=180,
    )

    fig.patch.set_facecolor(FIGURE_BG)
    ax.set_facecolor("#E8EAED")

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])

    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        title,
        color=TITLE_COLOR,
        fontsize=27,
        fontweight="bold",
        loc="left",
        pad=28,
    )

    ax.text(
        0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        color=TEXT_COLOR,
        fontsize=14,
        ha="left",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw lines before station markers.
    line_handles = draw_tube_lines(ax)

    fig.legend(
        handles=line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=6,
        frameon=(BACKGROUND_MODE == "white"),
        fontsize=11,
        labelcolor=TEXT_COLOR,
        handlelength=2.2,
        columnspacing=2.0,
        facecolor=LEGEND_BG,
        edgecolor=LEGEND_EDGE,
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.92,
        top=0.91,
        bottom=0.12,
    )

    return fig, ax


# --------------------------------------------------
# 8. Save helper
# --------------------------------------------------

def save_map(fig, filename):

    save_kwargs = {
        "dpi": 300,
        "bbox_inches": "tight",
        "facecolor": FIGURE_BG,
    }

    if BACKGROUND_MODE == "transparent":
        save_kwargs["transparent"] = True

    else:
        save_kwargs["transparent"] = False

    fig.savefig(
        OUTPUT_DIR / filename,
        **save_kwargs,
    )

    plt.close(fig)


# --------------------------------------------------
# 9. Map 1: 2019 baseline passenger flow
# --------------------------------------------------

fig, ax = create_base_map(
    "05  |  2019: Before the pandemic",
    "Passenger flow at each Underground station"
)

valid_flow = df["Flow_2019"].gt(0)

scatter = ax.scatter(
    df.loc[valid_flow, "X"],
    df.loc[valid_flow, "Y"],
    c=df.loc[valid_flow, "Flow_2019"],
    cmap="viridis",
    norm=LogNorm(
        vmin=df.loc[valid_flow, "Flow_2019"].min(),
        vmax=df.loc[valid_flow, "Flow_2019"].max(),
    ),
    s=65,
    alpha=0.95,
    edgecolors=STATION_EDGE,
    linewidths=0.65,
    zorder=3,
)

cbar = fig.colorbar(
    scatter,
    ax=ax,
    fraction=0.035,
    pad=0.025,
)

cbar.set_label(
    "Passenger flow (log scale)",
    color=TITLE_COLOR,
    fontsize=13,
)

cbar.ax.tick_params(
    colors=TICK_COLOR,
    labelsize=11,
)

cbar.outline.set_edgecolor("#BFC5CC")

save_map(fig, "map_2019.png")


# --------------------------------------------------
# 10. Shared function for recovery maps
# --------------------------------------------------

def create_recovery_map(year, recovery_column, filename):

    fig, ax = create_base_map(
        f"{'06' if year == 2022 else '07'} | {year}: Recovery relative to 2019",
        "Station-level passenger flow as a percentage of the 2019 baseline"
    )
    

    comparable = (
        df["Is_Comparable"]
        & df[recovery_column].notna()
        & np.isfinite(df[recovery_column])
    )

    incomparable = (
        ~df["Is_Comparable"]
        & df[recovery_column].notna()
        & np.isfinite(df[recovery_column])
    )

    # Identical colour scale for both years.
    norm = TwoSlopeNorm(
        vmin=0,
        vcenter=100,
        vmax=150,
    )

    scatter = ax.scatter(
        df.loc[comparable, "X"],
        df.loc[comparable, "Y"],
        c=df.loc[comparable, recovery_column],
        cmap="RdYlGn",
        norm=norm,
        s=65,
        alpha=0.97,
        edgecolors=STATION_EDGE,
        linewidths=0.65,
        zorder=3,
    )

    # Incomparable stations: hollow grey circles.
    ax.scatter(
        df.loc[incomparable, "X"],
        df.loc[incomparable, "Y"],
        facecolors="none",
        edgecolors=INCOMPARABLE_EDGE,
        s=75,
        linewidths=1.5,
        label="Not comparable",
        zorder=4,
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.035,
        pad=0.025,
        extend="max",
    )

    cbar.set_label(
        "Recovery relative to 2019 (%)",
        color=TITLE_COLOR,
        fontsize=13,
    )

    cbar.ax.tick_params(
        colors=TICK_COLOR,
        labelsize=11,
    )

    cbar.outline.set_edgecolor("#BFC5CC")

    station_legend = ax.legend(
        loc="lower left",
        facecolor=LEGEND_BG,
        edgecolor=LEGEND_EDGE,
        labelcolor=TEXT_COLOR,
        fontsize=11,
        framealpha=0.95,
    )

    station_legend.set_zorder(10)

    save_map(fig, filename)


# --------------------------------------------------
# 11. Map 2 and Map 3
# --------------------------------------------------

create_recovery_map(
    2022,
    "Recovery_2022",
    "map_2022.png",
)

create_recovery_map(
    2025,
    "Recovery_2025",
    "map_2025.png",
)


# --------------------------------------------------
# 12. Finished
# --------------------------------------------------

print("\nFinished generating maps:")

print("Background mode:", BACKGROUND_MODE)

for filename in [
    "map_2019.png",
    "map_2022.png",
    "map_2025.png",
]:
    path = OUTPUT_DIR / filename

    print(
        filename,
        "|",
        f"{path.stat().st_size / 1024:.1f} KB",
        "|",
        f"{path.stat().st_size / 1024 / 1024:.2f} MB",
    )

print("\nAll maps saved successfully.")
print("Output folder:", OUTPUT_DIR)
