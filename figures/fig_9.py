# ==== Generate Key map + 2×2 hazard panels (two regions, RP=10 & 100) ====
# - Uses 1-sec hazard classes (A–F) with area summaries
# - Clean layout, compact legend, basemap, scalebar, north arrow

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import contextily as cx
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from matplotlib import patheffects as pe

# ----------------------- CONFIG -----------------------
# Hi-res Depth binaries 
FOLDER_DEPTH = "../scripts"    # where depths_{rp}_lev.bin placed
RPS = [10, 100]                                          # panels per region

# Key map backdrop: any GeoTIFF in EPSG:4326 (e.g., simulated depth)
KEYMAP_RASTER = "./inp_data/depth_2019_nat.tif"

# Two AOIs (EPSG:4326) — adjust if needed
# Region [a]
A_W, A_E = 138.24, 138.30
A_S, A_N = 36.66, 36.73

# Region [b]
B_W, B_E = 138.10, 138.14
B_S, B_N = 36.53, 36.56

# Full-chunk bounds for the binary arrays (1-sec cutout)
WEST_FULL, EAST_FULL = 138.0, 139.0
SOUTH_FULL, NORTH_FULL = 36.0, 37.0
NX, NY = 3600, 3600
DX = 1.0 / 3600.0

# Key map extent
KEY_W, KEY_E = 138.05, 138.40
KEY_S, KEY_N = 36.40, 36.91

# Output
OUT_DIR = "./"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "figure_9.png")

# Figure style
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 600,
    "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
})

# Hazard classes (A–F)
hazard_bins = [0.00, 0.5, 3.0, 5.0, 10.0, 20.0, np.inf]
hazard_ids  = ['A', 'B', 'C', 'D', 'E', 'F']
hazard_text = ['0.00–0.5 m', '0.5–3.0 m', '3.0–5.0 m', '5.0–10.0 m', '10.0–20.0 m', '>20.0 m']
hazard_labels = [f"{hazard_ids[i]}: {hazard_text[i]}" for i in range(len(hazard_ids))]
hazard_colors = ['#ffff99', '#fdbf6f', '#fb8072', '#e34a33', '#d53e4f', '#9e0142']
cmap = ListedColormap(hazard_colors)
norm = BoundaryNorm(range(1, len(hazard_bins)), cmap.N)

# ----------------------- HELPERS -----------------------
to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def ext3857(W, E, S, N):
    x0, y0 = to_3857.transform(W, S); x1, y1 = to_3857.transform(E, N)
    return [x0, x1, y0, y1]

def draw_box(ax, W,E,S,N, label):
    x0,y0 = to_3857.transform(W,S); x1,y1 = to_3857.transform(E,N)
    ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, ec="k", lw=1.0, zorder=5))
    ax.text((x0+x1)/2, y1, label, ha="center", va="bottom",
            fontsize=9, fontweight="bold", zorder=6)

def add_scalebar(ax, km=2, loc="lower right"):
    # In lon/lat axes: approximate 1 deg lon ~ 111 km * cos(lat). We set a small visual bar.
    bar = AnchoredSizeBar(ax.transAxes, 0.18, f"{km} km", loc,
                          pad=0.25, color="k", frameon=False, size_vertical=0.01)
    ax.add_artist(bar)

def add_north_arrow(ax, x=0.08, y=0.92):
    ax.annotate("N", xy=(x, y), xytext=(x, y-0.10), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0),
                ha="center", va="center", fontsize=6)

def format_coord(x, pos): return f"{x:.2f}"

def compute_grid_area(nx, ny, lat_origin):
    R = 6371000.0
    pixel_size = 1.0/3600.0
    lat_array = np.linspace(lat_origin + (ny/2)*pixel_size,
                            lat_origin - (ny/2)*pixel_size, ny)
    pixel_width  = (R*np.radians(pixel_size))*np.cos(np.radians(lat_array))[:,None]
    pixel_height = R*np.radians(pixel_size)
    return np.repeat(pixel_width, nx, axis=1) * pixel_height  # m²

def crop_indices(W, E, S, N):
    ix0 = int((W - WEST_FULL)  / DX)
    ix1 = int((E - WEST_FULL)  / DX)
    iy0 = int((NORTH_FULL - N) / DX)
    iy1 = int((NORTH_FULL - S) / DX)
    return ix0, ix1, iy0, iy1

def load_depth_crop(rp, W,E,S,N):
    path = os.path.join(FOLDER_DEPTH, f"depths_{rp}_lev.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.fromfile(path, dtype=np.float32).reshape((NY, NX))
    ix0, ix1, iy0, iy1 = crop_indices(W,E,S,N)
    sub = arr[iy0:iy1, ix0:ix1]
    sub = np.where(sub < 0.05, np.nan, sub)  # ignore tiny ripples
    return sub

def make_hazard_and_areas(depth_crop, W, E, S, N):
    # use the cropped array's shape to avoid off-by-one issues
    ny_sel, nx_sel = depth_crop.shape
    grid_area = compute_grid_area(nx_sel, ny_sel, S)  # m²

    hmap = np.full_like(depth_crop, np.nan)
    areas = []
    for i in range(len(hazard_bins)-1):
        mask = (depth_crop >= hazard_bins[i]) & (depth_crop < hazard_bins[i+1])
        hmap[mask] = i + 1
        areas.append(np.nansum(grid_area[mask]) / 1e6)  # km²
    return hmap, areas

def add_arrow_keymap(ax, lon0, lat0, lon1, lat1,
                     color="crimson", lw=1.5, head=10, zorder=6):
    """Draw an arrow on the KEY MAP using lon/lat, transforming to EPSG:3857."""
    x0, y0 = to_3857.transform(lon0, lat0)
    x1, y1 = to_3857.transform(lon1, lat1)
    ann = ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=head),
        zorder=zorder
    )
    # white halo so it stays visible on the basemap
    ann.set_path_effects([pe.Stroke(linewidth=lw+1.5, foreground="white"),
                          pe.Normal()])
    return ann

def add_arrow_panel(ax, lon0, lat0, lon1, lat1,
                    color="crimson", lw=2.0, head=12, zorder=6):
    """Arrow on REGION PANELS (they already use lon/lat axes)."""
    ann = ax.annotate(
        "", xy=(lon1, lat1), xytext=(lon0, lat0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=head),
        zorder=zorder
    )
    ann.set_path_effects([pe.Stroke(linewidth=lw+1.5, foreground="white"),
                          pe.Normal()])
    return ann

    
# ----------------------- FIGURE -----------------------
fig = plt.figure(figsize=(17.4/2.54, 10.8/2.54), constrained_layout=True)
gs  = fig.add_gridspec(2, 3, width_ratios=[1.10, 1, 1], wspace=0.045, hspace=0.06)

# (a) Key map
ax_key = fig.add_subplot(gs[:,0])
with rasterio.open(KEYMAP_RASTER) as src:
    win = from_bounds(KEY_W, KEY_S, KEY_E, KEY_N, src.transform)
    key = src.read(1, window=win).astype(float)
    nod = src.nodata
    if nod is not None: key[key==nod] = np.nan
ax_key.imshow(key, extent=ext3857(KEY_W,KEY_E,KEY_S,KEY_N),
              cmap="Blues", vmin=np.nanpercentile(key,5),
              vmax=np.nanpercentile(key,95), origin="upper", alpha=0.65, zorder=1)
try:
    cx.add_basemap(ax_key, crs="EPSG:3857",
                   source=cx.providers.CartoDB.Positron,
                   zoom=11, zorder=0, attribution_size=5, alpha=0.9)
except Exception as e:
    print("Basemap warning:", e)

# --- manual arrows on key map (edit coordinates) ---
add_arrow_keymap(ax_key, 138.21, 36.585, 138.250, 36.605)  # lower arrow
add_arrow_keymap(ax_key, 138.33, 36.79, 138.345, 36.825)  # upper arrow

# keymap grid/ticks (simple lat/lon decoration)
def add_graticule(ax, W, E, S, N, lon_step=0.1, lat_step=0.1):
    # target lon/lat lines to show on the key map
    lons = np.arange(np.ceil(W/lon_step)*lon_step,
                     np.floor(E/lon_step)*lon_step + 1e-9, lon_step)
    lats = np.arange(np.ceil(S/lat_step)*lat_step,
                     np.floor(N/lat_step)*lat_step + 1e-9, lat_step)

    # draw grid lines in 3857 space, but remember tick positions in 3857
    xticks_3857 = []
    for lon in lons:
        x, _ = to_3857.transform(lon, 0.0)              # x depends only on lon in Mercator
        y0 = to_3857.transform(lon, S)[1]
        y1 = to_3857.transform(lon, N)[1]
        ax.plot([x, x], [y0, y1], color="0.75", lw=0.5, alpha=0.7, zorder=0)
        xticks_3857.append(x)

    yticks_3857 = []
    for lat in lats:
        y = to_3857.transform(0.0, lat)[1]              # y depends only on lat
        x0 = to_3857.transform(W, lat)[0]
        x1 = to_3857.transform(E, lat)[0]
        ax.plot([x0, x1], [y, y], color="0.75", lw=0.5, alpha=0.7, zorder=0)
        yticks_3857.append(y)

    # set bounds and label ticks in degrees (to match region panels)
    x0, y0 = to_3857.transform(W, S)
    x1, y1 = to_3857.transform(E, N)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

    ax.set_xticks(xticks_3857); ax.set_xticklabels([f"{lon:.2f}" for lon in lons])
    ax.set_yticks(yticks_3857); ax.set_yticklabels([f"{lat:.2f}" for lat in lats])


add_graticule(ax_key, KEY_W,KEY_E,KEY_S,KEY_N, lon_step=0.10, lat_step=0.10)
ax_key.set_title("Key Map", loc="center")
draw_box(ax_key, A_W,A_E,A_S,A_N, "a")
draw_box(ax_key, B_W,B_E,B_S,B_N, "b")
add_scalebar(ax_key, km=5); add_north_arrow(ax_key, x=0.08, y=0.92)

# Panels helper
def plot_panel(ax, rp, W,E,S,N, title):
    depth_crop = load_depth_crop(rp, W,E,S,N)
    hmap, areas = make_hazard_and_areas(depth_crop, W,E,S,N)

    im = ax.imshow(hmap, extent=[W,E,S,N], origin="upper",
                   cmap=cmap, norm=norm, alpha=0.65, zorder=1)
    try:
        cx.add_basemap(ax, crs="EPSG:4326",
                       source=cx.providers.CartoDB.Positron,
                       zoom=15, zorder=0, attribution_size=5)
    except Exception as e:
        print("Basemap error:", e)
    ax.set_title(title, fontsize=7)
    ax.set_xlim(W,E); ax.set_ylim(S,N)
    ax.xaxis.set_major_formatter(FuncFormatter(format_coord))
    ax.yaxis.set_major_formatter(FuncFormatter(format_coord))
    ax.set_xticks(np.arange(W, E+0.0001, 0.02))
    ax.set_yticks(np.arange(S, N+0.0001, 0.01))
    ax.tick_params(labelsize=6)
    # Area summary
    txt = "\n".join([f"{hazard_ids[i]}: {areas[i]:.2f} km²" for i in range(len(hazard_ids))])
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=6,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))

# (b)(c) Region [a] — RP 10 / 100
ax_b = fig.add_subplot(gs[0,1]); plot_panel(ax_b, RPS[0], A_W,A_E,A_S,A_N, f"RP = {RPS[0]} years")
ax_c = fig.add_subplot(gs[0,2]); plot_panel(ax_c, RPS[1], A_W,A_E,A_S,A_N, f"RP = {RPS[1]} years")

# (d)(e) Region [b] — RP 10 / 100
ax_d = fig.add_subplot(gs[1,1]); plot_panel(ax_d, RPS[0], B_W,B_E,B_S,B_N, f"RP = {RPS[0]} years")
ax_e = fig.add_subplot(gs[1,2]); plot_panel(ax_e, RPS[1], B_W,B_E,B_S,B_N, f"RP = {RPS[1]} years")

# Shared legend (bottom center)
legend_handles = [Patch(facecolor=hazard_colors[i], label=hazard_labels[i])
                  for i in range(len(hazard_labels))]
fig.legend(handles=legend_handles, ncol=3, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, -0.06), handlelength=1.0,
           columnspacing=1.0, labelspacing=0.3)

plt.savefig(OUT_FILE, dpi=600, bbox_inches="tight", pad_inches=0.02)
#plt.show()
print("Saved:", OUT_FILE)

