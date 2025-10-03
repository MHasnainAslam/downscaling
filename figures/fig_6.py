# ==== Two-panel confusion maps with basemap + LAT/LON GRID (GMD-friendly) ====

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from pyproj import Transformer
import contextily as ctx

# ---------- INPUTS ----------
obs_path = "./inp_data/Chick_Rs_ds_digitized.tif"
sim_paths = {
    "lev": "./inp_data/depths_1000_lev.tif",
    "nat": "./inp_data/depths_1000_nat.tif"
}
aoi_shapefile = "./inp_data/shp-files/River_AOI.shp"  # polygons to EXCLUDE (mask==1); set None to skip

# Lon/lat bbox (EPSG:4326)
WEST, EAST = 138.24, 138.30
SOUTH, NORTH = 36.66, 36.72

# ---------- GRATICULE SETTINGS (tweak these) ----------
GRATICULE_LON_STEP = 0.02   # increase for fewer x labels (e.g., 0.06–0.08)
GRATICULE_LAT_STEP = 0.02   # increase for fewer y labels (e.g., 0.03)
GRATICULE_LABEL_STRIDE = 1  # 2 = label every other grid line
GRATICULE_TICK_PAD = 2     # pixels between ticks and labels (more = bigger gap)

# ---------- style ----------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
})

def add_scalebar(ax, length_km=2.0, loc="lower right"):
    length_m = length_km * 1000.0
    bar = AnchoredSizeBar(ax.transData, length_m, f"{int(length_km)} km",
                          loc, pad=0.3, color="k", frameon=False,
                          size_vertical=(ax.get_ylim()[1]-ax.get_ylim()[0])*0.008)
    ax.add_artist(bar)

def add_north_arrow(ax, x=0.06, y=0.88):
    ax.annotate("N", xy=(x, y), xytext=(x, y-0.10),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0),
                ha="center", va="center", fontsize=9)

# ---------- helper: lat/lon graticule over EPSG:3857 axes ----------
def nice_step(span_deg):
    if span_deg > 5:   return 1.0
    if span_deg > 2:   return 0.5
    if span_deg > 1:   return 0.2
    if span_deg > 0.5: return 0.1
    if span_deg > 0.2: return 0.05
    return 0.02

def frange(start, stop, step):
    """Inclusive grid *within* [start, stop] at multiples of `step`."""
    a = np.ceil(start / step) * step          # first line >= start
    b = np.floor(stop  / step) * step         # last  line <= stop
    if b < a - 1e-12:
        return np.array([])                   # nothing in range
    n = int(round((b - a) / step)) + 1
    return np.round(a + np.arange(n) * step, 10)

# 2-decimal coord labels
def format_lon(lon):
    hemi = "E" if lon >= 0 else "W"
    return f"{abs(lon):.2f}°{hemi}"

def format_lat(lat):
    hemi = "N" if lat >= 0 else "S"
    return f"{abs(lat):.2f}°{hemi}"

def add_graticule(ax, west, east, south, north,
                  crs_from="EPSG:4326", crs_to="EPSG:3857",
                  color="0.5", lw=0.5, alpha=0.6,
                  lon_step=None, lat_step=None,
                  label_stride=1, tick_pad=4, xtick_rot=0):
    """Draw lon/lat grid and set tick labels in degrees with spacing controls."""
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

    if lon_step is None: lon_step = nice_step(east - west)
    if lat_step is None: lat_step = nice_step(north - south)

    lons = frange(west,  east,  lon_step)
    lats = frange(south, north, lat_step)

    # grid lines
    for lon in lons:
        lat_line = np.linspace(south, north, 120)
        x, y = transformer.transform(np.full_like(lat_line, lon), lat_line)
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=1)
    for lat in lats:
        lon_line = np.linspace(west, east, 120)
        x, y = transformer.transform(lon_line, np.full_like(lon_line, lat))
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=1)

    # ticks/labels (sparse via label_stride)
    mid_lat = 0.5 * (south + north)
    mid_lon = 0.5 * (west + east)
    label_lons = lons[::label_stride]
    label_lats = lats[::label_stride]

    xtick_pos = [transformer.transform(lon, mid_lat)[0] for lon in label_lons]
    ytick_pos = [transformer.transform(mid_lon, lat)[1] for lat in label_lats]

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([format_lon(lon) for lon in label_lons],
                       rotation=xtick_rot, ha=("center" if xtick_rot == 0 else "right"))
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([format_lat(lat) for lat in label_lats])

    # spacing between ticks and labels
    ax.tick_params(length=3, pad=tick_pad)

    # show frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.3")

# ---------- read AOI ----------
aoi = None
if aoi_shapefile and os.path.exists(aoi_shapefile):
    try:
        aoi = gpd.read_file(aoi_shapefile).to_crs("EPSG:4326")
    except Exception as e:
        print("AOI load warning:", e)
        aoi = None

# ---------- observed flood mask ----------
with rasterio.open(obs_path) as obs_src:
    obs_window = from_bounds(WEST, SOUTH, EAST, NORTH, obs_src.transform)
    obs = obs_src.read(1, window=obs_window)
    obs_transform = obs_src.window_transform(obs_window)
    obs_nodata = obs_src.nodata
    if obs_nodata is not None:
        obs = np.where(obs == obs_nodata, 0, obs)
obs_flood = (obs > 0)

# ---------- 3857 extent for plotting/basemap ----------
to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = to_3857.transform(WEST, SOUTH)
xmax, ymax = to_3857.transform(EAST, NORTH)
extent_3857 = [xmin, xmax, ymin, ymax]

# ---------- colors (TN transparent) ----------
c_tp = "#1b9e77"; c_fp = "#d95f02"; c_fn = "#7570b3"
cmap = ListedColormap([(0,0,0,0), c_tp, c_fp, c_fn])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4, clip=True)
legend_handles = [Patch(facecolor=c_tp, label="Perfect (TP)"),
                  Patch(facecolor=c_fp, label="Over estimation (FP)"),
                  Patch(facecolor=c_fn, label="Under estimation (FN)")]

# ---------- figure ----------
fig_w_cm, fig_h_cm = 17.4, 9.2
fig = plt.figure(figsize=(fig_w_cm/2.54, fig_h_cm/2.54), constrained_layout=True)
gs = fig.add_gridspec(1, 2, wspace=0.06)
panel_labels = ["lev [a]", "nat [a]"]

for i, (label, sim_path) in enumerate(sim_paths.items()):
    ax = fig.add_subplot(gs[0, i])

    with rasterio.open(sim_path) as sim_src:
        sim_window = from_bounds(WEST, SOUTH, EAST, NORTH, sim_src.transform)
        sim = sim_src.read(1, window=sim_window)
        sim_transform = sim_src.window_transform(sim_window)
        sim_nodata = sim_src.nodata
        if sim_nodata is not None:
            sim = np.where(sim == sim_nodata, 0, sim)

    sim_valid = (sim != sim_nodata) if sim_nodata is not None else np.ones_like(sim, bool)

    if aoi is not None and len(aoi):
        aoi_mask = rasterize(
            [(geom, 1) for geom in aoi.geometry],
            out_shape=sim.shape,
            transform=sim_transform,
            fill=0, all_touched=True, dtype=np.uint8
        )
    else:
        aoi_mask = np.zeros(sim.shape, dtype=np.uint8)

    sim_flood = (sim > 0) & sim_valid & (aoi_mask == 0)

    # confusion: 0 TN, 1 TP, 2 FP, 3 FN
    conf = np.zeros(sim.shape, dtype=np.uint8)
    conf[obs_flood & sim_flood] = 1
    conf[(~obs_flood) & sim_flood] = 2
    conf[obs_flood & (~sim_flood)] = 3

    # metrics
    TP = np.sum(conf == 1); FP = np.sum(conf == 2); FN = np.sum(conf == 3); TN = np.sum(conf == 0)
    IoU = TP / (TP + FP + FN) if (TP + FP + FN) else np.nan
    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else np.nan
    FBI = (TP + FP) / (TP + FN) if (TP + FN) else np.nan

    # draw confusion layer (TN transparent) on basemap extent
    ax.imshow(conf, extent=extent_3857, cmap=cmap, norm=norm,
              interpolation="nearest", origin="upper", zorder=2)

    # basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron,
                        zoom=14, zorder=0, attribution_size=6)
    except Exception as e:
        print("Basemap warning:", e)

    # add graticule: 2-dec labels, horizontal lon, coarser spacing, bigger gap
    add_graticule(
        ax, WEST, EAST, SOUTH, NORTH,
        lon_step=GRATICULE_LON_STEP,
        lat_step=GRATICULE_LAT_STEP,
        label_stride=GRATICULE_LABEL_STRIDE,
        tick_pad=GRATICULE_TICK_PAD,
        xtick_rot=0  # horizontal longitudes
    )

    ax.set_title(panel_labels[i], pad=2)

    # stats box
    ax.text(0.02, 0.98, f"IoU {IoU:.2f}\nACC {ACC:.2f}\nFBI {FBI:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9), zorder=4)

# save
out_dir = "./"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "figure_6.png"), dpi=600)


