# ==== Key map + 2×2 panels (compact, consistent with final style) ====

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box, LineString
from shapely.ops import linemerge
from pyproj import Transformer
import contextily as ctx
from matplotlib import patheffects as pe

# ----------------------- EASY KNOBS -----------------------
KEY_LON_STEP, KEY_LAT_STEP = 0.10, 0.10
AOI_LON_STEP, AOI_LAT_STEP = 0.01, 0.01
AOI_LABEL_STRIDE = 2
TICK_PAD = 6
FIG_W_CM, FIG_H_CM = 17.4, 10.8
# ----------------------------------------------------------
'''
# Robust PROJ
conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
proj_dir = os.path.join(conda_prefix, "share", "proj")
os.environ["PROJ_LIB"] = proj_dir
os.environ["PYPROJ_GLOBAL_CONTEXT"] = "ON"
'''
# ---------- INPUTS ----------
obs_path = "./inp_data/2019_flood_obs.tif" 
sim_lev  = "./inp_data/depth_2019_lev.tif" #simulated output for a day of event converted into tif
sim_nat  = "./inp_data/depth_2019_nat.tif" #simulated output for a day of even  converted into tif
keymap_raster   = sim_nat
rivers_centerline = "/data7/hasnain/Data/paper/shp-files/W05_river12.gpkg"
aoi_shapefile     = "./inp_data/shp-files/River_AOI.shp"
REVERSE_FLOW = False

# Extents (EPSG:4326)
KEY_W, KEY_E = 138.05, 138.40
KEY_S, KEY_N = 36.40, 36.91
A_W, A_E = 138.24, 138.30
A_S, A_N = 36.66, 36.72
B_W, B_E = 138.09, 138.17
B_S, B_N = 36.53, 36.58

# ---------- styles  ----------
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 600,
    "font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
})

to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def ext3857(W, E, S, N):
    x0, y0 = to_3857.transform(W, S); x1, y1 = to_3857.transform(E, N)
    return [x0, x1, y0, y1]

# --- map aids ---
def _choose_scalebar_km(W, E, S):
    x0,_ = to_3857.transform(W, S); x1,_ = to_3857.transform(E, S)
    width_km = (x1 - x0) / 1000.0
    for k in [5, 3, 2, 1]:
        if k <= width_km * 0.3:
            return k
    return max(1, round(width_km * 0.2))

def add_scalebar(ax, km=2, loc="lower right"):
    bar = AnchoredSizeBar(ax.transData, km*1000.0, f"{int(km)} km",
                          loc, pad=0.25, color="k", frameon=False,
                          size_vertical=(ax.get_ylim()[1]-ax.get_ylim()[0])*0.0065)
    ax.add_artist(bar)

def add_north_arrow(ax, x=0.06, y=0.90):
    ax.annotate("N", xy=(x, y), xytext=(x, y-0.10), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9),
                ha="center", va="center", fontsize=6)

# ---- lat/lon grid helpers (numbers only) ----
def frange_inside(start, stop, step):
    a = np.ceil(start / step) * step
    b = np.floor(stop  / step) * step
    if b < a - 1e-12:
        return np.array([])
    n = int(round((b - a) / step)) + 1
    return np.round(a + np.arange(n) * step, 10)

def add_graticule(ax, W, E, S, N,
                  lon_step, lat_step,
                  label_stride=1, tick_pad=6, xtick_rot=0,
                  color="0.5", lw=0.5, alpha=0.6):
    lons = frange_inside(W, E, lon_step)
    lats = frange_inside(S, N, lat_step)

    # grid lines (draw in 3857)
    for lon in lons:
        lat_line = np.linspace(S, N, 200)
        x, y = to_3857.transform(np.full_like(lat_line, lon), lat_line)
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=1)
    for lat in lats:
        lon_line = np.linspace(W, E, 200)
        x, y = to_3857.transform(lon_line, np.full_like(lon_line, lat))
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=1)

    # ticks/labels (plain decimals)
    mid_lat = 0.5 * (S + N)
    mid_lon = 0.5 * (W + E)
    label_lons = lons[::label_stride]
    label_lats = lats[::label_stride]
    xtick_pos = [to_3857.transform(lon, mid_lat)[0] for lon in label_lons]
    ytick_pos = [to_3857.transform(mid_lon, lat)[1] for lat in label_lats]

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"{lon:.2f}" for lon in label_lons],
                       rotation=xtick_rot, ha=("center" if xtick_rot == 0 else "right"))
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"{lat:.2f}" for lat in label_lats])

    ax.tick_params(direction="out", length=3, width=0.6, pad=tick_pad)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8); s.set_color("0.3")

    xmin, ymin = to_3857.transform(W, S)
    xmax, ymax = to_3857.transform(E, N)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# ---------- data-driven flow arrows ----------
def load_river_lines(path):
    try:
        return gpd.read_file(path).to_crs("EPSG:4326")
    except Exception as e:
        print("River load warning:", e); return None

def mainline_in_bounds(gdf_lines, W, E, S, N):
    if gdf_lines is None or gdf_lines.empty: return None
    bbox = box(W, S, E, N)
    clip = gpd.overlay(gdf_lines, gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326"), how="intersection")
    if clip.empty: return None
    clip_m = clip.to_crs("EPSG:3857")
    return clip_m.geometry.iloc[clip_m.length.values.argmax()]

def draw_flow_arrow(ax, line_3857, where=0.60, span=0.05, color="crimson", lw=1.5, head=9):
    if line_3857 is None: return
    line = linemerge(line_3857)
    if not isinstance(line, LineString): return
    L = line.length
    p0 = line.interpolate(max(0, (where-span/2))*L)
    p1 = line.interpolate(min(1, (where+span/2))*L)
    x0,y0 = p0.x, p0.y; x1,y1 = p1.x, p1.y
    if REVERSE_FLOW: x0,y0,x1,y1 = x1,y1,x0,y0
    ann = ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                      arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                      shrinkA=0, shrinkB=0, mutation_scale=head),
                      zorder=6)
    
    ann.set_path_effects([pe.Stroke(linewidth=lw+1.5, foreground="white"), pe.Normal()])

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


# ---------- optional AOI mask ----------
aoi = None
if aoi_shapefile and os.path.exists(aoi_shapefile):
    try:    aoi = gpd.read_file(aoi_shapefile).to_crs("EPSG:4326")
    except Exception as e: print("AOI load warning:", e)

rivers = load_river_lines(rivers_centerline)

# ---------- observed flood mask ----------
def read_obs_mask(bounds):
    W,E,S,N = bounds
    with rasterio.open(obs_path) as src:
        win = from_bounds(W,S,E,N, src.transform)
        arr = src.read(1, window=win)
        nod = src.nodata
        if nod is not None: arr = np.where(arr==nod, 0, arr)
    return (arr>0)

# ---------- confusion renderer ----------
c_tp, c_fp, c_fn = "#1b9e77", "#d95f02", "#7570b3"
cmap_conf = ListedColormap([(0,0,0,0), c_tp, c_fp, c_fn])
norm_conf = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], 4, clip=True)
legend_handles = [Patch(facecolor=c_tp, label="Hit (TP)"),
                  Patch(facecolor=c_fp, label="False alarm (FP)"),
                  Patch(facecolor=c_fn, label="Miss (FN)")]

def draw_confusion(ax, sim_path, bounds, title, show_left=False, show_bottom=False, show_north=False):
    W,E,S,N = bounds
    obs_flood = read_obs_mask(bounds)

    with rasterio.open(sim_path) as src:
        win = from_bounds(W,S,E,N, src.transform)
        sim = src.read(1, window=win)
        tx  = src.window_transform(win)
        nod = src.nodata
        if nod is not None: sim = np.where(sim==nod, 0, sim)
    sim_valid = np.ones_like(sim, bool) if nod is None else (sim != nod)

    if aoi is not None and len(aoi):
        aoi_mask = rasterize([(g,1) for g in aoi.geometry],
                             out_shape=sim.shape, transform=tx,
                             fill=0, all_touched=True, dtype=np.uint8)
    else:
        aoi_mask = np.zeros(sim.shape, np.uint8)

    sim_flood = (sim>0) & sim_valid & (aoi_mask==0)

    conf = np.zeros(sim.shape, np.uint8)
    conf[ obs_flood &  sim_flood] = 1
    conf[~obs_flood &  sim_flood] = 2
    conf[ obs_flood & ~sim_flood] = 3

    TP = np.sum(conf==1); FP=np.sum(conf==2); FN=np.sum(conf==3); TN=np.sum(conf==0)
    IoU = TP/(TP+FP+FN) if (TP+FP+FN) else np.nan
    ACC = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) else np.nan
    FBI = (TP+FP)/(TP+FN) if (TP+FN) else np.nan

    ax.imshow(conf, extent=ext3857(W,E,S,N), cmap=cmap_conf, norm=norm_conf,
              interpolation="nearest", origin="upper", zorder=2)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron,
                        zoom=14, zorder=0, attribution_size=5, alpha=0.9)
    except Exception as e:
        print("Basemap warning:", e)

    add_graticule(ax, W,E,S,N,
                  lon_step=AOI_LON_STEP, lat_step=AOI_LAT_STEP,
                  label_stride=AOI_LABEL_STRIDE, tick_pad=TICK_PAD, xtick_rot=0)

    ax.set_title(title, pad=2, loc="left")
    ax.text(0.02, 0.98, f"IoU {IoU:.2f}\nACC {ACC:.2f}\nFBI {FBI:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.95),
            fontsize=6, zorder=4)

    line = mainline_in_bounds(rivers, W,E,S,N)
    draw_flow_arrow(ax, line, where=0.60, span=0.06)

# ---------- FIGURE LAYOUT ----------
fig = plt.figure(figsize=(FIG_W_CM/2.54, FIG_H_CM/2.54), constrained_layout=True)
gs  = fig.add_gridspec(2, 3, width_ratios=[1.10, 1, 1], wspace=0.045, hspace=0.05)

# (a) KEY MAP
ax_key = fig.add_subplot(gs[:,0])
with rasterio.open(keymap_raster) as src:
    win = from_bounds(KEY_W, KEY_S, KEY_E, KEY_N, src.transform)
    depth = src.read(1, window=win).astype(float)
    nod = src.nodata
    if nod is not None: depth[depth==nod] = np.nan
ax_key.imshow(depth, extent=ext3857(KEY_W,KEY_E,KEY_S,KEY_N),
              cmap="Blues",
              vmin=np.nanpercentile(depth,5), vmax=np.nanpercentile(depth,95),
              origin="upper", alpha=0.65, zorder=1) 
try:
    ctx.add_basemap(ax_key, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron,
                    zoom=11, zorder=0, attribution_size=5, alpha=0.9)
except Exception as e:
    print("Basemap warning:", e)

add_graticule(ax_key, KEY_W,KEY_E,KEY_S,KEY_N,
              lon_step=KEY_LON_STEP, lat_step=KEY_LAT_STEP,
              label_stride=1, tick_pad=TICK_PAD, xtick_rot=0)

ax_key.set_title("Key Map", loc="center")
add_scalebar(ax_key, km=_choose_scalebar_km(KEY_W,KEY_E,KEY_S))
add_north_arrow(ax_key, x=0.08, y=0.92)

# --- manual arrows on key map (edit coordinates) ---
add_arrow_keymap(ax_key, 138.21, 36.585, 138.250, 36.605)  # lower arrow
add_arrow_keymap(ax_key, 138.33, 36.79, 138.345, 36.825)  # upper arrow

def draw_box(ax, W,E,S,N, label):
    x0,y0 = to_3857.transform(W,S); x1,y1 = to_3857.transform(E,N)
    ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, ec="k", lw=1.0, zorder=5))
    ax.text((x0+x1)/2, y1, label, ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=6)

draw_box(ax_key, A_W,A_E,A_S,A_N, "a")
draw_box(ax_key, B_W,B_E,B_S,B_N, "b")
for (W,E,S,N) in [(A_W,A_E,A_S,A_N), (B_W,B_E,B_S,B_N)]:
    line = mainline_in_bounds(rivers, W,E,S,N)
    draw_flow_arrow(ax_key, line, where=0.60, span=0.06)

# (b)–(e) AOI panels
ax_b = fig.add_subplot(gs[0,1]); draw_confusion(ax_b, sim_lev, (A_W,A_E,A_S,A_N), "2019_lev [a]", show_left=True,  show_bottom=False, show_north=False)
ax_c = fig.add_subplot(gs[0,2]); draw_confusion(ax_c, sim_nat, (A_W,A_E,A_S,A_N), "2019_nat [a]", show_left=False, show_bottom=False, show_north=False)
ax_d = fig.add_subplot(gs[1,1]); draw_confusion(ax_d, sim_lev, (B_W,B_E,B_S,B_N), "2019_lev [b]", show_left=True,  show_bottom=True,  show_north=False)
ax_e = fig.add_subplot(gs[1,2]); draw_confusion(ax_e, sim_nat, (B_W,B_E,B_S,B_N), "2019_nat [b]", show_left=False, show_bottom=True,  show_north=False)

# shared legend (space above)
fig.legend(handles=[Patch(facecolor="#1b9e77", label="Perfect (TP)"),
                    Patch(facecolor="#d95f02", label="Over estimation (FP)"),
                    Patch(facecolor="#7570b3", label="Under estimation (FN)")],
           ncol=3, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, -0.06), handlelength=1.0,
           columnspacing=1.0, labelspacing=0.3)

# save
out = "./"
os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, "figure_5.png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
#plt.show()

