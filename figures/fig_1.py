import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.ticker import FuncFormatter
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import pandas as pd
from cmcrameri import cm

# --- Font size settings ---
'''
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20
})

'''
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Region of interest in degrees
WEST, EAST = 138.32, 138.42
SOUTH, NORTH = 36.79, 36.89

# File paths
levfrc_tif = "./inp_data/lev_frc_tej_1min.tif"
levhgt_tif = "./inp_data/levhgt2.tif"
levee_data = "./inp_data/shp-files/n35e135_levee_data.shp"
catmxy_path = './inp_data/catmxy_1sec_Chik.tif'

# Load raster windows within bounding box
with rasterio.open(levfrc_tif) as src1:
    window1 = rasterio.windows.from_bounds(WEST, SOUTH, EAST, NORTH, src1.transform)
    levfrc = src1.read(1, window=window1, masked=True)
    levfrc = np.ma.masked_equal(levfrc, 0)
    extent1 = [WEST, EAST, SOUTH, NORTH]
    crs1 = src1.crs

with rasterio.open(levhgt_tif) as src2:
    window2 = rasterio.windows.from_bounds(WEST, SOUTH, EAST, NORTH, src2.transform)
    levhgt = src2.read(1, window=window2, masked=True)
    levhgt = np.ma.masked_equal(levhgt, -9999)
    extent2 = [WEST, EAST, SOUTH, NORTH]

# Load levee shapefile and crop
levees = gpd.read_file(levee_data).to_crs(crs1)
levees_crop = levees.cx[WEST:EAST, SOUTH:NORTH]

# Load unit catchment raster and extract polygons
with rasterio.open(catmxy_path) as src:
    window = rasterio.windows.from_bounds(WEST, SOUTH, EAST, NORTH, src.transform)
    cat_ids = src.read(1, window=window).astype(np.float32)
    cat_ids = np.round(cat_ids, 4)  # <-- round to 4 decimal places
    transform = src.window_transform(window)
    masked_ids = np.ma.masked_less(cat_ids, 0).filled(-1)

    # 2. Flatten and factor unique values to assign them unique integer labels
    flat_vals = cat_ids.flatten()
    unique_vals, inverse = np.unique(flat_vals, return_inverse=True)
    encoded_ids = inverse.reshape(cat_ids.shape)

    # 3. Build label lookup: index -> rounded float value
    label_map = dict(enumerate(unique_vals))

    # 4. Mask invalids
    masked_ids = np.ma.masked_less(cat_ids, 0).filled(-1)
    encoded_ids = np.where(masked_ids > 0, encoded_ids, -1)

    # 5. Extract shapes
    shape_gen = shapes(encoded_ids.astype(np.int32), mask=(encoded_ids != -1), transform=transform)

    # 6. Map back to rounded float values
    polys, labels = zip(*[(shape(geom), label_map[val]) for geom, val in shape_gen])

    gdf_catchments = gpd.GeoDataFrame(
        {"cat_id": labels},
        geometry=gpd.GeoSeries(polys, index=pd.RangeIndex(len(polys))),
        crs="EPSG:4326"
    )
    gdf_catchments = gdf_catchments.cx[WEST:EAST, SOUTH:NORTH]

# Coordinate formatter
def format_coord(x, pos):
    return f"{x:.2f}"

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Levee Fraction
im1 = axes[0].imshow(levfrc, extent=extent1, origin="upper", cmap=cm.batlow, zorder=1)
axes[0].set_title("Levee Fraction")
cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Panel 2: Levee Height
im2 = axes[1].imshow(levhgt, extent=extent2, origin="upper", cmap=cm.batlow, zorder=1)
axes[1].set_title("Levee Height [m]")
cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# Common formatting
for ax in axes:
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)
    ctx.add_basemap(ax, crs=crs1.to_string(), source=ctx.providers.CartoDB.Positron, alpha=1, zorder=0)
    ax.xaxis.set_major_formatter(FuncFormatter(format_coord))
    ax.yaxis.set_major_formatter(FuncFormatter(format_coord))
    ax.set_xticks(np.arange(WEST, EAST + 0.01, 0.05))
    ax.set_yticks(np.arange(SOUTH, NORTH + 0.01, 0.05))
    levees_crop.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.8, alpha=0.5, zorder=2)
    gdf_catchments.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=0.4, alpha=0.2, zorder=3)

plt.tight_layout()
plt.savefig("figure_1.png", dpi=300)





