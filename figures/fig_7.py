import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import matplotlib.ticker as mticker
import contextily as cx
import geopandas as gpd
from shapely.geometry import box
import os
from cmcrameri import cm

# ------------------ Configuration ------------------
catmxy_path = './inp_data/catmxy_1sec_Chik.tif'
depth_nat_path = './inp_data/depths_1sec_nolev.bin'
depth_lev_path = './inp_data/depths_1sec_lev.bin'
levee_data = './inp_data/shp-files/n35e135_levee_data.shp'
shapefile_folder = './inp_data/shp-files'
shapefile_names = ['cat554-919.shp', 'cat656-910.shp']
target_ids = [910.0565, 919.0554]

SHOW_LEVEE_ON_NAT = False
SHOW_LEVEE_ON_LEV = True

# ------------------ Helper Function ------------------
def compute_grid_area(nx, ny, lat_origin):
    R = 6371000
    pixel_size = 1.0 / 3600
    lat_array = np.linspace(
        lat_origin + (ny / 2) * pixel_size,
        lat_origin - (ny / 2) * pixel_size, ny
    )
    pixel_width = (R * np.radians(pixel_size)) * np.cos(np.radians(lat_array))[:, np.newaxis]
    pixel_height = R * np.radians(pixel_size)
    return np.repeat(pixel_width, nx, axis=1) * pixel_height

# ------------------ Load Raster ------------------
with rasterio.open(catmxy_path) as src:
    catmxy = src.read(1).astype(np.float32)
    transform = src.transform
    bounds = src.bounds
    crs = src.crs
    ny, nx = catmxy.shape
    south = bounds.bottom

# Load depth data
depth_nat = np.fromfile(depth_nat_path, dtype=np.float32).reshape(3600, 3600)[:ny, :nx]
depth_lev = np.fromfile(depth_lev_path, dtype=np.float32).reshape(3600, 3600)[:ny, :nx]
depth_nat = np.where(depth_nat < 0.05, np.nan, depth_nat)
depth_lev = np.where(depth_lev < 0.05, np.nan, depth_lev)

# ------------------ Load and Crop Levee Data ------------------
levee_vector = gpd.read_file(levee_data)
levee_vector = levee_vector.to_crs("EPSG:4326")

# Compute area
grid_area = compute_grid_area(nx, ny, south)

# Load shapefiles
shapefiles = [gpd.read_file(os.path.join(shapefile_folder, f)).to_crs("EPSG:4326") for f in shapefile_names]

# ------------------ Prepare Plot ------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()
PANEL_LABELS = ['(a) Unit Catchment A', '(b) Unit Catchment A', '(c) Unit Catchment B', '(d) Unit Catchment B']

plot_idx = 0
results = []
all_mask = (catmxy > 0)

# ------------------ Loop through Target IDs ------------------
for target_id in target_ids:
    mask = (catmxy == target_id)
    y_idx, x_idx = np.where(mask)
    if len(x_idx) == 0:
        continue

    x0 = transform * (x_idx.min(), 0)
    x1 = transform * (x_idx.max(), 0)
    y0 = transform * (0, y_idx.max())
    y1 = transform * (0, y_idx.min())
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    masked_nat = np.where(mask, depth_nat, np.nan)
    masked_lev = np.where(mask, depth_lev, np.nan)

    area_nat = np.nansum(grid_area[~np.isnan(masked_nat)]) / 1e6
    area_lev = np.nansum(grid_area[~np.isnan(masked_lev)]) / 1e6
    results.append({
        "catchment_id": target_id,
        "area_nat_km2": area_nat,
        "area_lev_km2": area_lev,
        "reduction_%": 100 * (area_nat - area_lev) / area_nat if area_nat else np.nan
    })

    for condition, data, cmap, area_val in zip(
        ["Natural", "Levee"],
        [masked_nat, masked_lev],
        ["Blues", "Blues"],
        [area_nat, area_lev]
    ):
        ax = axes[plot_idx]
        ax.set_title(f"{PANEL_LABELS[plot_idx]} — {condition}" ,fontsize=22, pad=8)
        ax.set_xlim(x0[0], x1[0])
        ax.set_ylim(y0[1], y1[1])

        cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron)
        im = ax.imshow(data, extent=extent, origin='upper', cmap=cm.batlow, vmin=0, vmax=5, alpha=0.5)

        ax.contour(all_mask, levels=[0.5], colors='gray', linewidths=0.6, extent=extent)
        ax.contour(mask, levels=[0.5], colors='black', linewidths=1.3, extent=extent)

        # Overlay levee pixels where value == 1
        if (condition == "Natural" and SHOW_LEVEE_ON_NAT) or (condition == "Levee" and SHOW_LEVEE_ON_LEV):
            levee_vector.plot(ax=ax, edgecolor='black', linewidth=0.8, facecolor='none', zorder=10)

        # Overlay shapefiles
        for gdf in shapefiles:
            gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, zorder=5)

        # Labels
        ax.set_xlabel("Longitude", fontsize=22)
        ax.set_ylabel("Latitude", fontsize=22)
        ax.tick_params(labelsize=20)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        ax.text(0.06, 0.94, f'{condition[:3]}: {area_val:.2f} km²',
                transform=ax.transAxes,
                fontsize=20,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.85, pad=0.02)
        cbar.set_label('Flood Depth (m)', fontsize=22)
        cbar.ax.tick_params(labelsize=20)

        plot_idx += 1

# ------------------ Finalize ------------------
plt.tight_layout()
plt.savefig('figure_7.png', dpi=300)

# Print Summary Table
df = pd.DataFrame(results)
print(df)

