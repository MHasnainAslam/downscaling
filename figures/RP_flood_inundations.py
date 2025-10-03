import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.ma as ma
import geopandas as gpd
from shapely.geometry import box
from matplotlib.ticker import FuncFormatter

def compute_grid_area(nx, ny, lat_origin):
    """Computes grid area dynamically based on latitude."""
    earth_radius = 6371000  # meters
    pixel_size = 1.0 / 3600  # degrees

    lat_array = np.linspace(
        lat_origin + (ny / 2) * pixel_size,
        lat_origin - (ny / 2) * pixel_size,
        ny
    )
    pixel_width = (earth_radius * np.radians(pixel_size)) * np.cos(np.radians(lat_array))[:, np.newaxis]
    pixel_height = earth_radius * np.radians(pixel_size)
    return np.repeat(pixel_width, nx, axis=1) * pixel_height

def format_coord(x, pos):
    return f"{x:.2f}"

# ------------------ Configuration ------------------

folder = '../scripts'
#folder = '/data7/hasnain/analysis/downscaling'   #downscale flood vomules to inundation depths for all the rps (listed in this code) in '../scripts' directory then change folder path to '../scripts'
levee_data = './inp_data/shp-files/n35e135_levee_data.shp'

rps = [5, 10, 30, 50, 100, 150, 200, 500, 1000]
nx, ny = 3600, 3600
west_full, east_full = 138.0, 139.0
south_full, north_full = 36.0, 37.0
WEST, EAST = 138.10, 138.14
SOUTH, NORTH = 36.53, 36.56
dx = 1.0 / 3600.0

ix_start = int((WEST - west_full) / dx)
ix_end = int((EAST - west_full) / dx)
iy_start = int((north_full - NORTH) / dx)
iy_end = int((north_full - SOUTH) / dx)

nx_sel, ny_sel = ix_end - ix_start, iy_end - iy_start
grid_area = compute_grid_area(nx_sel, ny_sel, SOUTH)

# ------------------ Load and Crop Levee Data ------------------
levee_vector = gpd.read_file(levee_data)
levee_vector = levee_vector.to_crs("EPSG:4326")

bbox = box(WEST, SOUTH, EAST, NORTH)
levee_vector_crop = levee_vector[levee_vector.intersects(bbox)]

# ------------------ Plot Setup ------------------
ncols = 3
nrows = int(np.ceil(len(rps) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
#plt.subplots_adjust(wspace=0.05, hspace=0.25)
plt.subplots_adjust(wspace=0.1, hspace=0.35)
axes = axes.flatten()

for idx, rp in enumerate(rps):
    lev_file = os.path.join(folder, f"depths_{rp}_lev.bin")   #check the file name in "../scripts/" directory
    nat_file = os.path.join(folder, f"depths_{rp}_Nat.bin")

    lev = np.fromfile(lev_file, dtype=np.float32).reshape(ny, nx)
    nat = np.fromfile(nat_file, dtype=np.float32).reshape(ny, nx)

    lev_crop = lev[iy_start:iy_end, ix_start:ix_end]
    nat_crop = nat[iy_start:iy_end, ix_start:ix_end]

    # Adjust masking to reveal more depth
    lev_masked = np.where(lev_crop < 0.001, np.nan, lev_crop)
    nat_masked = np.where(nat_crop < 0.001, np.nan, nat_crop)

    area_lev = np.nansum(grid_area[~np.isnan(lev_masked)]) / 1e6
    area_nat = np.nansum(grid_area[~np.isnan(nat_masked)]) / 1e6

    ax = axes[idx]

    # Plot natural flood first
    ax.imshow(ma.masked_invalid(nat_masked), extent=[WEST, EAST, SOUTH, NORTH],
              origin='upper', cmap='Reds', vmin=0, vmax=15, alpha=0.9)

    # Plot levee flood over natural
    im = ax.imshow(ma.masked_invalid(lev_masked), extent=[WEST, EAST, SOUTH, NORTH],
                   origin='upper', cmap='Blues', vmin=0, vmax=15, alpha=1)

    # Plot levee lines
    levee_vector_crop.plot(ax=ax, edgecolor='black', linewidth=0.3,
                           facecolor='none', zorder=10)

    # Force axis limits to match subdomain
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)

    # Grid & ticks
    ax.set_title(f"RP = {rp} years", fontsize=14)
    ax.set_xticks(np.linspace(WEST, EAST, 4))
    ax.set_yticks(np.linspace(SOUTH, NORTH, 4))
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(format_coord))
    ax.yaxis.set_major_formatter(FuncFormatter(format_coord))

    ax.text(
        0.02, 0.96,
        f'Nat: {area_nat:.2f} km²\nLev: {area_lev:.2f} km²',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )

# Hide unused subplots
for idx in range(len(rps), len(axes)):
    fig.delaxes(axes[idx])

# Add colorbar
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.6,
                    orientation='vertical', label='Inundation Depth (m)')

# Save and display
plt.savefig('flood_hazard_area_with_levee_overlay_14km_us.png', dpi=600, bbox_inches='tight')

