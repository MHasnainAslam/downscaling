#!/usr/bin/env python
# coding: utf-8

# This code is to distribute coarse-resolution storage data into high-resolution flood depths while considering levee protection. 
# The scenarios analyzed are Scenario A (without Levee) and Scenario B (with levee)

# Prerequisites to use this code are
#	1. Pre-computed storage (simulated storage or frequency-based storage of any return period) for Natural and Levee based scenarios. this code example used frequency-based storage
#	2. Levee parameters (Levee fraction and levee height)
#	3. High Resolution Maps 

# Created by Hasnain (March 2025)

#=================================
#=================================

import os
import matplotlib.pyplot
import numpy as np
import pandas as pd
from pylab import *
import rasterio
from rasterio.transform import from_origin
import yaml

# ================================
# 🔹 Read  File Paths
# ================================
base_dir = "/data7/hasnain/cmf_v420_pkg/map/tej_01min_0"		#Map used for simulations
sec_dir = os.path.join(base_dir, "1sec")				#Hi-res Map
RP_sto_path = "/data7/hasnain/analysis/frequency/n-year-storage/"	#pre-computed storages for various RPs are stored here

levfrc_file = os.path.join(base_dir, "levfrc.bin")
levhgt_file = os.path.join(base_dir, "levhgt2.bin")
params_file = os.path.join(base_dir, "params.txt")
rivhgt_file = os.path.join(base_dir, "rivhgt.bin")
rivwth_file = os.path.join(base_dir, "rivwth_gwdlr.bin")
rivlen_file = os.path.join(base_dir, "rivlen.bin")
location_file = os.path.join(sec_dir, "location.txt")

# ================================
# 🔹 Define Return Period and Extraction Bounding Box (User-Specified)
# ================================
#return_periods = [5, 10, 30, 50, 80, 100, 120, 150, 200, 500, 1000]
rp = 1000 	# select the return period to be used for mapping

#Chikuma River domain to downscale
WEST, EAST = 138.0, 139.0
SOUTH, NORTH = 36.0, 37.0

# the corresponding floodplain layer of each pixel
CATMZZ_NAME = "catmz100.bin"   # or "catmzz.bin" if that’s your tile naming

try:
    with open("../configs/paths.yml") as f:           # or "configs/paths.yml" (see note #3)
        CFG = yaml.safe_load(f)
    base_dir = CFG.get("base_dir", base_dir)
    sec_dir  = os.path.join(base_dir, "1sec")
    RP_sto_path = CFG.get("rp_storage_path", RP_sto_path)

    DS = CFG.get("downscale", {})
    rp    = DS.get("rp", rp)
    WEST  = DS.get("west", WEST)
    EAST  = DS.get("east", EAST)
    SOUTH = DS.get("south", SOUTH)
    NORTH = DS.get("north", NORTH)
    CATMZZ_NAME = DS.get("catmzz_name", CATMZZ_NAME)
except FileNotFoundError:
    pass

# check map domain and data shape. read params file from base directory
with open(params_file, "r") as f:
    lines = f.readlines()

nx_inp = int(lines[0].split()[0])  # Grid count (east-west)
ny_inp = int(lines[1].split()[0])  # Grid count (north-south)
WEST_DOMAIN = float(lines[4].split()[0])  # Western boundary of full domain
EAST_DOMAIN = float(lines[5].split()[0])  # Eastern boundary of full domain
SOUTH_DOMAIN = float(lines[6].split()[0])  # Southern boundary of full domain
NORTH_DOMAIN = float(lines[7].split()[0])  # Northern boundary of full domain

print(
    f"Extracted from params.txt: nx_inp={nx_inp}, ny_inp={ny_inp}, WEST={WEST_DOMAIN}, EAST={EAST_DOMAIN}, SOUTH={SOUTH_DOMAIN}, NORTH={NORTH_DOMAIN}"
)

# ================================
# 🔹 Read Binary Data from Storage, Levee Fraction, and Levee Height Files
# ================================
rivhgt_data = np.fromfile(rivhgt_file, dtype=np.float32).reshape(ny_inp, nx_inp)
rivwth_data = np.fromfile(rivwth_file, dtype=np.float32).reshape(ny_inp, nx_inp)
rivlen_data = np.fromfile(rivlen_file, dtype=np.float32).reshape(ny_inp, nx_inp)

levfrc_data = np.fromfile(levfrc_file, dtype=np.float32).reshape(ny_inp, nx_inp)
levhgt_data = np.fromfile(levhgt_file, dtype=np.float32).reshape(ny_inp, nx_inp)

# calculate river channel storage
Channel_storage = ma.masked_less(rivlen_data * rivwth_data * rivhgt_data, 0)

Channel_storage_min, Channel_storage_max = (
    np.nanmin(Channel_storage),
    np.nanmax(Channel_storage),
)
print(f"Channel Storage: Min = {Channel_storage_min}, Max = {Channel_storage_max}")

input_filename_nat = os.path.join(RP_sto_path, f"nat_storage_RP{rp}.bin")
input_filename_lev = os.path.join(RP_sto_path, f"lev_storage_RP{rp}.bin")

storage_1min_with_levee = np.fromfile(input_filename_lev, dtype=np.float32).reshape(ny_inp, nx_inp)
storage_1min_without_levee = np.fromfile(input_filename_nat, dtype=np.float32).reshape(ny_inp, nx_inp)

print("Storage data loaded successfully for both cases!")

# ================================
# 🔹 Convert Bounding Box to Grid Indices (1-min Resolution)
# ================================
x_min = int((WEST - WEST_DOMAIN) * 60)
x_max = int((EAST - WEST_DOMAIN) * 60)

y_min = int((NORTH_DOMAIN - NORTH) * 60)  # Latitude decreases as index increases
y_max = int((NORTH_DOMAIN - SOUTH) * 60)

# **Ensure indices are within bounds**
x_min = max(0, min(nx_inp - 1, x_min))
x_max = max(0, min(nx_inp - 1, x_max))
y_min = max(0, min(ny_inp - 1, y_min))
y_max = max(0, min(ny_inp - 1, y_max))

print(
    f"Computed Grid Indices: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}"
)


# ================================
# 🔹 Compute 1-sec Grid Shape for Selected Domain
# ================================
nx_selected = int(round((EAST - WEST) * 3600))
ny_selected = int(round((NORTH - SOUTH) * 3600))

# **Preallocate Arrays for Extracted Data**
extracted_flddif = np.zeros((ny_selected, nx_selected), dtype=np.float32)
extracted_catmzz = np.zeros((ny_selected, nx_selected), dtype=np.int8)
extracted_catm_x = np.zeros((ny_selected, nx_selected), dtype=np.int16)
extracted_catm_y = np.zeros((ny_selected, nx_selected), dtype=np.int16)

# ================================
# 🔹 Compute Grid Areas for Selected Domainn (to be used in calculation of storage curves)
# ================================
def compute_grid_area(nx, ny, lat_origin):
    """Computes grid area dynamically based on latitude."""
    earth_radius = 6371000  # Earth radius in meters
    pixel_size = 1.0 / 3600  # 1-sec resolution (degrees)

    # Compute latitude array based on the selected domain's min latitude
    lat_array = np.linspace(
        lat_origin + (ny / 2) * pixel_size, lat_origin - (ny / 2) * pixel_size, ny
    )

    # Compute pixel width dynamically based on latitude
    pixel_width = (earth_radius * np.radians(pixel_size)) * np.cos(
        np.radians(lat_array)
    )[:, np.newaxis]
    pixel_height = earth_radius * np.radians(pixel_size)

    # Compute grid area (m²)
    grid_area = np.repeat(pixel_width, nx, axis=1) * pixel_height

    if grid_area.shape != (ny, nx):
        raise ValueError(f"Grid area shape mismatch: {grid_area.shape} vs ({ny}, {nx})")

    return grid_area


# **Determine Minimum Latitude in the Selected Domain**
min_lat = SOUTH  # Since our domain is SOUTH=36, NORTH=37 for chikuma river

selected_grid_area = compute_grid_area(int(nx_selected), int(ny_selected), min_lat)

print("Grid Area Computed for Selected 1-Sec Domain")
print(f"Grid Area Shape: {selected_grid_area.shape}")

# ================================
# 🔹 Identify Relevant 1-Sec Tiles
# ================================
selected_tiles = []
tile_metadata = {}

with open(location_file, "r") as f:
    lines = f.readlines()[2:]  # Skip header

    for line in lines:
        parts = line.split()
        tile_name = parts[1]
        tile_west, tile_east, tile_south, tile_north = map(float, parts[2:6])
        tile_nx, tile_ny = map(int, parts[6:8])

        if not (
            tile_east < WEST
            or tile_west > EAST
            or tile_north < SOUTH
            or tile_south > NORTH
        ):
            selected_tiles.append(tile_name)
            tile_metadata[tile_name] = {
                "west": tile_west,
                "east": tile_east,
                "south": tile_south,
                "north": tile_north,
                "nx": tile_nx,
                "ny": tile_ny,
            }

print("Selected 1-Sec Tiles:", selected_tiles)


# ================================
# 🔹 Extract Data Tile-by-Tile and Populate into Preallocated Arrays
# ================================
for tile, meta in tile_metadata.items():
    flddif_path = os.path.join(sec_dir, f"{tile}.flddif.bin")
    catmzz_path = os.path.join(sec_dir, f"{tile}.{CATMZZ_NAME}")
    catmxy_path = os.path.join(sec_dir, f"{tile}.catmxy.bin")

    if not os.path.exists(flddif_path) or not os.path.exists(catmzz_path):
        print(f"Warning: Missing files for {tile}. Skipping.")
        continue

    # Load Data from Binary Files
    flddif = np.fromfile(flddif_path, dtype=np.float32).reshape(meta["ny"], meta["nx"])
    catmzz = np.fromfile(catmzz_path, dtype=np.int8).reshape(meta["ny"], meta["nx"])
    catmxy = np.fromfile(catmxy_path, dtype="int16").reshape(2, meta["ny"], meta["nx"])

    catm_x, catm_y = catmxy[0] - 1, catmxy[1] - 1  # Extract Catchment IDs

    # Compute Overlapping Indices in the Final Array
    x_start_global = int((meta["west"] - WEST) * 3600)
    y_start_global = int((NORTH - meta["north"]) * 3600)

    x_end_global = min(x_start_global + meta["nx"], nx_selected)
    y_end_global = min(y_start_global + meta["ny"], ny_selected)

    # Compute Local Tile Indices
    x_start_tile = max(0, -x_start_global)
    y_start_tile = max(0, -y_start_global)

    x_end_tile = min(meta["nx"], nx_selected - x_start_global)
    y_end_tile = min(meta["ny"], ny_selected - y_start_global)

    print(f"DEBUG: Tile {tile}")
    print(
        f"  - Global Indices: x_start={x_start_global}, x_end={x_end_global}, y_start={y_start_global}, y_end={y_end_global}"
    )
    print(
        f"  - Tile Indices: x_start_tile={x_start_tile}, x_end_tile={x_end_tile}, y_start_tile={y_start_tile}, y_end_tile={y_end_tile}"
    )

    # Ensure indices are valid before assignment
    if (y_end_global > y_start_global) and (x_end_global > x_start_global):
        extracted_flddif[y_start_global:y_end_global, x_start_global:x_end_global] = (
            flddif[y_start_tile:y_end_tile, x_start_tile:x_end_tile]
        )

        extracted_catmzz[y_start_global:y_end_global, x_start_global:x_end_global] = (
            catmzz[y_start_tile:y_end_tile, x_start_tile:x_end_tile]
        )

        extracted_catm_x[y_start_global:y_end_global, x_start_global:x_end_global] = (
            catm_x[y_start_tile:y_end_tile, x_start_tile:x_end_tile]
        )

        extracted_catm_y[y_start_global:y_end_global, x_start_global:x_end_global] = (
            catm_y[y_start_tile:y_end_tile, x_start_tile:x_end_tile]
        )

        print(f"uccessfully extracted and mapped data from {tile}")
    else:
        print(f"Skipping tile {tile} due to empty slice")

print(f"Final Extracted Data Shape: {extracted_flddif.shape}")

# ================================
# Function to Calculate Storage Curves
# ================================
def calc_storage_curve(
    water_surface_levels: np.ndarray, flddif: np.ndarray, grdarea: np.ndarray
):
    """Calculate the storage curve.

    Args:
        water_surface_levels (np.ndarray): 1D array of water surface levels.
        flddif (np.ndarray): Field difference data.
        grdarea (np.ndarray): Grid area.
    """

    unique_levels = np.unique(
        np.concatenate((flddif.flatten(), water_surface_levels.flatten()))
    )
    flooded_area = np.histogram(
        flddif,
        bins=np.concatenate((unique_levels, [np.inf])),
        weights=grdarea,
    )[0][:-1]
    volume = np.cumsum(np.diff(unique_levels) * np.cumsum(flooded_area))
    volume = np.concatenate(([0], volume))

    return np.interp(water_surface_levels, unique_levels, volume)


# ================================
# 🔹 Function to convert Coarse resolution storage to Hi-res flood depths
# ================================

#----- Scenario A: Natural -----

def distribute_storage(
    storage_1min, flddif_data, grdarea, catm_x, catm_y, x_min, x_max, y_min, y_max
):
    """
    Distributes 1-minute simulated storage to 1-second resolution considering unique catchments.
    Computes the storage capacity within a catchment and distributes the storage at that level.

    Parameters:
    storage_1min (2D numpy array): 1-minute storage values.
    flddif_data (2D numpy array): Field difference data at 1-second resolution (selected domain).
    grdarea (2D numpy array): Precomputed grid area at 1-second resolution (selected domain).
    catm_x, catm_y (2D numpy arrays): Catchment identifiers at 1-sec resolution (full domain).
    x_min, x_max, y_min, y_max (int): Bounding box indices for the selected domain.

    Returns:
    storage_1sec (2D numpy array): 1-second resolution storage values.
    depths_1sec (2D numpy array): Computed depth values at 1-second resolution.
    """
    # Create empty storage and depth arrays
    storage_1sec = np.zeros_like(flddif_data)
    depths_1sec = np.zeros_like(flddif_data)

    # Identify valid storage cells within the selected domain
    y_idx, x_idx = np.where((storage_1min[y_min:y_max, x_min:x_max] > 0))
    y_idx += y_min  # Adjust indices to match full domain
    x_idx += x_min
    unique_catchments = set(zip(x_idx, y_idx))

    # instead of doing computations on 2D array, we flatten the data for easier processing
    orig_shape = catm_x.shape
    catm_x = catm_x.flatten()
    catm_y = catm_y.flatten()
    flddif_data = flddif_data.flatten()
    grdarea = grdarea.flatten()
    storage_1sec = storage_1sec.flatten()
    depths_1sec = depths_1sec.flatten()

    df = pd.DataFrame(
        {
            "idx": np.arange(catm_x.size),
            "catm_x": catm_x,
            "catm_y": catm_y,
        }
    )
    df["catm"] = (
        df["catm_x"].astype(np.int64) * (np.max(df["catm_y"]) + 1) + df["catm_y"]
    )

    for _, group in df.groupby("catm"):
        x_idx = group["catm_x"].iloc[0]
        y_idx = group["catm_y"].iloc[0]
        idx = np.array(group["idx"])  # List of indexes for this catchment

        if (x_idx, y_idx) not in unique_catchments:
            continue

        # Extract relevant grid cells within selected domain
        flddif_valid = flddif_data[idx]
        grdarea_valid = grdarea[idx]

        # Sort flddif in ascending order (low elevation first)
        sorted_indices = np.argsort(flddif_valid)
        flddif_sorted = flddif_valid[sorted_indices]
        grdarea_sorted = grdarea_valid[sorted_indices]

        # Compute total storage (to be distributed in Hi-rse maps) for each unit catchment catchment
        storage_total = storage_1min[y_idx, x_idx] - Channel_storage[y_idx, x_idx]

        # Compute volume capacity at different water heights
        water_surface_levels = np.linspace(
            flddif_sorted.min(), flddif_sorted.max(), len(flddif_sorted)
        )
        volume_capacity = calc_storage_curve(
            water_surface_levels, flddif_sorted, grdarea_sorted
        )

        # Find the required water level for storage
        if storage_total > volume_capacity[-1]:
            remaining_storage = storage_total - volume_capacity[-1]
            max_area = np.sum(grdarea_sorted)
            extra_height = remaining_storage / max_area
            water_surface = flddif_sorted[-1] + extra_height
        else:
            water_surface = np.interp(
                storage_total, volume_capacity, water_surface_levels
            )

        # Compute water depths
        water_depths = np.maximum(water_surface - flddif_data[idx], 0)

        # Assign computed values to storage and depth arrays
        storage_1sec[idx] = storage_total
        depths_1sec[idx] = water_depths

    # Reshape arrays to original 2D shape
    storage_1sec = storage_1sec.reshape(orig_shape)
    depths_1sec = depths_1sec.reshape(orig_shape)

    return storage_1sec, depths_1sec

storage_1sec, depths_1sec = distribute_storage(
    storage_1min_without_levee,
    extracted_flddif,
    selected_grid_area,
    extracted_catm_x,
    extracted_catm_y,
    x_min,
    x_max,
    y_min,
    y_max,
)

depths_1sec.tofile(f"depths_{rp}_Nat.bin")

#----- Scenario B: With Levee-----

#calculation for inside outside storage based on catmzz for grids with levee and then downscale 

def distribute_storage(
    storage_1min, flddif_data, catmzz_data, grdarea, catm_x, catm_y, x_min, x_max, y_min, y_max
):

    # Step 1: Initialize empty storage, depth, and water surface arrays
    storage_1sec = np.zeros_like(flddif_data)
    depths_1sec = np.zeros_like(flddif_data)
    water_surface_1sec = np.zeros_like(flddif_data)

    # Identify valid storage cells
    y_idx, x_idx = np.where((storage_1min[y_min:y_max, x_min:x_max] > 0))
    y_idx += y_min
    x_idx += x_min
    unique_catchments = set(zip(x_idx, y_idx))

    # Flatten data arrays for easier processing
    orig_shape = flddif_data.shape
    flddif_data = flddif_data.flatten()
    catmzz_data = catmzz_data.flatten()
    grdarea = grdarea.flatten()
    storage_1sec = storage_1sec.flatten()
    depths_1sec = depths_1sec.flatten()
    water_surface_1sec = water_surface_1sec.flatten()

    df = pd.DataFrame(
        {"idx": np.arange(catm_x.size), "catm_x": catm_x.flatten(), "catm_y": catm_y.flatten()}
    )
    df["catm"] = (
        df["catm_x"].astype(np.int64) * (np.max(df["catm_y"]) + 1) + df["catm_y"]
    )

    for _, group in df.groupby("catm"):
        x_idx = group["catm_x"].iloc[0]
        y_idx = group["catm_y"].iloc[0]
        idx = np.array(group["idx"])

        if (x_idx, y_idx) not in unique_catchments:
            continue

        # Extract grid cell values
        flddif_valid = flddif_data[idx]
        grdarea_valid = grdarea[idx]
        catmzz_valid = catmzz_data[idx]

        # Define levee parameters
        levhgt_value = levhgt_data[y_idx, x_idx]
        levfrc_value = levfrc_data[y_idx, x_idx]

        # Compute total storage
        storage_total = storage_1min[y_idx, x_idx] - Channel_storage[y_idx, x_idx]

        # **Case 1: No levee parameters available (Use Total Storage Curve Directly)**
        if np.isnan(levhgt_value) or levfrc_value == 0 or levfrc_value == 1:
            # Compute volume capacity for entire floodplain (no levee case)
            sorted_indices = np.argsort(flddif_valid)
            flddif_sorted = flddif_valid[sorted_indices]
            grdarea_sorted = grdarea_valid[sorted_indices]

            water_surface_levels = np.linspace(flddif_sorted.min(), flddif_sorted.max(), len(flddif_sorted))
            volume_total = calc_storage_curve(water_surface_levels, flddif_sorted, grdarea_sorted)

            # Interpolate water surface using total storage curve
            water_surface_total = np.interp(storage_total, volume_total, water_surface_levels)

            # Apply to all grids in the catchment
            water_surface = np.full(len(idx), water_surface_total)
            water_depths = np.maximum(water_surface - flddif_valid, 0)

            # Store values
            storage_1sec[idx] = storage_total
            depths_1sec[idx] = water_depths
            water_surface_1sec[idx] = water_surface

            continue  # Skip to next catchment

        # **Case 2: Levee parameters available**

        #inside_mask = catmzz_valid <= (levfrc_value * 10) # for 10 layered reletaive pixels
        inside_mask = catmzz_valid <= max((levfrc_value * 100),1) #for 100 layered relative pixels
        outside_mask = ~inside_mask

        # Sort based on elevation (flddif)
        sorted_indices_inside = np.argsort(flddif_valid[inside_mask])
        sorted_indices_outside = np.argsort(flddif_valid[outside_mask])
        sorted_indices_total = np.argsort(flddif_valid)

        flddif_inside_sorted = flddif_valid[inside_mask][sorted_indices_inside]
        grdarea_inside_sorted = grdarea_valid[inside_mask][sorted_indices_inside]

        flddif_outside_sorted = flddif_valid[outside_mask][sorted_indices_outside]
        grdarea_outside_sorted = grdarea_valid[outside_mask][sorted_indices_outside]

        flddif_total_sorted = flddif_valid[sorted_indices_total]
        grdarea_total_sorted = grdarea_valid[sorted_indices_total]

        # Compute storage volume curves
        water_surface_levels_in = np.linspace(flddif_inside_sorted.min(), levhgt_value, len(flddif_inside_sorted))
        water_surface_levels_out = np.linspace(flddif_outside_sorted.min(), levhgt_value, len(flddif_outside_sorted))
        water_surface_levels_total = np.linspace(flddif_total_sorted.min(), flddif_total_sorted.max(), len(flddif_total_sorted))

        volume_inside = calc_storage_curve(water_surface_levels_in, flddif_inside_sorted, grdarea_inside_sorted)
        volume_outside = calc_storage_curve(water_surface_levels_out, flddif_outside_sorted, grdarea_outside_sorted)
        volume_total = calc_storage_curve(water_surface_levels_total, flddif_total_sorted, grdarea_total_sorted)

        max_storage_inside = max(volume_inside)
        max_storage_outside = max(volume_outside)
        max_storage_total = max(volume_total)

        # **Fix: Initialize `water_surface_total`**
        
        # Compute water levels
        if storage_total <= max_storage_inside:
            water_surface_inside = np.interp(storage_total, volume_inside, water_surface_levels_in)
            water_surface_outside = 0  # Outside remains dry
        elif storage_total <= (max_storage_inside + max_storage_outside):
            outside_storage = storage_total - max_storage_inside
            water_surface_inside = levhgt_value
            water_surface_outside = np.interp(outside_storage, volume_outside, water_surface_levels_out)
        else:
            water_surface_inside = water_surface_outside = np.interp(storage_total, volume_total, water_surface_levels_total)

        # Assign computed values
        water_surface_inside_grid = np.full(len(flddif_inside_sorted), water_surface_inside)
        water_surface_outside_grid = np.full(len(flddif_outside_sorted), water_surface_outside)

        water_surface = np.zeros(len(idx))
        water_surface[inside_mask] = water_surface_inside_grid
        water_surface[outside_mask] = water_surface_outside_grid

        water_depths = np.maximum(water_surface - flddif_valid, 0)

        # Store values in output arrays
        storage_1sec[idx] = storage_total
        depths_1sec[idx] = water_depths
        water_surface_1sec[idx] = water_surface

        if y_idx == 565 and x_idx == 910:  #565, 910 ; 556, 919 ; 549, 921; 548, 922
            print(f"Catchment ({y_idx}, {x_idx})")
            print(f"Storage total: {storage_total}")
            print(f"Levee Height: {levhgt_value}")
            print(f"Levee fraction: {levfrc_value}")
            print(f"inside_mask shape: {inside_mask.shape}, water_surface_inside shape: {water_surface_inside_grid.shape}")
            print(f"outside_mask shape: {outside_mask.shape}, water_surface_outside shape: {water_surface_outside_grid.shape}")
            print(f"Inside Grid Count: {np.count_nonzero(inside_mask)}")
            print(f"outside Grid Count: {np.count_nonzero(outside_mask)}")

            #extract data to check the computed curves
            df_dict = {
                "water_surface_levels_inside": list(water_surface_levels_in),
                "volume_inside": list(volume_inside),
                "water_surface_levels_outside": list(water_surface_levels_out),
                "volume_outside": list(volume_outside),
                "water_surface_levels_total": list(water_surface_levels_total),
                "volume_total": list(volume_total),
            }

            # **Find max length among all lists**
            max_len = max(len(lst) for lst in df_dict.values())

            # **Pad shorter lists with NaN**
            for key in df_dict:
                df_dict[key] += [np.nan] * (max_len - len(df_dict[key]))  # Padding

            # **Create DataFrame & Save as CSV**
            df = pd.DataFrame(df_dict)
            df.to_csv(f"catchment_{int(y_idx)}_{int(x_idx)}_volume_data.csv", index=False)

            resolution = 1 / 3600  
            transform = from_origin(x_min * resolution, y_max * resolution, resolution, resolution)

            # --- Create flddif map for this catchment only ---
            flddif_catchment = np.full_like(flddif_data, np.nan, dtype='float32')
            flddif_catchment[idx] = flddif_data[idx]
            flddif_catchment_2d = flddif_catchment.reshape(orig_shape)

            # Save as GeoTIFF
            with rasterio.open(
                f"catchment_{y_idx}_{x_idx}_flddif.tif",
                'w',
                driver='GTiff',
                height=flddif_catchment_2d.shape[0],
                width=flddif_catchment_2d.shape[1],
                count=1,
                dtype='float32',
                crs="EPSG:4326",  # Change this if needed
                transform=transform,  # Use same transform as for catmxy
                nodata=np.nan,
            ) as dst:
                dst.write(flddif_catchment_2d.astype('float32'), 1)
            
    # Reshape outputs to original 2D shape
    storage_1sec = storage_1sec.reshape(orig_shape)
    depths_1sec = depths_1sec.reshape(orig_shape)
    water_surface_1sec = water_surface_1sec.reshape(orig_shape)

    return storage_1sec, depths_1sec, water_surface_1sec

storage_1sec, depths_1sec, water_surface_1sec = distribute_storage(
    storage_1min_with_levee,
    extracted_flddif,
    extracted_catmzz,
    selected_grid_area,
    extracted_catm_x,
    extracted_catm_y,
    x_min,
    x_max,
    y_min,
    y_max,
)

depths_1sec.tofile(f"depths_{rp}_lev.bin")
