import os
import numpy as np
import pandas as pd
import rasterio
from lmoments3 import distr
from scipy.stats import kstest
import warnings

base_dir = "/data7/hasnain/cmf_v420_pkg/map/tej_01min_0"
sto_path = "/data7/hasnain/cmf_v420_pkg/out/tej_vic_lev2/"
output_dir = "n-year-storage"

# Specify the range of years to process
start_year = 1979
end_year = 2019

# Fixed return periods for which storage will be calculated
fixed_return_periods = [5, 10, 30, 50, 80, 100, 120, 150, 200, 500, 1000]

# Enable or disable specific distributions for analysis
enabled_distributions = ["GEV", "GAM", "PE3", "GUM", "WEI", "WAK"]

try:
    import yaml
    with open("../configs/paths.yml") as f:
        CFG = yaml.safe_load(f) or {}

    base_dir = CFG.get("base_dir", base_dir)
    sto_path = CFG.get("sto_path_lev", sto_path)
    output_dir = CFG.get("output_dir", output_dir)

    YRS = CFG.get("analysis_years", {})
    start_year = int(YRS.get("start", start_year))
    end_year   = int(YRS.get("end",   end_year))

    RPS = CFG.get("return_periods", {})
    lev_rps = RPS.get("RPs")
    if lev_rps is not None:
        if isinstance(lev_rps, (list, tuple)):
            fixed_return_periods = [int(x) for x in lev_rps]
        else:
            fixed_return_periods = [int(x) for x in str(lev_rps).split(",") if x]

    ED = CFG.get("enabled_distributions")
    if ED is not None:
        if isinstance(ED, (list, tuple)):
            enabled_distributions = [str(x).strip() for x in ED]
        else:
            enabled_distributions = [s.strip() for s in str(ED).split(",") if s.strip()]
except FileNotFoundError:
    pass

params_file = os.path.join(base_dir, "params.txt")

with open(params_file, "r") as f:
    lines = f.readlines()

nx_inp = int(lines[0].split()[0])  # Grid count (east-west)
ny_inp = int(lines[1].split()[0])  # Grid count (north-south)
WEST_DOMAIN = float(lines[4].split()[0])  # Western boundary of full domain
EAST_DOMAIN = float(lines[5].split()[0])  # Eastern boundary of full domain
SOUTH_DOMAIN = float(lines[6].split()[0])  # Southern boundary of full domain
NORTH_DOMAIN = float(lines[7].split()[0])  # Northern boundary of full domain

print(
    f"✅ Extracted from params.txt: nx_inp={nx_inp}, ny_inp={ny_inp}, WEST={WEST_DOMAIN}, EAST={EAST_DOMAIN}, SOUTH={SOUTH_DOMAIN}, NORTH={NORTH_DOMAIN}"
)

grid_shape  = ny_inp, nx_inp
print(grid_shape)

annual_max_storage = np.full((end_year - start_year + 1, *grid_shape), np.nan)
print(annual_max_storage.shape)

for year in range(start_year, end_year + 1):
    print(f"Processing storage for year {year}...")
    try:
        # Paths to the storage components
        rivsto_file = os.path.join(sto_path, f"rivsto{year}.bin")
        fldsto_file = os.path.join(sto_path, f"fldsto{year}.bin")
        levsto_file = os.path.join(sto_path, f"levsto{year}.bin")

        # Read each file
        with open(rivsto_file, "rb") as f:
            rivsto = np.fromfile(f, dtype=np.float32).reshape(-1, *grid_shape)
        with open(fldsto_file, "rb") as f:
            fldsto = np.fromfile(f, dtype=np.float32).reshape(-1, *grid_shape)
        with open(levsto_file, "rb") as f:
            levsto = np.fromfile(f, dtype=np.float32).reshape(-1, *grid_shape)

        # Compute total storage
        total_storage = rivsto + fldsto + levsto


        # Find the annual maximum storage at each grid cell
        annual_max_storage[year - start_year] = np.nanmax(total_storage, axis=0)

    except Exception as e:
        warnings.warn(f"Error processing year {year}: {e}")
        continue

# Define the distributions to test
distributions = {
    "GEV": distr.gev,
    "GAM": distr.gam,
    "PE3": distr.pe3,
    "GUM": distr.gum,
    "WEI": distr.wei,
    "WAK": distr.wak
}

# Filter distributions based on enabled list
distributions = {name: dist for name, dist in distributions.items() if name in enabled_distributions}

# Function to calculate Goodness-of-Fit metrics (K-S only)
def compute_goodness_of_fit(storage, fitted_dist):
    ks_stat, _ = kstest(storage, lambda x: fitted_dist.cdf(x))
    return ks_stat

# Dictionary to store binary outputs for each return period
binary_results = {rp: np.full(grid_shape, np.nan, dtype=np.float32) for rp in fixed_return_periods}

# Main loop for frequency analysis
result_stats = []

for r in range(grid_shape[0]):
    for c in range(grid_shape[1]):
        if np.all(np.isnan(annual_max_storage[:, r, c])):
            continue
        storage = annual_max_storage[:, r, c]
        storage = storage[~np.isnan(storage)]
        if len(storage) < 2:
            continue
        try:
            row_results = {"GridRow": r, "GridCol": c}
            for dist_name, dist in distributions.items():
                try:
                    params = dist.lmom_fit(storage)
                    fitted_dist = dist(**params)
                    ks_stat = compute_goodness_of_fit(storage, fitted_dist)
                    row_results[f"{dist_name}_KS"] = ks_stat
                    
                    # Compute storage for each fixed return period
                    for rp in fixed_return_periods:
                        prob = 1 - 1 / rp
                        rp_storage = fitted_dist.ppf(prob)
                        row_results[f"{dist_name}_RP_{rp}_Storage"] = rp_storage
                        if dist_name == "GEV":
                            binary_results[rp][r, c] = rp_storage
                except Exception as e:
                    warnings.warn(f"Error fitting {dist_name} at cell ({r}, {c}): {e}")
                    continue
            result_stats.append(row_results)
        except Exception as e:
            warnings.warn(f"Error at cell ({r}, {c}): {e}")
            continue

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save statistics to CSV
stats_df = pd.DataFrame(result_stats)
output_csv =  os.path.join(output_dir, f"lev_Storage_Stats_with_Distributions_all_grids.csv")
stats_df.to_csv(output_csv, index=False)
print(f"Statistics saved to {output_csv}")

# Save binary results for each return period
for rp, binary_result in binary_results.items():
    binary_output_filename = os.path.join(output_dir, f"lev_storage_RP{rp}.bin")
    binary_result.tofile(binary_output_filename)
    print(f"Binary results saved to {binary_output_filename}")
