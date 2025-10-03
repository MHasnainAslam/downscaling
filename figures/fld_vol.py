import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- Configuration ----------------
folder = "../scripts/n-year-storage"
#folder = "/data7/hasnain/analysis/frequency/n-year-storage" #calculate RP_storage for all the entries in rp_list and save in "../scripts/n-year-storage" then update the path here
levfrc_path = "./inp_data/levfrc.bin" 
rp_list = [5, 10, 30, 50, 80, 100, 120, 150, 200, 500, 1000]
nx, ny = 1500, 1320  # from params.txt

# ---------------- Load levee mask ----------------
levfrc = np.fromfile(levfrc_path, dtype=np.float32).reshape(ny, nx)
lev_mask = levfrc > 0

# ---------------- Initialize ----------------
vol_nat, vol_lev, pct_reduction, valid_rps = [], [], [], []

# ---------------- Loop through RPs ----------------
for rp in rp_list:
    path_nat = os.path.join(folder, f"nat_storage_RP{rp}.bin")
    path_lev = os.path.join(folder, f"lev_storage_RP{rp}.bin")
    
    if not (os.path.isfile(path_nat) and os.path.isfile(path_lev)):
        print(f"Missing file for RP {rp}")
        continue

    try:
        nat = np.fromfile(path_nat, dtype=np.float32).reshape(ny, nx)
        lev = np.fromfile(path_lev, dtype=np.float32).reshape(ny, nx)
        
        # Apply mask
        nat = np.where((nat <= 0) | (~lev_mask), np.nan, nat)
        lev = np.where((lev <= 0) | (~lev_mask), np.nan, lev)
        
        total_nat = np.nansum(nat)
        total_lev = np.nansum(lev)
        reduction = 100 * (total_nat - total_lev) / total_nat if total_nat else 0

        vol_nat.append(total_nat)
        vol_lev.append(total_lev)
        pct_reduction.append(reduction)
        valid_rps.append(rp)

    except Exception as e:
        print(f"Error at RP{rp}: {e}")

# ---------------- Plotting ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Volume Comparison
ax1.plot(valid_rps, vol_nat, 'o-r', label='Natural')
ax1.plot(valid_rps, vol_lev, 'o-b', label='With Levee')
ax1.fill_between(valid_rps, vol_nat, vol_lev, color='gray', alpha=0.3)
ax1.set_title("Flood Volume per RP (Levee Grids Only)", fontsize=18)
ax1.set_xlabel("Return Period (Years)", fontsize=16)
ax1.set_ylabel("Flood Volume (mÂ³)", fontsize=16)
ax1.tick_params(axis='both', labelsize=14)
ax1.grid(True)
ax1.legend()

# Percentage Reduction
ax2.plot(valid_rps, pct_reduction,  's--g', label='Percent Reduction (%)')
ax2.set_title("Percentage Reduction in Volume", fontsize=18)
ax2.set_xlabel("Return Period (Years)", fontsize=16)
ax2.set_ylabel("Reduction (%)", fontsize=16)
ax2.tick_params(axis='both', labelsize=14)
ax2.grid(True, axis='y')

plt.tight_layout()
plt.savefig("Jp_flood_volume.png", dpi=300, bbox_inches='tight')
#plt.show()


