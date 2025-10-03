Note: installing and configuring CaMa-Flood model v4.2 is prerequisite. perform natural and with levee simulations (i.e. using 1-arcmin map of Japan as in this example)

# Downscaling of coarse resolution flood volume to High resolution depths
```
update all the file paths in "../configs/paths.yml"
```

## 0) Environment (Conda)
```bash
conda env create -f env/environment.yml
conda activate levee-ds
```
## 1) Compute n-year storage (Natural)
```bash
excecute "run_nat_storage.sh" from script directory for future Hazard Mapping or downscaling of return period based storage. 
in case of Hazard Mapping for specific historic event, running this script is not necessary. just provide the simulated storage (sum of rivsto and fldsto) output path to the code and identify the specific time/day for processing for example
"fldsto = np.fromfile(fldsto_file, dtype=np.float32).reshape(-1, ny_inp, nx_inp)[284, :, :]" for model simulated floodplain storage file for the year in which flood event occured. 
```

## 2) Compute n-year storage (Leveed)
```bash
excecute "run_lev_storage.sh" from script directory for future Hazard Mapping or downscaling of return period based storage. 
in case of Hazard Mapping for specific historic event, running this script is not necessary. just provide the simulated storage (sum of rivsto, fldsto and levsto ) output path to the code and identify the specific time/day for processing
```
## 3) Downscale selected RP to 1-sec depths (bbox example)
```bash
once decided the return period (or instance of flood event) for downscaling, execute "run_downscale.sh" to perform downscaling
before execution of downscaling script, check and update the RP such as 1000 (used in paths.yml) or any other.  
```
Notes:
- The map directory (`--base-dir`) must contain `params.txt` and the binary maps (`rivwth_gwdlr.bin`, `rivhgt.bin`, `rivlen.bin`, `levfrc.bin`, `levhgt2.bin`, and `1sec/location.txt` + tiles).
- The storage directory (`--rp-sto-path`) must contain `nat_storage_RP{rp}.bin` and `lev_storage_RP{rp}.bin` generated in steps (1) and (2) for future hazard maps.
- You can change the bbox to target other rivers.
- precomputed `nat_storage_RP{rp}.bin` and `lev_storage_RP{rp}.bin` using large model outputs are available in data repository. users can copy the data and put that in "scripts/n-year-storage" directory and directly execute downscaling step-3  

### Data sources & access
| Dataset                                          | Access / License                          | How to obtain / Reference |
|--------------------------------------------------|-------------------------------------------|---------------------------|
| CaMa-Flood model v4.2                            | Open-source, archived exact version       | DOI https://doi.org/10.5281/zenodo.11091435           |
| 1-arcsec river network & floodplain (Japan)      | **Restricted**                            | Request access from Dai Yamazaki (IIS, UTokyo) |
| GRADES hydrological forcing                      | Public                                    | Yang et al., 2021 (https://doi.org/10.1175/BAMS-D-20-0057.1) |
| MERIT Hydro topography                           | Public                                    | Yamazaki et al., 2019 (https://doi.org/10.1029/2019WR024873) |
| Global levee dataset 						       | Public                                    | Khanh et al., 2025 (https://doi.org/10.1029/2024GL114121)           |
| MLIT discharge & water level                     | Portal access w/ terms                    | https://www1.river.go.jp  |
| GSI hazard maps / Hagibis 2019 flood extent      | Public portals w/ terms                   | https://disaportal.gsi.go.jp/hazardmap/maps/         |

---

## Reproducing Figures

-prerequisite: Complete Steps 0–3 above (environment, n-year storage, and downscaling).  
-Figure scripts assume the repository layout used in this archive; adjust paths at the top of each script if needed.
-The data used to produce the figures are copied in `figures/inp_data/` directory

### Environment packages (for figure scripts)

All the plotting/geo packages required to successfully run the scrits are already included in "env/environment.yml":

```bash
#to avoid Qt/X display warnings do the following;
export MPLBACKEND=Agg   
```
---

### Figure 1 — (Levee fraction and Levee height) 
Script: `figures/fig_1.py`  
Run:
```bash
python figures/fig_1.py
```
Output: `figure_1`. (or as specified in the script)

### Figure 2 — Bar/schematic
Script: `figures/fig_2.py`  
Run:
```bash
python figures/fig_2.py
```
Output: `figure_2.png` (or as specified in the script).

### Figure 3 (part) — schematic illustration of unit catchment along with storage curves (inside/outside/total)
Script: `figures/storage_curve.py`   
Run:
```bash
python figures/storage_curve.py
```
Output: Storage_curves along with levee height reference for selected unit catchment

### Figure 5 — Levee fraction/height, AOI & catchments
Script: `figures/fig_5.py`  
Inputs: levee fraction/height rasters and vectors.  
Run:
```bash
python figures/fig_5.py
```
Notes: Uses `cmcrameri` colormaps. Update AOI bounds at the top if needed.
Output: `figure_5.png` (or as specified in the script).

### Figure 6 — Observed vs simulated confusion maps (Nat / Lev)
Script: `figures/fig_6.py`  
Inputs: observed flood mask, simulated depth rasters (Nat/Lev), optional AOI mask.  
Run:
```bash
python figures/fig_6.py
```
Output: figure that compared hazard maps (city map vs estimated) along with summary metrics printed in-panel.

### Figure 7 — Flood depths and extents in unit catchments (Nat & Lev)
Script: `figures/fig_7.py`  
Inputs: event/simulation rasters, river centerlines, optional AOI.  
Run:
```bash
python figures/fig_7.py
```
Output: `figure_7.png` (or as specified in the script).

### Figure 8 (part) - Multi‑RP tiled panels (Nat vs Lev over selected AOI)
Script: `figures/RP_flood_inundations.py`  
Inputs: downscaled binaries for multiple RPs, levee vectors; AOI bounds in script.  
Run:
```bash
python figures/RP_flood_inundations.py
```
Output: `flood_hazard_area_with_levee_overlay_14km_us.png` Grid of panels (one per RP) with Nat (reds) vs Lev (blues). Area inundated with nat and lev scenarios has been inserted in text boxes of each panel. 

### Figure 9 — Hazard classes & area summaries Map (RP=10 & 100)
Script: `figures/fig_9.py`  
Inputs: downscaled 1-sec depth binaries from Step-3.  
Run:
```bash
python figures/fig_9.py
```
Output: `figure_9.png` Panels per region with hazard classes; prints per-class area stats.

### Figure 10 — Flood‑volume summary across return periods (with supporting figure)
Script: `figures/fld_vol.py`  
Inputs: `nat_storage_RP{rp}.bin` and `lev_storage_RP{rp}.bin` for RPs of interest; levee mask.  
Run:
```bash
python figures/fld_vol.py
```
Output:  `Jp_flood_volume.png` flood volume trends and corresponding percentage reductions for each RP.

---


