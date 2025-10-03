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
once decided the return period (or instance of flood event) for downscaling, execute "run_downscale.sh" to operform downscaling
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
