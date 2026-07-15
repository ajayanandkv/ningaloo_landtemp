#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

# Avoid GEOS topology errors
matplotlib.rcParams["agg.path.chunksize"] = 10000

# Rotated pole setup
pole_latitude  = 60.31
pole_longitude = 147.63
rotated_pole = ccrs.RotatedPole(pole_latitude=pole_latitude, pole_longitude=pole_longitude)

# Paths and settings
base_path = "/data2/ajay/WRF_MHW/wrf_outputs/derived_outputs/mon_mean/"
domains = ["R3", "R5"]  # average both
ensemble_dates = ["2011-01-13", "2011-01-14", "2011-01-15", "2011-01-16", "2011-01-17"]

runs = ["CNTL", "EXP-SST80", "EXP-SST60", "EXP-SST40", "EXP-SST20"]
titles_rev = ["20SST-ClimSST", "40SST-ClimSST", "60SST-ClimSST", "80SST-ClimSST", "CNTL-ClimSST"]

# Plot order (to match your original: reversed)
#titles_rev = titles[::-1]
runs_rev = runs[::-1]

# Load lat/lon from R5 grid (as you did)
ref_file = "/data2/ajay/WRF_MHW/wrf_outputs/compressed_outputs/R5/CNTL/2011-01-13/wrfout_d01_2011-03-31_00:00:00"
ds_ref = xr.open_dataset(ref_file)
lat = ds_ref["XLAT"].isel(Time=0).squeeze()
lon = ds_ref["XLONG"].isel(Time=0).squeeze()

# ---------------- PRE-SCAN for robust vmax across ALL domains/ens/runs ----------------
all_diffs = []

for run in runs_rev:
    for domain in domains:
        for date in ensemble_dates:
            run_file  = os.path.join(base_path, domain, run, date, "slp_d01_2011-03-monmean.nc")
            clim_file = os.path.join(base_path, domain, "EXP-SSTClim", date, "slp_d01_2011-03-monmean.nc")

            if not (os.path.exists(run_file) and os.path.exists(clim_file)):
                continue

            with xr.open_dataset(run_file) as rds, xr.open_dataset(clim_file) as cds:
                diff = (rds["slp"] - cds["slp"]).values
                all_diffs.append(diff)

stack = np.array(all_diffs) if len(all_diffs) > 0 else np.array([])
if stack.size == 0:
    vmax = 1.0
else:
    vmax = float(np.nanpercentile(np.abs(stack), 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
vmin = -vmax

# ---------------- PLOT ----------------
fig, axs = plt.subplots(
    nrows=1, ncols=5, figsize=(22, 4),
    subplot_kw={"projection": rotated_pole}
)
fig.subplots_adjust(wspace=0.2)
fig.suptitle(r"March, 2011 - $\Delta$ MSLP(hPa)", fontsize=16, weight="bold")

for j, run in enumerate(runs_rev):
    diffs_all = []

    for domain in domains:
        for date in ensemble_dates:
            run_file  = os.path.join(base_path, domain, run, date, "slp_d01_2011-03-monmean.nc")
            clim_file = os.path.join(base_path, domain, "EXP-SSTClim", date, "slp_d01_2011-03-monmean.nc")

            if not (os.path.exists(run_file) and os.path.exists(clim_file)):
                continue

            run_da  = xr.open_dataset(run_file)["slp"]
            clim_da = xr.open_dataset(clim_file)["slp"]
            diffs_all.append(run_da - clim_da)

    if len(diffs_all) == 0:
        # still draw an empty panel so the figure layout stays consistent
        ax = axs[j]
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.set_title(titles_rev[j], fontsize=12, weight="bold")
        ax.text(0.02, 0.95, f"({chr(97 + j)})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")
        continue

    # Concatenate across domains × ensemble_dates
    slp_stack = xr.concat(diffs_all, dim="ensemble")
    mean_diff = slp_stack.mean(dim="ensemble")

    # Significance mask (≥80% same sign)
    n_models = slp_stack.sizes["ensemble"]
    threshold = int(np.ceil(0.8 * n_models))
    signs = np.sign(slp_stack)
    significant_mask = ((signs > 0).sum(dim="ensemble") >= threshold) | \
                       ((signs < 0).sum(dim="ensemble") >= threshold)

    ax = axs[j]
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    ax.text(0.02, 0.95, f"({chr(97 + j)})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

    p = ax.pcolormesh(
        lon, lat, mean_diff.values,
        cmap=plt.cm.RdBu_r,
        shading="auto",
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )

    # Optional stippling (sparse)
    # step = 7
    # y_idx, x_idx = np.where(significant_mask.values)
    # keep = (np.arange(len(y_idx)) % step == 0)
    # ax.plot(lon.values[y_idx[keep], x_idx[keep]],
    #         lat.values[y_idx[keep], x_idx[keep]],
    #         "+", color="k", markersize=0.4,
    #         transform=ccrs.PlateCarree(), alpha=0.5)

    ax.set_title(titles_rev[j], fontsize=12, weight="bold")
    print(f"{titles_rev[j]} | n={n_models} | Min={mean_diff.min().item():.3f} Max={mean_diff.max().item():.3f}")

# Shared colorbar
cbar_ax = fig.add_axes([0.3, 0.06, 0.4, 0.04])
fig.colorbar(p, cax=cbar_ax, orientation="horizontal", label= "hPa")

plt.savefig("MSLP_diff_March.jpg", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

