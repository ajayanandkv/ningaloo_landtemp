#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams["agg.path.chunksize"] = 10000
#matplotlib.rcParams["contour.algorithm"] = "mpl2014" 
# ================== CONFIG ==================
base_path = "/data2/ajay/WRF_MHW/wrf_outputs/derived_outputs/mon_mean"
domains = ["R3", "R5"]
ensemble_dates = ["2011-01-13","2011-01-14","2011-01-15","2011-01-16","2011-01-17"]

year, month, mname = "2011", "03", "March"
runs   = ["EXP-SST20","EXP-SST40","EXP-SST60","EXP-SST80","CNTL"]
titles = ["20SST-ClimSST","40SST-ClimSST","60SST-ClimSST","80SST-ClimSST","CNTL-ClimSST"]

# Crop (apply to ALL plots)
CROP = dict(south_north=slice(15, -5), west_east=slice(5, -5))

# 5 rows (each row has its own colorbar, except row 2+3 share)
ROWS = [
    dict(name="DLR",    kind="wrf",   var="GLW",      scale=1.0, unit="W m$^{-2}$", mask="land"),
    dict(name="Low CF", kind="cloud", var="cloudfrac",layer=0,   scale=100.0, unit="%", mask="land"),
    dict(name="Mid CF", kind="cloud", var="cloudfrac",layer=1,   scale=100.0, unit="%", mask="land"),
    dict(name="Q2",     kind="wrf",   var="Q2",       scale=1.0, unit="g kg$^{-1}$", mask="land"),
    dict(name="PBLH",   kind="wrf",   var="PBLH",     scale=1.0, unit="m", mask="land"),
]

CMAP = "seismic"
dpi = 300

# Rotated pole
rotated_pole = ccrs.RotatedPole(pole_latitude=60.31, pole_longitude=147.63)

# ================== HELPERS ==================
def get_latlon_and_masks(domain_ref="R5"):
    ref = f"/data2/ajay/WRF_MHW/wrf_outputs/compressed_outputs/{domain_ref}/CNTL/2011-01-13/wrfout_d01_{year}-03-31_00:00:00"
    ds = xr.open_dataset(ref)

    lat = ds["XLAT"].isel(Time=0, **CROP).squeeze()
    lon = ds["XLONG"].isel(Time=0, **CROP).squeeze()
    mrsos = ds["SMOIS"].isel(Time=0, soil_layers_stag=3, **CROP).squeeze()

    ocean = xr.where(mrsos > 0.5, 1, 0)  # 1 land
    land  = xr.where(mrsos > 0.5, 0, 1)  # 1 ocean

    ds.close()
    return lon, lat, land, ocean

def load_diff_ensemble(domain, run, var):
    runf  = f"{base_path}/{domain}/{run}/ensemble/{var}_d01_{year}-{month}-monmean.nc"
    climf = f"{base_path}/{domain}/EXP-SSTClim/ensemble/{var}_d01_{year}-{month}-monmean.nc"
    if not (os.path.exists(runf) and os.path.exists(climf)):
        return None
    with xr.open_dataset(runf) as dr, xr.open_dataset(climf) as dc:
        return dr[var].isel(**CROP) - dc[var].isel(**CROP)

def load_diff_by_dates(domain, run, var):
    diffs = []
    for date in ensemble_dates:
        runf  = f"{base_path}/{domain}/{run}/{date}/{var}_d01_{year}-{month}-monmean.nc"
        climf = f"{base_path}/{domain}/EXP-SSTClim/{date}/{var}_d01_{year}-{month}-monmean.nc"
        if not (os.path.exists(runf) and os.path.exists(climf)):
            continue
        diffs.append(
            xr.open_dataset(runf)[var].isel(**CROP) -
            xr.open_dataset(climf)[var].isel(**CROP)
        )
    if len(diffs) == 0:
        return None
    return xr.concat(diffs, dim="ens").mean("ens")

def mean_diff_row(run, row):
    if row["kind"] == "cloud":
        diffs = []
        for dom in domains:
            runf  = f"{base_path}/{dom}/{run}/ensemble/cloudfrac_d01_{year}-{month}-monmean.nc"
            climf = f"{base_path}/{dom}/EXP-SSTClim/ensemble/cloudfrac_d01_{year}-{month}-monmean.nc"
            if not (os.path.exists(runf) and os.path.exists(climf)):
                continue
            with xr.open_dataset(runf) as dr, xr.open_dataset(climf) as dc:
                da = (dr["cloudfrac"].isel(ncl0=row["layer"], **CROP) -
                      dc["cloudfrac"].isel(ncl0=row["layer"], **CROP)) * row["scale"]
                diffs.append(da)
        if len(diffs) == 0:
            return None
        return xr.concat(diffs, dim="dom").mean("dom")

    diffs = []
    for dom in domains:
        da = load_diff_ensemble(dom, run, row["var"])
        if da is None:
            da = load_diff_by_dates(dom, run, row["var"])
        if da is not None:
            diffs.append(da * row["scale"])
    if len(diffs) == 0:
        return None
    return xr.concat(diffs, dim="dom").mean("dom")

def apply_mask(da, land_mask, ocean_mask, how):
    if da is None or how == "none":
        return da
    if how == "land":
        return da.where(land_mask == 1)
    if how == "ocean":
        return da.where(ocean_mask == 1)
    return da

def symmetric_vlims(arrs, pct=99):
    vals = []
    for a in arrs:
        if a is None:
            continue
        v = a.values
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(np.abs(v).ravel())
    if not vals:
        return -1, 1
    flat = np.concatenate(vals)
    vmax = np.nanpercentile(flat, pct)
    return -vmax, vmax

# ================== MAIN ==================
lon, lat, land_mask, ocean_mask = get_latlon_and_masks("R5")

nrows, ncols = 5, 5
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(22, 16),
    subplot_kw={"projection": rotated_pole}
)

# 2) reduce gaps
fig.subplots_adjust(wspace=-0.5, hspace=0.1)

panel = 0
cloud_mappable = None  # 1) shared colorbar for row 2 & 3

for r, row in enumerate(ROWS):
    row_maps = []
    for run in runs:
        da = mean_diff_row(run, row)
        da = apply_mask(da, land_mask, ocean_mask, row["mask"])
        row_maps.append(da)

    if row["name"] in ["Low CF", "Mid CF"]:
        vmin, vmax = -8, 8
    else:
        vmin, vmax = symmetric_vlims(row_maps, pct=99)

    mappable = None
    for c, run in enumerate(runs):
        ax = axs[r, c]
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")

        da = row_maps[c]
        if da is None:
            ax.text(0.5, 0.5, "Missing", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=10)
        else:
            data = np.ma.masked_invalid(da.values)
            levels = np.linspace(vmin, vmax, 21)
            mappable = ax.pcolormesh(
                lon, lat, data, shading="auto",
                cmap=CMAP,
                norm=TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax),
                transform=ccrs.PlateCarree()
            )

        # 3) panel label fontsize
        ax.text(0.02, 0.95, f"({chr(97+panel)})", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")
        panel += 1

        # 3) main headings fontsize
        if r == 0:
            ax.set_title(titles[c], fontsize=16, weight="bold")
        if c == 0:
            ax.text(-0.12, 0.5, r"$\Delta$" + row["name"], transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontsize=16, weight="bold")

        # capture mappable for shared cloud colorbar
        if r in [1, 2] and mappable is not None:
            cloud_mappable = mappable

    if mappable is not None and r not in [1, 2]:
        top = 0.92 - r * (0.90 / nrows)

        # manual vertical tweaks
        if r == 0:   # 1st row: move a bit down
            top -= 0.03
        if r in [3, 4]:  # 4th & 5th rows: move a bit up
            top += 0.04

        height = 0.80 / nrows - 0.02
        cax = fig.add_axes([0.84, top - height, 0.015, height*0.85])
        cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
        cb.set_label(row["unit"], fontsize=15)


    # individual colorbars for rows except low/mid cloud rows
   # if mappable is not None and r not in [1, 2]:
   #     top = 0.92 - r * (0.90 / nrows)
   #     height = 0.80 / nrows - 0.02
   #     cax = fig.add_axes([0.84, top - height, 0.015, height])
   #     cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
   #     cb.set_label(row["unit"], fontsize=15)

# 1) shared colorbar for rows 2 & 3 (r=1,2)
if cloud_mappable is not None:
    height_one = 0.80 / nrows - 0.02
    top1 = (0.92 - 1 * (0.90 / nrows)) + 0.03
    bottom2 = (0.92 - 2 * (0.90 / nrows)) - height_one
    cax = fig.add_axes([0.84, bottom2, 0.015, (2 * height_one + 0.02)*0.85])
    cb = fig.colorbar(cloud_mappable, cax=cax, orientation="vertical")
    cb.set_label("%", fontsize=15)

outname = "dVariables.jpg"
plt.savefig(outname, dpi=dpi, bbox_inches="tight")
#plt.show()
plt.close(fig)
print(f"Saved {outname}")

