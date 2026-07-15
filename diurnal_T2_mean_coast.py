#!/usr/bin/env python3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.rcParams["agg.path.chunksize"] = 10000

ncfile = "T2diff_diurnal_coast_March2011.nc"
outpng = "diurnal_T2diff_March2011_coast.pdf"
dpi = 300

mask_file = "CoastalLandMask_N12.nc"
mask_var  = "coastal_land_mask_inbox"
ref_file  = "/data2/ajay/WRF_MHW/wrf_outputs/compressed_outputs/R5/CNTL/2011-01-13/wrfout_d01_2011-03-31_00:00:00"

pole_latitude  = 60.31
pole_longitude = 147.63
rotated_pole = ccrs.RotatedPole(pole_latitude=pole_latitude, pole_longitude=pole_longitude)

ds = xr.open_dataset(ncfile)

R3   = ds["T2diff_R3"].values
R5   = ds["T2diff_R5"].values
MEAN = ds["T2diff_R3R5mean"].values
hours = ds["hour"].values

with xr.open_dataset(ref_file) as ds_ref:
    lat = ds_ref["XLAT"].isel(Time=0).squeeze()
    lon = ds_ref["XLONG"].isel(Time=0).squeeze()

with xr.open_dataset(mask_file) as dm:
    coastal_mask = dm[mask_var].astype("int8")

coastal_mask = coastal_mask.rename({k: v for k, v in zip(coastal_mask.dims, lat.dims)})
mask_to_plot = xr.where(coastal_mask == 1, 1.0, np.nan)

plt.figure(figsize=(8.8, 5))
ax = plt.gca()

def add_block(data, name):
    mu = np.nanmean(data, axis=0)
    sd = np.nanstd(data, axis=0)
    line, = ax.plot(hours, mu, linewidth=3, label=f"{name} mean")
    box = ax.fill_between(hours, mu - sd, mu + sd, alpha=0.18, label=f"{name} ±1 SD")
    return line, box

h1, b1 = add_block(R3, "R3")
h2, b2 = add_block(R5, "R5")
h3, b3 = add_block(MEAN, "R3+R5")

ax.set_xlabel("Hour (AWST)")
ax.set_ylabel("$\Delta$T2(°C)")
ax.set_title("March 2011, $\Delta$T2 diurnal cycle [CNTL − ClimSST]")
ax.set_xticks(hours)
ax.grid(alpha=0.3)

# Ordered legend: means first column, SD second column
#handles = [h1, b1, h2, b2, h3, b3]
#labels = ["R3 mean", "R3 ± 1 SD",
#          "R5 mean", "R5 ± 1 SD",
#          "R3+R5 mean", "R3+R5 ± 1 SD"]

#ax.legend(handles, labels, ncol=2, fontsize=8,
#          columnspacing=1.6, handlelength=2.5,
#          loc="best")

# Column-wise order (matplotlib default)
handles = [h1, h2, h3,   # first column: means
           b1, b2, b3]   # second column: ±1 SD

labels = ["R3 mean", "R5 mean", "R3+R5 mean",
          "R3 ± 1 SD", "R5 ± 1 SD", "R3+R5 ± 1 SD"]

ax.legend(handles, labels, ncol=2, fontsize=8, columnspacing=1.6, handlelength=2.5,loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=True)

# Inset coastal mask
axins = plt.gcf().add_axes([0.79, 0.14, 0.20, 0.20], projection=rotated_pole)
axins.add_feature(cfeature.OCEAN, facecolor="white")
axins.add_feature(cfeature.LAND, facecolor="lightgray")
axins.coastlines(linewidth=0.35)

axins.pcolormesh(lon, lat, mask_to_plot.values, shading="auto",
                 transform=ccrs.PlateCarree())

axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig(outpng, dpi=dpi, bbox_inches="tight")
plt.show()
plt.close()

ds.close()
print("Saved:", outpng)

