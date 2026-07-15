import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Setup
var = "SST"
path1 = "/data/ajay/MHW-THW/wrf_input/data/clm_wrflow"
path2 = "/data/ajay/MHW-THW/wrf_input/data/"
path3 = "/data/ajay/MHW-THW/wrf_input/data/org_wrflow"
clm_pct = ["20", "40", "60", "80"]  # Exclude 'Clim' here, handle it separately
fname = "wrflowinp_d01_2011-01-13"

# Load SST Obsr (100SST)
ds_obsr = xr.open_dataset(os.path.join(path3, fname))
sst_obsr = ds_obsr[var][456:, :, :] - 273.5
sst_obsr_mean = sst_obsr.mean(dim="Time")

# Load ClimSST
ds_clim = xr.open_dataset(os.path.join(path1, "EXP-SSTClim", fname))
sst_clim = ds_clim[var][456:, :, :] - 273.5
sst_clim_mean = sst_clim.mean(dim="Time")

# Load lat/lon and land mask
ds_ref = xr.open_dataset(os.path.join(path2, "wrfout_d01_2011-03-23_00:00:00"))
lat = ds_ref["XLAT"].isel(Time=0).values
lon = ds_ref["XLONG"].isel(Time=0).values
mrsos = ds_ref["SMOIS"].isel(Time=0, soil_layers_stag=3).squeeze()
wrflmask = xr.where(mrsos > 0.5, 0, 1)

# Masked Obsr & Clim
sst_obsr_masked = sst_obsr_mean.where(wrflmask == 0)
sst_clim_masked = sst_clim_mean.where(wrflmask == 0)

# Compute differences with ClimSST
diff_data = []
for pct in clm_pct:
    ds_exp = xr.open_dataset(os.path.join(path1, f"EXP-SST{pct}", fname))
    sst_exp = ds_exp[var][456:, :, :] - 273.5
    sst_exp_mean = sst_exp.mean(dim="Time")
    diff = sst_exp_mean - sst_clim_mean
    diff_data.append(diff.where(wrflmask == 0))

# Add Obsr - ClimSST
diff_data.append((sst_obsr_mean - sst_clim_mean).where(wrflmask == 0))
diff_titles = ["20SST - ClimSST", "40SST - ClimSST", "60SST - ClimSST", "80SST - ClimSST", "CNTL - ClimSST"]

# Plot
rotated_pole = ccrs.RotatedPole(pole_latitude=60.31, pole_longitude=147.63)
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), subplot_kw={'projection': rotated_pole})
fig.subplots_adjust(wspace=0.1, hspace=0.25)
fig.suptitle("", fontsize=16, weight='bold')

# --- Top row ---
for i, (data, title) in enumerate([(sst_obsr_masked, "CNTL (Observed)"), (sst_clim_masked, "ClimSST")]):
    ax = axs[0, i]
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    levels = np.linspace(20, 26, 13)
    p0 = ax.contourf(lon, lat, data, levels=levels, cmap='plasma_r', extend='both', transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, weight="bold")
    ax.text(0.02, 0.95, f"({chr(97+i)})", transform=ax.transAxes,
        fontsize=16, fontweight="bold", va="top", ha="left")

# Remove unused top row axes
for i in range(2, 5):
    fig.delaxes(axs[0, i])

# Colorbar for top row
cbar_ax0 = fig.add_axes([0.43, 0.57, 0.01, 0.28])
fig.colorbar(p0, cax=cbar_ax0, orientation='vertical').set_label("(°C)", rotation=0,  fontsize=14, weight='bold', labelpad=25)
cbar_ax0.yaxis.set_label_position('right')
cbar_ax0.yaxis.set_label_coords(1.5, -0.05)

# --- Bottom row: Differences ---
levels_diff = np.linspace(-3, 3, 21)
for i, (data, title) in enumerate(zip(diff_data, diff_titles)):
    ax = axs[1, i]
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    p = ax.contourf(lon, lat, data, levels=levels_diff, cmap='twilight_shifted', extend='both', transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, weight="bold")
    ax.text(0.02, 0.95, f"({chr(99+i)})", transform=ax.transAxes,
        fontsize=16, fontweight="bold", va="top", ha="left")

# Shared horizontal colorbar for bottom row
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.03])
cbar = fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
cbar.set_label("(°C)", fontsize=14, weight='bold', labelpad=5)
cbar.ax.tick_params(labelsize=12)

# Save
plt.savefig("sst_obsr_clim_diff.pdf", dpi=300, bbox_inches='tight')
plt.close()

