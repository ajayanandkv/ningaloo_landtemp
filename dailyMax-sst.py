import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load datasets
f = xr.open_dataset("sst_anom_2011_new.nc")
#f2 = xr.open_dataset("lsmask.nc")
f2 = xr.open_dataset("/data2/AGCD/awap_landmask.nc")
g = xr.open_dataset("dailymax_anom_2011.nc")

# Masking
v2 = f2['landmask']
v3 = v2.astype(float)

# Corrected time selection for February and March anomalies
#sst_feb = f['sst'].isel(time=1).where(v3 == 1)
#sst_mar = f['sst'].isel(time=2).where(v3 == 1)
sst_feb = f['sst'].isel(time=1)
sst_mar = f['sst'].isel(time=2)

# Align dimensions explicitly
mask = xr.DataArray(v3.values, dims=("lat", "lon"), coords={"lat": g['lat'], "lon": g['lon']})
tmax_feb = g['tmax'].isel(time=1).where(mask == 1)
tmax_mar = g['tmax'].isel(time=2).where(mask == 1)
print(g['tmax'].isel(time=1).shape)
print(v3.shape)

# Set up 1x2 plot
fig = plt.figure(figsize=(18, 9), dpi=300)

# Define levels and color map
levels = np.arange(-2.5, 2.6, 0.25)
cmap = 'coolwarm'

# Spatial plots for February and March
months = ['February 2011', 'March 2011']
datasets = [(sst_feb, tmax_feb), (sst_mar, tmax_mar)]

for i, ((sst, tmax), month) in enumerate(zip(datasets, months), start=1):
    ax = fig.add_subplot(1, 2, i, projection=ccrs.PlateCarree())

    ax.set_extent([104, 120, -40, -20], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)

    #contour1 = ax.contourf(tmax['lon'], tmax['lat'], tmax, levels=levels, cmap=cmap, extend='both', antialiased=True)
    #ax.contourf(sst['lon'], sst['lat'], sst, levels=levels, cmap=cmap, extend='both', antialiased=True)

    contour1 = ax.contourf(sst['lon'], sst['lat'], sst, levels=levels, cmap=cmap, extend='both', antialiased=True)
    ax.contourf(tmax['lon'], tmax['lat'], tmax, levels=levels, cmap=cmap, extend='both', antialiased=True)

    gl = ax.gridlines(draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}
    gl.ylocator = plt.MultipleLocator(4)
    ax.set_title(month, fontsize=22, pad=12)
    #ax.text(-0.05, 1.05, f"{chr(96 + i)})", transform=ax.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')
    ax.text(0.02, 0.95, f"({chr(96 + i)})", transform=ax.transAxes,
        fontsize=26, fontweight='bold', va='top', ha='left')

plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Add color bar
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.55])
cbar = fig.colorbar(contour1, cax=cbar_ax, orientation='vertical', shrink=0.2, anchor=(0.4, 0.435))
cbar.set_label('°C', fontsize=20)
cbar.ax.tick_params(labelsize=16)

plt.savefig("sst-tmax_feb_mar_2011.pdf", bbox_inches='tight', dpi=300)

