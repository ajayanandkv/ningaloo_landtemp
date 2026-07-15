import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

# ===== Settings =====
base_path   = "/data2/Jatin/For_Ajay"
model_vars  = ["tasmax", "tasmin"]   # AUS-18 model variables
obs_vars    = ["tmax", "tmin"]       # AGCD regridded variables
plot_titles = ["T2MAX", "T2MIN"]
years       = range(1980, 2021)      # 1980–2020 inclusive
MARCH_IDX   = 2                       # March (0-based)
domain_R3   = "R3"
domain_R5   = "R5"
file_save   = "bias_T2MAX-T2MIN_NarClim.jpg"

# Color/levels
cmap    = "RdBu_r"
vmin, vmax = -4.0, 4.0  # consistent color range across panels

# ===== Projection (NarClim style) =====
pole_latitude  = 60.31
pole_longitude = 141.38
rotated_pole = ccrs.RotatedPole(
    pole_latitude=pole_latitude,
    pole_longitude=pole_longitude,
    central_rotated_longitude=180
)
f_wrf = xr.open_dataset("/data/ajay/MHW-THW/wrf_output/data/mrsos_AUS-18_ECMWF-ERA5_evaluation_r1i1p1f1_R3_v1_day_20110101-20111231.nc")
#mrsos = f_wrf["mrsos"].isel(time=0).squeeze()
#wrflmask = xr.where(mrsos==1, 0, 1)
mrsos = f_wrf["mrsos"].isel(time=0).squeeze()
wrflmask = xr.where(mrsos.isnull(), 0, 1)  # 1 = valid, 0 = missing

def march_clim_model(varname, domain):
    """March climatology (1980–2020) for AUS-18 model var (Kelvin→°C). Returns (clim, lat, lon)."""
    chunks = []
    lat = lon = None
    for y in years:
        f = f"{base_path}/{varname}_AUS-18_ERA5_evaluation_r1i1p1f1_{domain}_v1_mon_{y}01-{y}12.nc"
        ds = xr.open_dataset(f)
        da = ds[varname].isel(time=MARCH_IDX) - 273.15  # K -> °C
        chunks.append(da)
        if lat is None:
            lat = ds["lat"]
            lon = ds["lon"]
    clim = xr.concat(chunks, dim="year").mean(dim="year")
    return clim, lat, lon

def march_clim_obs(varname):
    """March climatology (1980–2020) for regridded AGCD var (already °C)."""
    chunks = []
    for y in years:
        f = f"/data/ajay/MHW-THW/wrf_output/regrid/agcd_v1-0-1_{varname}_month_{y}.nc"
        ds = xr.open_dataset(f)
        da = ds[varname].isel(time=MARCH_IDX)
        chunks.append(da)
    clim = xr.concat(chunks, dim="year").mean(dim="year")
    return clim

# ===== Compute climatologies & bias =====
bias_maps = []
lon2d = lat2d = None

for mvar, ovar in zip(model_vars, obs_vars):
    # Model climatology for R3 & R5, then average
    clim_R3, lat_m, lon_m = march_clim_model(mvar, domain_R3)
    clim_R5, _, _         = march_clim_model(mvar, domain_R5)
    model_march_clim = (clim_R3 + clim_R5) / 2.0

    # Build lon/lat 2D once from model coords
    if lon2d is None or lat2d is None:
        lon_vals = lon_m.values
        lat_vals = lat_m.values
        if lon_vals.ndim == 1 and lat_vals.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        else:
            lon2d, lat2d = lon_vals, lat_vals

    # Obs climatology (assumes same grid/order as model regrid)
    obs_march_clim = march_clim_obs(ovar)

    # --- Align obs dims to model dims (name-only change) ---
    if "lat" in obs_march_clim.dims and "lon" in obs_march_clim.dims:
        obs_march_clim = obs_march_clim.rename({"lat": "rlat", "lon": "rlon"})
    # make sure order matches
    obs_march_clim = obs_march_clim.transpose("rlat", "rlon")

    # (optional but safe) carry over rlat/rlon coords from the model field
    obs_march_clim = obs_march_clim.assign_coords(
        rlat=model_march_clim["rlat"],
        rlon=model_march_clim["rlon"]
    )


    # Sanity checks
    assert model_march_clim.shape == obs_march_clim.shape, (
        f"Shape mismatch: model {model_march_clim.shape} vs obs {obs_march_clim.shape}."
    )
    assert list(model_march_clim.dims) == list(obs_march_clim.dims), (
        f"Dim mismatch: model {model_march_clim.dims} vs obs {obs_march_clim.dims}."
    )

    # Bias = model climatology − obs climatology
    bias = (model_march_clim - obs_march_clim).load()
    bias = xr.where(wrflmask == 1, bias, np.nan)
    bias_maps.append(bias)
   # bias_maps = mask(bias_maps,wrflmask,1)

# ===== Plot (NarClim map & gridline routines) =====
fig = plt.figure(figsize=(13, 8))
axes = [
    plt.subplot(1, 2, 1, projection=rotated_pole),
    plt.subplot(1, 2, 2, projection=rotated_pole),
]

def decorate_axis(ax):
    ax.coastlines(linewidth=1.0)
    ax.set_extent([90, 220, -30, 20], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linestyle='--', linewidth=0.5)
    gl.top_labels = gl.right_labels = False
    latlon_tick_spacing = 10
    gl.xlocator = mticker.FixedLocator(np.arange(90, 221, latlon_tick_spacing))
    gl.ylocator = mticker.FixedLocator(np.arange(-30, 21, latlon_tick_spacing))
    ax.set_xlim(-27, 13)
    ax.set_ylim(-18.2, 22.7)
    return gl

labels = ["(a)", "(b)"]
mappable = None

for i, ax in enumerate(axes):
    decorate_axis(ax)
    pc = ax.pcolormesh(
        lon2d, lat2d, bias_maps[i].values,
        cmap=cmap, vmin=vmin, vmax=vmax, shading='auto',
        transform=ccrs.PlateCarree(), edgecolors='face', antialiased=False
    )
    # Panel label
    ax.text(0.02, 0.96, labels[i], transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2.5))
    # Subplot title
    ax.set_title(plot_titles[i], fontsize=16, fontweight="bold", pad=6)
    mappable = pc

# === Adjust layout ===
fig.suptitle("[NarClim2.0−AGCD], March (1980–2020)",
             fontsize=15, fontweight="bold", y=0.97)

#fig.suptitle("(NarClim2.0−AGCD), March climatology (1980–2020)",
#             fontsize=15, fontweight="bold", y=0.90)

# Shared horizontal colorbar (closer to figure)
cbar = fig.colorbar(mappable, ax=axes, orientation='horizontal',
                    pad=0.06, fraction=0.04, shrink=0.95, aspect=40)
cbar.set_label("(°C)", fontsize=11)

#plt.subplots_adjust(top=0.90, bottom=0.10, wspace=0.08)
plt.subplots_adjust(top=2.2, bottom=0.2, wspace=0.08)
plt.savefig(file_save, dpi=300, bbox_inches="tight")
plt.close()

