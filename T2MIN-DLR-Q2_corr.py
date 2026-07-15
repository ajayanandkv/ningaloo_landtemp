#!/usr/bin/env python3

import os
import numpy as np
import xarray as xr
import xskillscore as xs
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

matplotlib.rcParams["agg.path.chunksize"] = 10000

# =================== CONFIG ===================
daily_base = "/pvol/ajay/data2/ajay/WRF_MHW/wrf_outputs/derived_outputs/daily_mean"
run_cntl = "CNTL"
run_clim = "EXP-SSTClim"
domains = ["R3", "R5"]

year, month, mname = "2011", "03", "Mar"
dpi = 300
CMAP = "Spectral_r"
ALPHA = 0.05

# Crop
CROP = dict(south_north=slice(15, -5), west_east=slice(5, -5))

# Rotated pole projection
proj = ccrs.RotatedPole(pole_latitude=60.31, pole_longitude=147.63)

# Coastal mask
mask_file = "CoastalLandMask_N12.nc"
mask_var  = "coastal_land_mask_inbox"   # 1=keep, 0=mask out

# =================== HELPERS ===================
def month_days(y, m):
    import calendar
    nd = calendar.monthrange(int(y), int(m))[1]
    return [f"{y}-{m}-{d:02d}" for d in range(1, nd + 1)]

def get_lonlat_and_landmask(domain_ref="R5"):
    ref = (
        f"wrfout_d01_{year}-03-31_00:00:00"
    )
    ds = xr.open_dataset(ref)

    lat = ds["XLAT"].isel(Time=0, **CROP).squeeze()
    lon = ds["XLONG"].isel(Time=0, **CROP).squeeze()

    # Keep your original convention
    mrsos = ds["SMOIS"].isel(Time=0, soil_layers_stag=3, **CROP).squeeze()
    land_mask = xr.where(mrsos > 0.5, 0, 1)  # 1 = land

    ds.close()
    return lon, lat, land_mask

def load_coastal_mask():
    with xr.open_dataset(mask_file) as dm:
        m = dm[mask_var]
        if ("south_north" in m.dims) and ("west_east" in m.dims):
            m = m.isel(**CROP)
        return m.squeeze().astype("int8")

def _daily_file(domain, run, folder, datestr):
    return f"{daily_base}/{domain}/{run}/ensemble/{folder}/{folder}_d01_{datestr}_en.nc"

def open_daily_da(domain, run, folder, varname, datestr):
    f = _daily_file(domain, run, folder, datestr)
    if not os.path.exists(f):
        return None
    ds = xr.open_dataset(f)

    if varname in ds:
        name = varname
    else:
        name = next((k for k in ds.data_vars if k.lower() == varname.lower()), None)
    if name is None:
        ds.close()
        return None

    da = ds[name].isel(**CROP).squeeze().load()
    ds.close()
    return da

def daily_delta_stack_wrf(domain, folder, varname, days, scale=1.0):
    arr = []
    for d in days:
        a = open_daily_da(domain, run_cntl, folder, varname, d)
        b = open_daily_da(domain, run_clim, folder, varname, d)
        if (a is None) or (b is None):
            continue
        arr.append(((a - b) * scale).expand_dims(time=[np.datetime64(d)]))
    return xr.concat(arr, dim="time") if arr else None

def daily_delta_stack_cloud(domain, cloud_layer, days, scale=100.0):
    arr = []
    for d in days:
        a = open_daily_da(domain, run_cntl, "cloudfrac", "cloudfrac", d)
        b = open_daily_da(domain, run_clim, "cloudfrac", "cloudfrac", d)
        if (a is None) or (b is None):
            continue

        if "ncl0" not in a.dims:
            raise ValueError(f"{domain} {d}: expected 'ncl0' in cloudfrac dims, got {a.dims}")

        a = a.isel(ncl0=cloud_layer)
        b = b.isel(ncl0=cloud_layer)

        arr.append(((a - b) * scale).expand_dims(time=[np.datetime64(d)]))

    return xr.concat(arr, dim="time") if arr else None

def corr_and_p(a, b, dim="time"):
    a, b = xr.align(a, b, join="inner")
    r = xr.corr(a, b, dim=dim)
    p = xs.pearson_r_p_value(a, b, dim=dim)
    return r, p

def landmask(da, land_mask):
    if da is None:
        return None
    return da.where(land_mask == 1)

def coastal_mean_r(r_da, coast_mask):
    if r_da is None:
        return np.nan
    v = r_da.where(coast_mask == 1)
    return float(v.mean(skipna=True).values)

def plot_4panel(lon, lat, r_maps, p_maps, outname):
    titles = [
        r"corr($\Delta$T2MIN, $\Delta$DLR)",
        r"corr($\Delta$DLR, $\Delta$LowCF)",
        r"corr($\Delta$DLR, $\Delta$MidCF)",
        r"corr($\Delta$DLR, $\Delta$Q2)",
    ]

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), subplot_kw={"projection": proj})
    pmesh = None

    for i, ax in enumerate(axs):
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")

        r = r_maps[i]
        p = p_maps[i]

        if (r is None) or (p is None):
            ax.text(0.5, 0.5, "Missing", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=18)
            ax.set_title(titles[i], fontsize=24)
            continue

        pmesh = ax.pcolormesh(
            lon, lat, r,
            vmin=-1, vmax=1, cmap=CMAP,
            transform=ccrs.PlateCarree(), shading="auto"
        )

        sig = (p < ALPHA)
        ax.contourf(
            lon, lat, sig.astype(int),
            levels=[0.5, 1.5],
            hatches=["...."], colors="none",
            transform=ccrs.PlateCarree()
        )

        ax.set_title(titles[i], fontsize=24)
        ax.text(0.02, 0.95, f"({chr(97+i)})", transform=ax.transAxes,
                fontsize=22, va="top", ha="left")

    cax = fig.add_axes([0.30, 0.08, 0.40, 0.04])
    fig.colorbar(pmesh, cax=cax, orientation="horizontal", label="r")
    fig.subplots_adjust(wspace=0.1)
    plt.savefig(outname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outname}")

# =================== MAIN ===================
if __name__ == "__main__":
    days = month_days(year, month)
    lon, lat, land_mask = get_lonlat_and_landmask("R5")
    coast_mask = load_coastal_mask()

    outprefix = f"corr_dVar{mname}_{year}"

    r_by_dom = {}
    p_by_dom = {}

    for dom in domains:
        # Daily deltas
        d_dlr   = daily_delta_stack_wrf(dom, "GLW",   "GLW",   days, scale=1.0)   # DLR from GLW
        d_t2min = daily_delta_stack_wrf(dom, "T2MIN", "T2MIN", days, scale=1.0)
        d_q2    = daily_delta_stack_wrf(dom, "Q2",    "Q2",    days, scale=1.0)
        d_lowcf = daily_delta_stack_cloud(dom, cloud_layer=0, days=days, scale=100.0)
        d_midcf = daily_delta_stack_cloud(dom, cloud_layer=1, days=days, scale=100.0)

        if any(x is None for x in [d_dlr, d_t2min, d_q2, d_lowcf, d_midcf]):
            print(f"{dom}: one or more daily delta stacks missing (check files).")

        # Correlations (order as requested)
        r1, p1 = corr_and_p(d_t2min, d_dlr)     # dT2MIN vs dDLR
        r2, p2 = corr_and_p(d_dlr,   d_lowcf)   # dDLR vs dLowCF
        r3, p3 = corr_and_p(d_dlr,   d_midcf)   # dDLR vs dMidCF
        r4, p4 = corr_and_p(d_dlr,   d_q2)      # dDLR vs dQ2

        # Land mask for all
        r1, p1 = landmask(r1, land_mask), landmask(p1, land_mask)
        r2, p2 = landmask(r2, land_mask), landmask(p2, land_mask)
        r3, p3 = landmask(r3, land_mask), landmask(p3, land_mask)
        r4, p4 = landmask(r4, land_mask), landmask(p4, land_mask)

        r_by_dom[dom] = [r1, r2, r3, r4]
        p_by_dom[dom] = [p1, p2, p3, p4]

        # ---- coastal-mask mean r (print only) ----
        print(f"\n{dom} coastal-mask mean r (mask==1):")
        print(f"  r(ΔT2MIN,ΔDLR) = {coastal_mean_r(r1, coast_mask):.3f}")
        print(f"  r(ΔDLR,ΔLowCF) = {coastal_mean_r(r2, coast_mask):.3f}")
        print(f"  r(ΔDLR,ΔMidCF) = {coastal_mean_r(r3, coast_mask):.3f}")
        print(f"  r(ΔDLR,ΔQ2)    = {coastal_mean_r(r4, coast_mask):.3f}")

        plot_4panel(lon, lat, r_by_dom[dom], p_by_dom[dom], f"{outprefix}_{dom}.jpg")

    # -------- R3R5 mean (average r and p after aligning) --------
    if "R3" in r_by_dom and "R5" in r_by_dom:
        r_mean, p_mean = [], []
        for i in range(4):
            ra, rb = xr.align(r_by_dom["R3"][i], r_by_dom["R5"][i], join="inner")
            pa, pb = xr.align(p_by_dom["R3"][i], p_by_dom["R5"][i], join="inner")
            r_mean.append((ra + rb) / 2.0)
            p_mean.append((pa + pb) / 2.0)

        # ---- coastal-mask mean r for R3R5mean (print only) ----
        print("\nR3R5mean coastal-mask mean r (mask==1):")
        print(f"  r(ΔT2MIN,ΔDLR) = {coastal_mean_r(r_mean[0], coast_mask):.3f}")
        print(f"  r(ΔDLR,ΔLowCF) = {coastal_mean_r(r_mean[1], coast_mask):.3f}")
        print(f"  r(ΔDLR,ΔMidCF) = {coastal_mean_r(r_mean[2], coast_mask):.3f}")
        print(f"  r(ΔDLR,ΔQ2)    = {coastal_mean_r(r_mean[3], coast_mask):.3f}")

        plot_4panel(lon, lat, r_mean, p_mean, f"{outprefix}.jpg")

