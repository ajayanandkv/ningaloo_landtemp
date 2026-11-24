#!/usr/bin/env python
"""
Shared helpers for plotting the derived WRF ensembles.

Projection, cropping, colour scale and panel style follow the existing
plotting routine so new figures match the ones already in the paper.

Rotated pole
------------
    pole_latitude  = POLE_LAT          = 60.31
    pole_longitude = 180 - STAND_LON   = 147.63

Data are drawn with pcolormesh using the 2-D XLAT/XLONG and
transform=PlateCarree, so cells land correctly and the rotated pole is
the display projection.
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 10000       # avoids GEOS topology errors

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import xarray as xr

DERIVED = "/pvol/ajay/data2/ajay/WRF_MHW/wrf_outputs/derived_outputs"

PHYS = ["R3", "R5"]
DATES = ["2011-01-13", "2011-01-14", "2011-01-15", "2011-01-16", "2011-01-17"]
REF_RUN = "EXP-SSTClim"

# What runs are called on figures. The directory name stays EXP-SSTClim;
# only the label changes.
RUN_LABEL = {"EXP-SSTClim": "ClimSST"}


def rlab(run):
    """Figure label for a run directory name."""
    return RUN_LABEL.get(run, run)

# SST runs in increasing perturbation order, with the labels used before
SST_RUNS = ["EXP-SST20", "EXP-SST40", "EXP-SST60", "EXP-SST80", "CNTL"]
SST_TITLES = ["20SST-ClimSST", "40SST-ClimSST", "60SST-ClimSST",
              "80SST-ClimSST", "CNTL-ClimSST"]

POLE_LAT = 60.31
POLE_LON = 147.63            # = 180 - STAND_LON
ROTATED_POLE = ccrs.RotatedPole(pole_latitude=POLE_LAT, pole_longitude=POLE_LON)

CROP = dict(south_north=slice(15, -5), west_east=slice(5, -5))
CROP_NP = (slice(15, -5), slice(5, -5))                  # for raw numpy arrays

CMAP = plt.cm.RdYlBu                                     # positive = blue

# ---------------------------------------------------------------------
# FIGURE STYLING. Change these here and every figure follows; there are
# no font sizes hard-coded inside the plotting functions below.
#
# The published figures use large bold labels and panels packed close
# together. Matplotlib's defaults are much smaller and leave a wide gap
# between rows, which on a 5 x 5 grid wastes most of the page.
# ---------------------------------------------------------------------
FS_COL = 20        # column headings across the top
FS_ROW = 20        # row labels down the left
FS_PANEL = 16      # the (a) (b) (c) letters inside each panel
FS_CBAR = 18       # colour bar label
FS_TICK = 16       # colour bar tick numbers
FS_SUP = 22        # suptitle, when there is one

# Inches per panel. The ratio should roughly match the map's own aspect;
# if it does not, cartopy keeps the map's shape and the leftover space
# inside each axes box becomes the white gap you see between rows.
PANEL_W = 4.0
PANEL_H = 3.6

# Gaps between panels, as a fraction of panel size. bbox_inches="tight"
# in save() already trims the OUTSIDE of the figure, so these two control
# all the white space that is left.
WSPACE = 0.05
HSPACE = 0.05

# The sizes above are tuned for the 5-column figure. A 2- or 3-column
# figure is physically smaller, so the same point size looks far bigger
# on the page. panel_grid therefore scales them by the grid width,
# relative to 5 columns, and remembers the scale so draw() and cbar()
# match. Pass fscale= to panel_grid to override.
FSCALE_REF_COLS = 5
FSCALE_MIN = 0.6
_FSCALE = 1.0


def _fs(base):
    """A styling size, scaled for the current figure."""
    return base * _FSCALE


def set_fscale(x):
    """Set the font scale by hand. panel_grid calls this for you."""
    global _FSCALE
    _FSCALE = float(x)

# The onshore component is only meaningful near a coast. Masselink and
# Pattiaratchi (2001) give an inland penetration of 90-150 km at Perth,
# so beyond this the projection is masked rather than plotted.
MAX_COAST_DIST_KM = 150.0

BAND_NAMES = {0: "Low cloud", 1: "Mid cloud", 2: "High cloud"}

# Day/night windows for the hourly fields, AWST. Chosen so that every hour
# is used, the two windows are balanced, and the 3-hourly cloud samples
# fall correctly inside them (08/11/14/17 in day, 20/23/02/05 in night).
# Perth in March: sunrise 06:06-06:29, sunset 19:04-18:26.
DAY_HOURS = list(range(7, 19))                  # 07..18
NIGHT_HOURS = [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]

# per-unit (not ensemble) monthly files, needed for member spread and
# sign-agreement masks
UNIT_TEMPLATES = {
    "cloud": "daily_mean/{phy}/{run}/{date}/cloudfrac_daynight_d01_2011-03.nc",
    "uv10": "daily_mean/{phy}/{run}/{date}/uv10_daynight_d01_2011-03.nc",
    "diurnal": "diurnal/{phy}/{run}/{date}/diurnal_d01_2011-03.nc",
    "rain": "rain/{phy}/{run}/{date}/rain_d01_2011-03.nc",
    "energy": "energy/{phy}/{run}/{date}/energy_d01_2011-03.nc",
}

ENS_TEMPLATES = {
    "cloud_mon": "mon_mean/{phy}/{run}/ensemble/cloudfrac_daynight_d01_2011-03-monmean-ensmean.nc",
    "uv10_mon": "mon_mean/{phy}/{run}/ensemble/uv10_daynight_d01_2011-03-monmean-ensmean.nc",
    "diurnal": "diurnal/{phy}/{run}/ensemble/diurnal_d01_2011-03-ensmean.nc",
    "seabreeze": "diurnal/{phy}/{run}/ensemble/seabreeze_metrics_d01_2011-03.nc",
    "rain": "rain/{phy}/{run}/ensemble/rain_d01_2011-03-ensmean.nc",
    "energy": "energy/{phy}/{run}/ensemble/energy_d01_2011-03-ensmean.nc",
    "sbday": "sbday/{phy}/{run}/ensemble/sbday_d01_2011-03-freq.nc",
}


_FEATURES_OK = None


def _features_ok():
    """
    Whether cartopy's Natural Earth features can be drawn.

    Cartopy fetches the shapefiles lazily during draw, so a missing cache
    fails inside savefig rather than here. Resolving the path once, up
    front, keeps the failure early and readable, and lets the figure
    still be produced on a node with no outbound network (the model land
    mask is drawn instead).
    """
    global _FEATURES_OK
    if _FEATURES_OK is None:
        try:
            from cartopy.io import shapereader
            shapereader.natural_earth(resolution="110m", category="physical",
                                      name="coastline")
            _FEATURES_OK = True
        except Exception as e:
            print(f"note: cartopy coastlines unavailable ({type(e).__name__}); "
                  f"drawing the model land mask instead.")
            print("      To cache them, run once on a node with internet:")
            print("        python -c \"from cartopy.io import shapereader as s; "
                  "[s.natural_earth(resolution=r, category='physical', "
                  "name='coastline') for r in ('110m','50m')]\"")
            _FEATURES_OK = False
    return _FEATURES_OK


def unit_path(group, phy, run, date):
    return os.path.join(DERIVED, UNIT_TEMPLATES[group].format(
        phy=phy, run=run, date=date))


def ens_path(group, phy, run):
    return os.path.join(DERIVED, ENS_TEMPLATES[group].format(phy=phy, run=run))


def member_monthly(group, phy, run, date, var, scale=1.0):
    """
    March mean of one variable for one member, cropped.

    The unit files hold a 31-day series, so the monthly mean is taken
    here rather than read from the ensemble file; that is what allows a
    sign-agreement mask across members.
    """
    p = unit_path(group, phy, run, date)
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: not found: {p}")
    with xr.open_dataset(p) as ds:
        da = ds[var]
        if "time" in da.dims:
            da = da.mean(dim="time")
        da = da.isel(**CROP).load() * scale
    return da


def _agg_hours(da, how):
    """Day, night and all-hours aggregate of an hour=24 field."""
    agg = (lambda x: x.sum(dim="hour")) if how == "sum" else \
          (lambda x: x.mean(dim="hour"))
    return {"day": agg(da.sel(hour=DAY_HOURS)),
            "night": agg(da.sel(hour=NIGHT_HOURS)),
            "all": agg(da)}


def member_cycle(group, phy, run, date, var, scale=1.0):
    """Full diurnal cycle (hour=24) of one variable for one member."""
    p = unit_path(group, phy, run, date)
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: not found: {p}")
    with xr.open_dataset(p) as ds:
        if var not in ds:
            raise SystemExit(f"ERROR: {var} not in {p}\n"
                             f"       file has: {sorted(ds.data_vars)}")
        da = ds[var].isel(**CROP).load() * scale
    if "hour" not in da.dims:
        raise SystemExit(f"ERROR: {var} in {p} has no 'hour' dimension")
    return da


def member_daynight(group, phy, run, date, var, how="sum", scale=1.0):
    """
    Day and night aggregate of an hourly (hour=24) variable, cropped.

    how="sum"  totals over the window, for rainfall, giving mm per window
               per day
    how="mean" averages over the window, for winds, fluxes and state
    """
    return _agg_hours(member_cycle(group, phy, run, date, var, scale), how)


def stack_anomaly_daynight(group, var, run, when, how="sum", phys=None,
                           dates=None, ref=REF_RUN, scale=1.0):
    """Member-wise day or night anomaly from the hourly files."""
    phys = phys or PHYS
    dates = dates or DATES
    out = []
    for phy in phys:
        for date in dates:
            a = member_daynight(group, phy, run, date, var, how, scale)[when]
            b = member_daynight(group, phy, ref, date, var, how, scale)[when]
            out.append(a - b)
    return xr.concat(out, dim="member")


def stack_anomaly(group, var, run, phys=None, dates=None, ref=REF_RUN,
                  scale=1.0):
    """
    Every member's March-mean anomaly (run - ref), stacked on 'member'.

    Members are the product of phys and dates, so with both RCMs and five
    initialisation dates this is 10 members -- the same pooling used in
    the existing figures.
    """
    phys = phys or PHYS
    dates = dates or DATES
    out = []
    for phy in phys:
        for date in dates:
            a = member_monthly(group, phy, run, date, var, scale)
            b = member_monthly(group, phy, ref, date, var, scale)
            out.append(a - b)
    return xr.concat(out, dim="member")


def agreement_mask(stack, fraction=0.8):
    """True where at least `fraction` of members agree on the sign."""
    n = stack.sizes["member"]
    thresh = int(np.ceil(fraction * n))
    signs = np.sign(stack)
    return (((signs > 0).sum(dim="member") >= thresh) |
            ((signs < 0).sum(dim="member") >= thresh))


def ens_field(group, phy, run, var, scale=1.0):
    """One variable from an ensemble-mean file, cropped."""
    p = ens_path(group, phy, run)
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: not found: {p}\n"
                         f"       has the stage-2 ensemble for '{group}' been run?")
    with xr.open_dataset(p) as ds:
        if var not in ds:
            raise SystemExit(f"ERROR: {var} not in {p}\n"
                             f"       file has: {sorted(ds.data_vars)}")
        return (ds[var].isel(**CROP).load() * scale)


def ens_field_mean(group, phys, run, var, scale=1.0):
    """Ensemble field averaged over several RCMs."""
    fields = [ens_field(group, p, run, var, scale) for p in phys]
    return sum(fields) / len(fields)


def ens_daynight(group, phy, run, var, how="mean", scale=1.0):
    """Day and night aggregate from an ensemble-mean hourly file."""
    da = ens_field(group, phy, run, var, scale)
    if "hour" not in da.dims:
        raise SystemExit(f"ERROR: {var} in the '{group}' ensemble has no "
                         f"'hour' dimension")
    return _agg_hours(da, how)


def ens_daynight_mean(group, phys, run, var, how="mean", scale=1.0):
    """Day/night aggregate averaged over several RCMs."""
    parts = [ens_daynight(group, p, run, var, how, scale) for p in phys]
    return {k: sum(d[k] for d in parts) / len(parts)
            for k in ("day", "night", "all")}


# Which derived group holds a given variable. The groups overlap (PBLH is
# in both 'diurnal' and 'energy', for instance) and which variables exist
# depends on which stage-1 jobs were run, so this is resolved by looking
# at the files rather than hard-coded.
GROUP_SEARCH = ["diurnal", "energy", "rain"]
_GROUP_CACHE = {}


def resolve_group(var, phys=None, run=None, groups=None):
    phy = (phys or PHYS)[0]
    run = run or REF_RUN
    if var in _GROUP_CACHE:
        return _GROUP_CACHE[var]
    looked = []
    for g in (groups or GROUP_SEARCH):
        p = ens_path(g, phy, run)
        if not os.path.exists(p):
            looked.append(f"{g}: file missing")
            continue
        with xr.open_dataset(p) as ds:
            if var in ds:
                _GROUP_CACHE[var] = g
                return g
            looked.append(f"{g}: present but no '{var}'")
    raise KeyError(f"{var} not found in any group ({'; '.join(looked)})")


def band_masks(coast_file, specs):
    """
    Boolean masks for distance-to-coast bands.

    specs: list of (name, lo_km, hi_km, want_land) where want_land is
    True for land only, False for water only, None for either.
    Bands are lo < d <= hi, so they do not overlap.
    """
    with xr.open_dataset(coast_file) as c:
        dist = c["dist_coast_km"].isel(**CROP).values
        land = c["landmask"].isel(**CROP).values > 0.5
        valid = (c["coast_valid"].isel(**CROP).values.astype(bool)
                 if "coast_valid" in c else np.ones(dist.shape, bool))
    out = {}
    for name, lo, hi, want_land in specs:
        m = (dist > lo) & (dist <= hi) & valid
        if want_land is True:
            m &= land
        elif want_land is False:
            m &= ~land
        out[name] = m
        print(f"  {name:<9s} {lo:+6.0f} to {hi:+6.0f} km: {m.sum():6d} points")
    return out


def area_cycle(arr, mask):
    """Area mean over `mask` for each hour of an (hour, y, x) array."""
    a = np.asarray(arr, dtype=np.float64)
    return np.nanmean(np.where(mask[None], a, np.nan), axis=(1, 2))


def onshore(coast_file, u_east, v_north, max_dist_km=MAX_COAST_DIST_KM):
    """
    Project earth-relative winds onto the coastline normal.

    Returns (onshore, alongshore). Positive onshore is from sea to land.

    The result is NaN wherever the coastline direction is undefined --
    far from any coast the smoothed land mask is flat, so its gradient is
    rounding noise and the projection would be pure artefact. It is also
    NaN beyond max_dist_km from the coast, since the sea breeze does not
    reach there anyway (Masselink and Pattiaratchi 2001 give an inland
    penetration of 90-150 km at Perth).
    """
    with xr.open_dataset(coast_file) as c:
        oe = c["onshore_e"].isel(**CROP).values.astype(np.float64)
        on = c["onshore_n"].isel(**CROP).values.astype(np.float64)
        dist = (c["dist_coast_km"].isel(**CROP).values
                if "dist_coast_km" in c else None)
        valid = (c["coast_valid"].isel(**CROP).values.astype(bool)
                 if "coast_valid" in c else np.isfinite(oe))

    u = np.asarray(u_east, dtype=np.float64)
    v = np.asarray(v_north, dtype=np.float64)
    if u.shape != oe.shape:
        raise SystemExit(f"ERROR: wind grid {u.shape} != coast grid {oe.shape}")

    keep = valid & np.isfinite(oe) & np.isfinite(on)
    if dist is not None and max_dist_km is not None:
        keep &= np.abs(dist) <= max_dist_km

    on_comp = np.where(keep, u * oe + v * on, np.nan)
    al_comp = np.where(keep, -u * on + v * oe, np.nan)
    return on_comp, al_comp


def grid(coast_file):
    """Cropped land mask and 2-D lat/lon from the coast-geometry file."""
    with xr.open_dataset(coast_file) as c:
        land = c["landmask"].isel(**CROP).values
        lat = c["XLAT"].isel(**CROP).values
        lon = c["XLONG"].isel(**CROP).values
    return land, lat, lon


def panel_grid(nrows, ncols, figsize=None, row_labels=None, col_labels=None,
               suptitle=None, fscale=None):
    """Axes grid on the rotated pole, styled as in the existing figures."""
    set_fscale(max(FSCALE_MIN, min(1.0, ncols / FSCALE_REF_COLS))
               if fscale is None else fscale)
    if figsize is None:
        # only reserve the strip for a suptitle when there is one
        # The strip has to clear the column labels as well, or a short
        # one-row figure puts the suptitle straight on top of them.
        strip = 0.0 if not suptitle else (1.15 if col_labels else 0.7)
        figsize = (PANEL_W * ncols, PANEL_H * nrows + strip)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize,
                            subplot_kw={"projection": ROTATED_POLE})
    axs = np.atleast_2d(axs)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)

    for ax in axs.ravel():
        if _features_ok():
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor="lightgray")

    if col_labels is not None:
        for j, lab in enumerate(col_labels):
            axs[0, j].set_title(lab, fontsize=_fs(FS_COL), weight="bold")
    if row_labels is not None:
        for i, lab in enumerate(row_labels):
            axs[i, 0].text(-0.08, 0.5, lab, transform=axs[i, 0].transAxes,
                           rotation=90, va="center", ha="center",
                           fontsize=_fs(FS_ROW), weight="bold")
    if suptitle:
        # Reserve the strip in INCHES and place the title inside it.
        # Growing the figure alone does not work: subplots_adjust keeps
        # the axes at a fixed FRACTION of the height, so the panels grow
        # with the figure and the column labels rise to meet the title.
        figh = fig.get_size_inches()[1]
        fig.subplots_adjust(top=1.0 - (0.95 if col_labels else 0.55) / figh)
        fig.suptitle(suptitle, fontsize=_fs(FS_SUP), weight="bold",
                     y=1.0 - 0.30 / figh)
    return fig, axs


def draw(ax, lon, lat, field, vmin, vmax, cmap=CMAP, label=None,
         stipple=None, land=None, label_pos="left", label_halo=True):
    p = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="auto",
                      vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    if land is not None and not _features_ok():
        ax.contour(lon, lat, land, levels=[0.5], colors="k", linewidths=0.7,
                   transform=ccrs.PlateCarree())
    if stipple is not None:
        ax.contourf(lon, lat, stipple.astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["...."],
                    transform=ccrs.PlateCarree())
    if label is not None:
        x, ha = (0.98, "right") if label_pos == "right" else (0.02, "left")
        t = ax.text(x, 0.95, f"({label})", transform=ax.transAxes,
                    fontsize=_fs(FS_PANEL), fontweight="bold",
                    va="top", ha=ha, zorder=7)
        if label_halo:
            # A thin white outline. Invisible over pale shading, and the
            # difference between a readable and an unreadable label over
            # the dark end of a diverging colourmap.
            t.set_path_effects([pe.withStroke(linewidth=2.6,
                                              foreground="white")])
    return p


def quiver(ax, lon, lat, u, v, step=18, scale=None, key=None,
           key_label=None, color="k", width=0.0058,
           headwidth=5.0, headlength=6.5, headaxislength=5.5,
           key_xy=(0.78, -0.07), key_box=False):
    """
    Wind vectors, subsampled for legibility.

    u and v must be EARTH-relative (east, north) because the vectors are
    transformed from PlateCarree; grid-relative components would point
    the wrong way on this rotated grid.

    Head sizes answer Reviewer 2 comment 5, which asked for larger
    arrowheads on Fig. 10. They are set here rather than in one figure so
    that every wind panel in the paper matches. `key` draws the reference
    arrow with its magnitude, which the same comment also asked for; pass
    it on every quiver, not just the first.
    """
    sl = (slice(None, None, step), slice(None, None, step))
    # scale_units="width" only when a scale is given: the arrow length is
    # then magnitude/scale as a fraction of the axes width, which is the
    # same in every panel. Left to autoscale, matplotlib normalises each
    # axes to ITS OWN vectors, so the same wind speed is drawn at a
    # different length in each panel and a row of panels cannot be
    # compared. Always pass a scale on a multi-panel figure.
    kw = {} if scale is None else {"scale_units": "width"}
    q = ax.quiver(np.asarray(lon)[sl], np.asarray(lat)[sl],
                  np.asarray(u)[sl], np.asarray(v)[sl],
                  transform=ccrs.PlateCarree(), color=color,
                  scale=scale, width=width, headwidth=headwidth,
                  headlength=headlength, headaxislength=headaxislength,
                  pivot="middle", **kw)
    if key is not None:
        x, y = key_xy
        if key_box:
            # The key sits inside a panel, over the shading, so it needs
            # something behind it to stay readable.
            ax.add_patch(Rectangle((x - 0.10, y - 0.060), 0.45, 0.130,
                                   transform=ax.transAxes, zorder=8,
                                   facecolor="white", alpha=1.0,
                                   edgecolor="0.5", linewidth=0.6))
        k = ax.quiverkey(q, x, y, key,
                         key_label or f"{key:g} m s$^{{-1}}$",
                         labelpos="E", coordinates="axes",
                         fontproperties={"size": 9.5})
        k.set_zorder(9)
    return q


def hcolorbar_span(fig, mappable, label, x0, x1, y=0.06, h=0.022):
    """Horizontal colourbar spanning part of the figure width."""
    cax = fig.add_axes([x0, y, x1 - x0, h])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=_fs(FS_CBAR), weight="bold")
    cb.ax.tick_params(labelsize=_fs(FS_TICK))
    return cb


def hcolorbar(fig, mappable, label, rect=(0.3, 0.06, 0.4, 0.025)):
    cax = fig.add_axes(list(rect))
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=_fs(FS_CBAR), weight="bold")
    cb.ax.tick_params(labelsize=_fs(FS_TICK))
    return cb


def cbar(fig, handle, ax, label, orientation="vertical", **kw):
    """
    Colour bar with the sizes set by FS_CBAR and FS_TICK above.

    Every figure should go through this rather than calling
    fig.colorbar directly, otherwise the tick numbers stay at the
    matplotlib default and look tiny beside the bold panel labels.
    """
    cb = fig.colorbar(handle, ax=ax, orientation=orientation, **kw)
    cb.set_label(label, fontsize=_fs(FS_CBAR), weight="bold")
    cb.ax.tick_params(labelsize=_fs(FS_TICK))
    return cb


def save(fig, out_png, dpi=300, also_pdf=True):
    d = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(d, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print("wrote", out_png)
    if also_pdf:
        pdf = os.path.splitext(out_png)[0] + ".pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print("wrote", pdf)
    plt.close(fig)
