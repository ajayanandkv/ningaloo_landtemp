#!/usr/bin/env python
"""
Reviewer 2 comment 2: "the timing of the peak".

The day/night maps give composite means, which cannot show WHEN in the
day anything peaks. This figure shows the shape of the March-mean diurnal
cycle for every day/night variable in one place, area-averaged over the
paper's coastal land band, and prints the peak hour and the peak-hour
SHIFT between CNTL and ClimSST -- which is the number the reviewer is
asking for.

    one panel per variable, 3 per row
    x        local hour, 00-23 AWST
    lines    CNTL (solid) and ClimSST (dashed)
    shading  spread across the 10 members (2 RCMs x 5 start dates)
    grey     night (sunset to sunrise)
    dotted   observed Perth sea-breeze onset, 14:00
    inset    the averaging band, drawn inside one panel

Averaging uses CoastalLandMask_N12.nc, the same band as the sea-breeze
and map figures, so every number in the paper refers to one region.
--band falls back to the old distance-based bands (offshore, coastal,
inland) when no mask file is given.

Lines are the mean over all 10 members rather than one per RCM: the two
RCMs sit close together here, and four lines per panel obscured the
CNTL-ClimSST difference, which is what the panel is for. --by-rcm
restores the per-RCM split, and the peak table prints both either way.

Variables are found automatically in whichever derived group holds them
(diurnal, energy or rain), so this works whether or not the extended
stage 1 was run; anything missing is reported and its panel dropped.
u_onshore is computed here from U10E/V10E and the coastline normal.

Usage:
  python fig_diurnal_cycle.py <coast_geometry.nc> [CoastalLandMask_N12.nc]
        [out.png] [--run CNTL] [--ref EXP-SSTClim] [--by-rcm]
        [--vars T2,HFX,LH,...] [--inset-panel 5]
"""
import argparse
import os
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import wrfplot as wp

# name, lo_km, hi_km, want_land -- only used when no mask file is given
BANDS = {
    "offshore": ("Offshore", -50.0, 0.0, False),
    "coastal": ("Coastal land", 0.0, 50.0, True),
    "inland": ("Inland", 50.0, 150.0, True),
}

# one colour per RUN now, not per RCM
COLORS = {"run": "#1a1a1a", "ref": "#2c7fb8"}
RCM_COLORS = {"R3": "#2c7fb8", "R5": "#d6604d"}

# Perth, March 2011: sunrise 06:06-06:29, sunset 19:04-18:26 AWST.
SUNRISE, SUNSET = 6.3, 18.7
# Masselink and Pattiaratchi (2001): typical sea-breeze onset at Perth.
MP_ONSET = 14.0

ONSHORE = "u_onshore"       # pseudo-variable, computed from U10E/V10E

# variable: (axis label, scale, draw a zero line)
SPECS = {
    ONSHORE:        ("Onshore wind (m s$^{-1}$)",        1.0, True),
    "T2":           ("2 m temperature (K)",              1.0, False),
    "PBLH":         ("PBL height (m)",                   1.0, False),
    "HFX":          ("Sensible heat flux (W m$^{-2}$)",  1.0, True),
    "LH":           ("Latent heat flux (W m$^{-2}$)",    1.0, True),
    "RNET":         ("Net radiation (W m$^{-2}$)",       1.0, True),
    "SW_CRE":       ("SW cloud effect (W m$^{-2}$)",     1.0, True),
    "PREC_ACC_TOT": ("Rainfall (mm h$^{-1}$)",           1.0, False),
    "GRDFLX":       ("Ground heat flux (W m$^{-2}$)",    1.0, True),
    "TSK":          ("Skin temperature (K)",             1.0, False),
    "Q2":           ("2 m mixing ratio (g kg$^{-1}$)", 1000.0, False),
    "SWDOWN":       ("Downward SW (W m$^{-2}$)",         1.0, False),
    "GLW":          ("Downward LW (W m$^{-2}$)",         1.0, False),
    "OLR":          ("Outgoing LW at TOA (W m$^{-2}$)",  1.0, False),
    "cloud_low":    ("Low cloud fraction",               1.0, False),
    "cloud_mid":    ("Mid cloud fraction",               1.0, False),
    "cloud_high":   ("High cloud fraction",              1.0, False),
}

# Panel titles: the variable name is for us, not for a reader.
TITLES = {
    ONSHORE: "Onshore wind",
    "T2": "2 m temperature",
    "PBLH": "PBL height",
    "HFX": "Sensible heat flux",
    "LH": "Latent heat flux",
    "RNET": "Net radiation",
    "SW_CRE": "SW cloud effect",
    "PREC_ACC_TOT": "Rainfall",
    "GRDFLX": "Ground heat flux",
    "TSK": "Skin temperature",
    "Q2": "2 m mixing ratio",
    "SWDOWN": "Downward shortwave",
    "GLW": "Downward longwave",
    "OLR": "Outgoing longwave",
    "cloud_low": "Low cloud (sfc-970 hPa)",
    "cloud_mid": "Mid cloud (970-800 hPa)",
    "cloud_high": "High cloud (800-450 hPa)",
}

# RNET and SW_CRE are left out: neither reviewer asked for them. R2.3 asks
# for net shortwave and net longwave as separate terms, which RNET bundles
# together, and SW_CRE is a cloud-attribution diagnostic rather than a
# budget term. Both are still in SPECS, so --vars can bring them back.
#
# Rainfall is left out too, and lives in its own figure with cloud: the
# two answer Reviewer 2 comment 2 together, and rainfall on a linear axis
# beside fluxes of 200 W m-2 is unreadable. Five panels also frees the
# sixth cell for the band map, which is better than an inset over data.
DEFAULT_VARS = [ONSHORE, "T2", "PBLH", "HFX", "LH"]

# Cloud and rainfall together: Reviewer 2 comment 2 asks whether cloud
# forms and rain falls at night, which is one question about two fields.
CLOUD_RAIN_VARS = ["cloud_low", "cloud_mid", "cloud_high", "PREC_ACC_TOT"]

# Cloud is stored with a low_mid_high axis rather than as separate
# variables, so these are pseudo-variables resolved to (variable, index).
CLOUD_BANDS = {"cloud_low": 0, "cloud_mid": 1, "cloud_high": 2}


def grid_shape(n, want_spare=False):
    """
    Panel layout that leaves as few blank cells as possible.

    want_spare asks for exactly one blank cell when the count would
    otherwise fill the grid, so the band map has somewhere to go without
    covering data. It only widens the grid, never adds a row.
    """
    if n <= 3:
        return (1, n + 1) if want_spare else (1, n)
    if n == 4:
        return (2, 3) if want_spare else (2, 2)
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    return int(np.ceil(n / 3)), 3


def load_mask(path, var="auto"):
    """The paper's coastal land band, cropped to match the figures."""
    with xr.open_dataset(path) as ds:
        if var == "auto":
            cands = [v for v in ds.data_vars if ds[v].ndim == 2]
            if not cands:
                raise SystemExit(f"ERROR: no 2-D variable in {path}")
            var = cands[0]
            if len(cands) > 1:
                print(f"note: {path} has {cands}; using '{var}'")
        m = ds[var].isel(**wp.CROP).values
    mask = np.isfinite(m) & (np.nan_to_num(m) > 0.5)
    print(f"coastal band '{var}': {mask.sum()} grid points after crop")
    return mask


def coast_normal(coast_file):
    with xr.open_dataset(coast_file) as c:
        oe = c["onshore_e"].isel(**wp.CROP).values.astype(np.float64)
        on = c["onshore_n"].isel(**wp.CROP).values.astype(np.float64)
    return oe, on


# filled in by find_cloud_var(): (group, variable name, band dimension)
_CLOUD = None


def find_cloud_var():
    """
    Locate an HOURLY cloud-fraction variable carrying a vertical band axis.

    The day/night cloud files (cloud_mon) hold only two aggregates per
    March, which cannot make a diurnal cycle, so this searches the hourly
    groups for any variable with a low_mid_high-style dimension rather
    than guessing a name. Returns None, with a clear message, when the
    hourly cloud has not been produced.
    """
    global _CLOUD
    if _CLOUD is not None:
        return _CLOUD if _CLOUD != "none" else None
    phy, run = wp.PHYS[0], wp.REF_RUN
    for g in ("diurnal", "cloud", "cloudfrac"):
        if g not in wp.ENS_TEMPLATES:
            continue
        p = wp.ens_path(g, phy, run)
        if not os.path.exists(p):
            continue
        with xr.open_dataset(p) as ds:
            for name, da in ds.data_vars.items():
                band = next((d for d in da.dims
                             if "low" in d or "band" in d or "lev" in d), None)
                if band is not None and da.sizes[band] >= 3 \
                        and any(d in da.dims for d in ("hour", "Time", "time")):
                    _CLOUD = (g, name, band)
                    print(f"  cloud: using '{name}' from group '{g}' "
                          f"(band axis '{band}', {da.sizes[band]} levels)")
                    return _CLOUD
    _CLOUD = "none"
    print("  cloud: no HOURLY cloud-fraction variable found. The cloud "
          "files hold day and night means only,")
    print("         which cannot make a diurnal cycle. Stage 1 needs to "
          "write hourly cloud fraction first.")
    return None


def member_band_cycle(var, phy, run, mask, oe, on, group):
    """Area-mean diurnal cycle per member: (member, hour)."""
    out = []
    for date in wp.DATES:
        if var == ONSHORE:
            ue = wp.member_cycle("diurnal", phy, run, date, "U10E").values
            vn = wp.member_cycle("diurnal", phy, run, date, "V10E").values
            arr = ue * oe[None] + vn * on[None]
        elif var in CLOUD_BANDS:
            g, name, band = _CLOUD
            da = wp.member_cycle(g, phy, run, date, name)
            arr = da.isel({band: CLOUD_BANDS[var]}).values
        else:
            scale = SPECS.get(var, ("", 1.0, False))[1]
            arr = wp.member_cycle(group, phy, run, date, var, scale).values
        out.append(wp.area_cycle(arr, mask))
    return np.stack(out)


def resolve(var):
    """Which group holds this variable, or None if it is unavailable."""
    if var == ONSHORE:
        try:
            wp.resolve_group("U10E", groups=["diurnal"])
            return "diurnal"
        except KeyError as e:
            print(f"  SKIP {var}: {e}")
            return None
    if var in CLOUD_BANDS:
        found = find_cloud_var()
        if found is None:
            print(f"  SKIP {var}")
            return None
        return found[0]
    try:
        return wp.resolve_group(var)
    except KeyError as e:
        print(f"  SKIP {var}: {e}")
        return None


def pooled(cycles, run):
    """All members of one run, both RCMs stacked: (10, hour)."""
    parts = [cycles[(phy, run)] for phy in wp.PHYS if (phy, run) in cycles]
    return np.concatenate(parts) if parts else None


def panel(ax, hours, cycles, runs, spec, title, by_rcm=False,
          show_legend=False):
    label, _, zeroline = spec

    if by_rcm:
        for phy in wp.PHYS:
            for run, ls in zip(runs, ["-", "--"]):
                arr = cycles.get((phy, run))
                if arr is None:
                    continue
                c = RCM_COLORS.get(phy, "0.4")
                ax.fill_between(hours, arr.min(axis=0), arr.max(axis=0),
                                color=c, alpha=0.10, lw=0)
                ax.plot(hours, arr.mean(axis=0), ls, color=c, lw=1.8,
                        label=f"{phy} {wp.rlab(run)}")
    else:
        for run, ls, key in zip(runs, ["-", "--"], ["run", "ref"]):
            arr = pooled(cycles, run)
            if arr is None:
                continue
            c = COLORS[key]
            ax.fill_between(hours, arr.min(axis=0), arr.max(axis=0),
                            color=c, alpha=0.14, lw=0)
            ax.plot(hours, arr.mean(axis=0), ls, color=c, lw=2.2,
                    label=wp.rlab(run))

    ax.axvspan(0, SUNRISE, color="0.90", zorder=0)
    ax.axvspan(SUNSET, 23, color="0.90", zorder=0)
    ax.axvline(MP_ONSET, color="0.45", ls=":", lw=1.2)
    if zeroline:
        ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xlabel("Hour (AWST)")
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show_legend:
        ax.legend(fontsize=8, frameon=False, loc="best")


def diff_panel(ax, hours, cycles, runs, spec, title):
    """
    The run minus the reference, hour by hour, member by member.

    pooled() stacks members in the same (RCM, start date) order for both
    runs, so subtracting the two arrays pairs each member with itself.
    The shading is therefore the spread of the DIFFERENCE, which is much
    tighter than the spread of either run and is the right thing to judge
    the change against. Filled circles mark hours at which every member
    agrees on the sign.
    """
    label, _, _ = spec
    run, ref = runs
    A, B = pooled(cycles, run), pooled(cycles, ref)
    if A is None or B is None or A.shape != B.shape:
        ax.axis("off")
        return
    d = A - B
    m = d.mean(axis=0)
    agree = (d > 0).all(axis=0) | (d < 0).all(axis=0)

    ax.fill_between(hours, d.min(axis=0), d.max(axis=0),
                    color=COLORS["run"], alpha=0.16, lw=0)
    ax.plot(hours, m, "-", color=COLORS["run"], lw=2.2)
    if agree.any():
        ax.plot(hours[agree], m[agree], "o", ms=4.5, color=COLORS["run"],
                label=f"all {d.shape[0]} members agree")
    ax.axhline(0, color="k", lw=0.9)

    ax.axvspan(0, SUNRISE, color="0.90", zorder=0)
    ax.axvspan(SUNSET, 23, color="0.90", zorder=0)
    ax.axvline(MP_ONSET, color="0.45", ls=":", lw=1.2)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xlabel("Hour (AWST)")
    ax.set_ylabel(f"$\\Delta$ {label}", fontsize=9)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if agree.any():
        ax.legend(fontsize=8, frameon=False, loc="best")


def diff_table(v, cycles, runs):
    """Hour-by-hour difference, so the text can quote it, not the picture."""
    run, ref = runs
    A, B = pooled(cycles, run), pooled(cycles, ref)
    if A is None or B is None or A.shape != B.shape:
        return
    d = A - B
    m, n = d.mean(axis=0), d.shape[0]
    agree = (d > 0).all(axis=0) | (d < 0).all(axis=0)
    hi, lo = int(np.argmax(m)), int(np.argmin(m))
    day = list(range(7, 19))
    night = [h for h in range(24) if h not in day]
    print(f"\n{v}: {wp.rlab(run)} - {wp.rlab(ref)}")
    print(f"  largest increase {m[hi]:+.4f} at {hi:02d}:00, "
          f"largest decrease {m[lo]:+.4f} at {lo:02d}:00")
    print(f"  day 07-18 mean {m[day].mean():+.4f}, "
          f"night 19-06 mean {m[night].mean():+.4f}")
    print(f"  every one of {n} members agrees on the sign at "
          f"{int(agree.sum())} of 24 hours")


def band_map(fig, ax, coast_file, mask, rect, title=None):
    """
    Map of the averaging band, drawn over the box of an existing panel.

    A matplotlib axes cannot change projection after creation, so the
    host panel is switched off and a cartopy axes is placed on its box
    instead. Pass rect = (0, 0, 1, 1) to fill an empty grid cell, or a
    smaller rectangle to sit inside a panel that holds data.

    Called AFTER tight_layout: a manually placed axes confuses it.
    """
    with xr.open_dataset(coast_file) as c:
        lat = c["XLAT"].isel(**wp.CROP).values
        lon = c["XLONG"].isel(**wp.CROP).values
        land = c["landmask"].isel(**wp.CROP).values
    bb = ax.get_position()
    x0, y0, w, h = rect
    pos = [bb.x0 + x0 * bb.width, bb.y0 + y0 * bb.height,
           w * bb.width, h * bb.height]
    iax = fig.add_axes(pos, projection=wp.ROTATED_POLE)
    tr = ccrs.PlateCarree()
    iax.pcolormesh(lon, lat, land, cmap="Greys", vmin=0, vmax=2.4,
                   shading="auto", transform=tr, zorder=0)
    iax.contourf(lon, lat, mask.astype(float), levels=[0.5, 1.5],
                 colors=["#6a1b7a"], transform=tr, zorder=1)
    iax.contour(lon, lat, land, [0.5], colors="k", linewidths=0.4,
                transform=tr, zorder=2)
    if title:
        iax.set_title(title, fontsize=10, weight="bold")
    for spine in iax.spines.values():
        spine.set_linewidth(0.6)
    return iax


def peak_table(var, cycles, runs):
    """
    Peak hour, peak value and amplitude per run, and the CNTL - ref shift.

    Sub-hourly peak timing is not available (the data are hourly means),
    so a parabolic fit through the three points around the maximum gives
    the finer estimate printed as 'fine', which is the usual way to
    report a timing shift smaller than the sampling interval.

    Both the pooled and the per-RCM numbers are printed whatever the
    figure draws, since the per-RCM spread is worth knowing even when it
    is not shown.
    """
    def stats(c):
        k = int(c.argmax())
        km, kp = (k - 1) % 24, (k + 1) % 24
        denom = c[km] - 2 * c[k] + c[kp]
        frac = 0.0 if denom == 0 else 0.5 * (c[km] - c[kp]) / denom
        frac = float(np.clip(frac, -0.5, 0.5))
        return dict(hour=k, fine=(k + frac) % 24, peak=float(c.max()),
                    trough_hour=int(c.argmin()),
                    amp=float(c.max() - c.min()))

    def shift_of(a, b):
        s = a["fine"] - b["fine"]
        if s > 12:
            s -= 24
        if s < -12:
            s += 24
        return s

    print(f"\n{var}")
    print(f"  {'group':<9s} {'run':<10s} {'peak':>6s} {'fine':>6s} "
          f"{'value':>10s} {'trough':>7s} {'amp':>10s}")

    groups = [("all", {r: pooled(cycles, r) for r in runs})]
    groups += [(phy, {r: cycles.get((phy, r)) for r in runs})
               for phy in wp.PHYS]

    for name, d in groups:
        vals = {}
        for run in runs:
            if d.get(run) is None:
                continue
            vals[run] = stats(d[run].mean(axis=0))
            v = vals[run]
            print(f"  {name:<9s} {wp.rlab(run):<10s} {v['hour']:>4d}h "
                  f"{v['fine']:>6.2f} {v['peak']:>10.3f} "
                  f"{v['trough_hour']:>6d}h {v['amp']:>10.3f}")
        if len(vals) == 2:
            a, b = vals[runs[0]], vals[runs[1]]
            print(f"  {name:<9s} {'peak shift':<10s} {shift_of(a, b):+.2f} h, "
                  f"amplitude change {a['amp'] - b['amp']:+.3f}")


def main(a):
    runs = [a.run, a.ref]

    if a.mask:
        mask = load_mask(a.mask, a.mask_var)
        band_name = "the coastal land band"
    else:
        if a.band not in BANDS:
            raise SystemExit(f"--band must be one of {sorted(BANDS)}")
        b = BANDS[a.band]
        print("band:")
        mask = wp.band_masks(a.coast, [b])[b[0]]
        band_name = f"{b[0].lower()} ({b[1]:+.0f} to {b[2]:+.0f} km)"
    if mask.sum() == 0:
        raise SystemExit("ERROR: the averaging band contains no grid points")
    oe, on = coast_normal(a.coast)

    wanted = [v.strip() for v in a.vars.split(",")] if a.vars else DEFAULT_VARS
    print("\nlocating variables:")
    found = []
    for v in wanted:
        g = resolve(v)
        if g is not None:
            print(f"  {v:<14s} -> {g}")
            found.append((v, g))
    if not found:
        raise SystemExit("ERROR: none of the requested variables were found")

    hours = np.arange(24)
    data = {}
    for v, g in found:
        data[v] = {}
        for phy in wp.PHYS:
            for run in runs:
                data[v][(phy, run)] = member_band_cycle(v, phy, run, mask,
                                                        oe, on, g)
        print(f"  read {v}")

    # with --diff each variable gets its curves panel and, next to it, a
    # panel of the member-wise difference
    cells = []
    for v, _ in found:
        cells.append((v, "abs"))
        if a.diff:
            cells.append((v, "diff"))

    nrow, ncol = grid_shape(len(cells), want_spare=not a.no_band)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow),
                            squeeze=False)

    for k, (v, kind) in enumerate(cells):
        ax = axs[k // ncol, k % ncol]
        spec = SPECS.get(v, (v, 1.0, False))
        name = TITLES.get(v, v)
        if kind == "abs":
            panel(ax, hours, data[v], runs, spec,
                  f"({chr(97 + k)}) {name}",
                  by_rcm=a.by_rcm, show_legend=(k == 0))
        else:
            diff_panel(ax, hours, data[v], runs, spec,
                       f"({chr(97 + k)}) {name}, "
                       f"{wp.rlab(a.run)} $-$ {wp.rlab(a.ref)}")
    spare = list(range(len(cells), nrow * ncol))
    for k in spare:
        axs[k // ncol, k % ncol].axis("off")

    fig.suptitle("March 2011 diurnal cycle over the coastal land band; "
                 "shading is the member spread, grey is night, "
                 "dotted line the observed Perth sea-breeze onset (14:00)",
                 fontsize=11.5, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96 if nrow > 1 else 0.92])

    # after tight_layout: a manually placed axes confuses it.
    # An empty grid cell takes the band map at full size; only when the
    # grid is full does it go inside a panel, over data.
    if not a.no_band:
        if spare:
            k = spare[0]
            host, rect = axs[k // ncol, k % ncol], (0.06, 0.06, 0.88, 0.80)
            title = f"({chr(97 + len(cells))}) Averaging band"
        else:
            k = min(max(a.inset_panel, 0), len(found) - 1)
            host, rect, title = axs[k // ncol, k % ncol], a.inset, None
            print(f"\nnote: no empty panel, so the band map is drawn inside "
                  f"panel ({chr(97 + k)}); move it with --inset-panel/--inset")
        band_map(fig, host, a.coast, mask, rect, title=title)

    wp.save(fig, a.out)

    print(f"\n=== peak timing, averaged over {band_name} ===")
    print("'fine' is a parabolic fit around the hourly maximum, so a shift "
          "smaller than 1 h can be quoted.")
    print("'all' pools both RCMs (10 members); R3 and R5 are listed below it.")
    for v, _ in found:
        peak_table(v, data[v], runs)

    if a.diff:
        print(f"\n=== {wp.rlab(a.run)} minus {wp.rlab(a.ref)}, "
              f"hour by hour ===")
        for v, _ in found:
            diff_table(v, data[v], runs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("mask", nargs="?", default=None,
                    help="CoastalLandMask_N12.nc; omit to use --band instead")
    ap.add_argument("out", nargs="?", default="figs/diurnal_cycle.png")
    ap.add_argument("--run", default="CNTL")
    ap.add_argument("--ref", default="EXP-SSTClim")
    ap.add_argument("--mask-var", default="auto")
    ap.add_argument("--band", default="coastal",
                    help="offshore, coastal or inland; only used when no "
                         "mask file is given")
    ap.add_argument("--diff", action="store_true",
                    help="add a panel per variable showing the member-wise "
                         "difference between the two runs, with the hours "
                         "at which every member agrees on the sign")
    ap.add_argument("--by-rcm", action="store_true",
                    help="draw R3 and R5 separately instead of one pooled "
                         "mean per run")
    ap.add_argument("--inset-panel", type=int, default=5,
                    help="which panel holds the band map, 0-based "
                         "(default 5, the last of six)")
    ap.add_argument("--inset", type=float, nargs=4,
                    default=[0.34, 0.46, 0.32, 0.50],
                    help="inset position inside that panel: x0 y0 w h, 0-1. "
                         "The default sits over the middle of the day, which "
                         "is where the night-peaking rainfall panel is empty")
    ap.add_argument("--cloud-rain", action="store_true",
                    help="plot low, mid and high cloud fraction with "
                         "rainfall instead of the wind and flux panels")
    ap.add_argument("--no-band", action="store_true",
                    help="do not draw the averaging-band map; the grid is "
                         "then packed tight")
    ap.add_argument("--vars", default=None,
                    help="comma-separated variable list, overrides the default")
    args = ap.parse_args()
    if args.cloud_rain and not args.vars:
        args.vars = ",".join(CLOUD_RAIN_VARS)
        if args.out == "figs/diurnal_cycle.png":
            args.out = "figs/diurnal_cloud_rain.png"
        # Four panels pack into 2x2 with nothing left over, and the band
        # is the same one the main diurnal figure already shows, so the
        # map would cost a blank cell for nothing. --no-band is implied;
        # pass --inset-panel to put it inside a panel if it is wanted.
        if "--inset-panel" not in sys.argv:
            args.no_band = True
    main(args)
