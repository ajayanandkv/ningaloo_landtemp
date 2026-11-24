#!/usr/bin/env python
"""
Is the T2MAX increase really a residual of the nighttime warming?

The manuscript currently argues this by elimination: the CNTL - ClimSST
temperature difference is largest at night, decays through the morning,
and is at its smallest when T2MAX occurs, and no daytime process was
found that would warm the surface. A reviewer can reasonably answer that
a small difference at 2 pm does not show where that difference came
from. These are two direct tests instead.

    (a) inherited against generated
        Split the diurnal cycle of the T2 difference at the handover
        hour -- the hour the reference run reaches its daily minimum,
        which is when the night stops and the day starts.

            inherited  = dT2 at the handover hour, a constant
            generated  = dT2(t) - inherited

        If 'generated' stays at or below zero all day, no daytime
        process is adding heat and the afternoon difference is purely
        what survived the night. If it climbs after mid-morning,
        something daytime IS contributing and the paper has to name it.

    (b) does last night predict this afternoon?
        T2MIN occurs near dawn and T2MAX that same afternoon, so the two
        pair naturally within a local day. Regress the T2MAX difference
        on the T2MIN difference across every member and day.

            slope  = fraction of the night's warmth still present at T2MAX
            r^2    = how much of the T2MAX change it accounts for

        High r^2 with a slope below 1 is the residual signature in two
        numbers. Low r^2 means something else sets T2MAX.

    (c) where the carry-over is strong
        The same slope fitted at every land cell. This needs no
        sea-breeze classification, so it covers the whole land area.

Reads only derived output, so it runs on the plotting system.

Usage:
  python fig_residual_warming.py <coast_geometry.nc> <CoastalLandMask_N12.nc>
        [--run CNTL] [--ref EXP-SSTClim] [--outdir figs]
"""
import argparse
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import wrfplot as wp

TMAX_UNIT = "tmax/{phy}/{run}/{date}/tmax_d01_2011-03.nc"
TNAME, TNNAME = "T2MAX", "T2MIN"

SUNRISE, SUNSET = 6.3, 18.7      # Perth, March 2011
NICE_STEPS = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0)


def load_mask(path, var="auto"):
    with xr.open_dataset(path) as ds:
        if var == "auto":
            var = [v for v in ds.data_vars if ds[v].ndim == 2][0]
        m = ds[var].isel(**wp.CROP).values
    band = np.isfinite(m) & (np.nan_to_num(m) > 0.5)
    print(f"coastal band '{var}': {band.sum()} grid points after crop")
    return band


def band_cycle(run, mask):
    """Band-mean hourly T2 for every member: (member, 24)."""
    out = []
    for phy in wp.PHYS:
        for date in wp.DATES:
            arr = wp.member_cycle("diurnal", phy, run, date, "T2").values
            out.append(wp.area_cycle(arr, mask))
    return np.stack(out)


def daily_extremes(run, a):
    """TMAX and TMIN per member, day and cell."""
    tx, tn, days = [], [], None
    for phy in wp.PHYS:
        for date in wp.DATES:
            p = os.path.join(wp.DERIVED, a.tmax_template.format(
                phy=phy, run=run, date=date))
            if not os.path.exists(p):
                raise SystemExit(
                    f"ERROR: not found: {p}\n"
                    f"       run submit_sbday_s4, or point --tmax-template "
                    f"and\n       --tmax-var / --tmin-var at an existing "
                    f"daily file.")
            with xr.open_dataset(p) as ds:
                for v in (a.tmax_var, a.tmin_var):
                    if v not in ds:
                        raise SystemExit(
                            f"ERROR: '{v}' not in {p}\n"
                            f"       file has: {sorted(ds.data_vars)}")
                d = [str(x)[:10] for x in ds["day"].values] \
                    if "day" in ds.coords else None
                tx.append(ds[a.tmax_var].isel(**wp.CROP).values)
                tn.append(ds[a.tmin_var].isel(**wp.CROP).values)
            if days is None:
                days = d
    return np.stack(tx), np.stack(tn), days


def fit_cellwise(x, y):
    """
    Least-squares slope and r at every cell, over the pooled samples.

    x and y are (sample, y, x). Done with sums rather than a loop over
    cells, which would be tens of thousands of polyfit calls.
    """
    n = x.shape[0]
    xb, yb = x.mean(axis=0), y.mean(axis=0)
    dx, dy = x - xb, y - yb
    sxy = (dx * dy).sum(axis=0)
    sxx = (dx * dx).sum(axis=0)
    syy = (dy * dy).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(sxx > 0, sxy / sxx, np.nan)
        r = np.where((sxx > 0) & (syy > 0), sxy / np.sqrt(sxx * syy), np.nan)
    return slope, r, n


def nice_levels(field, what, pct=98, target=8, hard=None):
    v = np.asarray(field)
    f = v[np.isfinite(v)]
    if f.size == 0:
        return 1.0
    lo, hi = np.percentile(f, [100 - pct, pct])
    raw = hard if hard is not None else max(abs(lo), abs(hi))
    step = next((s for s in NICE_STEPS if raw / s <= target), NICE_STEPS[-1])
    lim = np.ceil(raw / step) * step
    print(f"  {what:<28s} data {lo:+.2f} to {hi:+.2f}, full {f.min():+.2f} "
          f"to {f.max():+.2f}  ->  scale +/-{lim:g}")
    return lim


def main(a):
    band = load_mask(a.mask, a.mask_var)
    land, lat, lon = wp.grid(a.coast)
    hours = np.arange(24)

    # ---------------------------------------------------- (a) decomposition
    print("\nreading hourly T2")
    c_r, c_f = band_cycle(a.run, band), band_cycle(a.ref, band)
    d_cyc = c_r - c_f                       # (member, 24)
    mu = d_cyc.mean(axis=0)

    # the handover hour: when the reference run is at its coldest, i.e.
    # the moment the night stops and the day's heating begins
    h0 = int(c_f.mean(axis=0).argmin())
    inherited = mu[h0]
    generated = mu - inherited
    hmax = int(c_f.mean(axis=0).argmax())

    print(f"\n--- (a) inherited against generated ---")
    print(f"  handover hour (coldest hour of {wp.rlab(a.ref)}): {h0:02d} AWST")
    print(f"  hour of the daily maximum:                      {hmax:02d} AWST")
    print(f"  dT2 at the handover hour (inherited) {inherited:+.3f} degC")
    print(f"  dT2 at the hour of the maximum       {mu[hmax]:+.3f} degC")
    print(f"  generated during the day             "
          f"{generated[hmax]:+.3f} degC")
    # The DECISION comes from the hour-to-hour change, not from the level.
    # Under pure decay the cumulative 'generated' is negative all day even
    # when a daytime source is present -- the source only makes it less
    # negative -- so a test on the level cannot tell the two apart. The
    # rate can: decay alone never turns the difference upward.
    tend = np.diff(mu)
    win = list(range(h0, max(h0 + 1, hmax)))
    rates = [tend[t] for t in win]
    peak = max(rates)
    hpeak = win[int(np.argmax(rates))]
    print(f"  cumulative 'generated' by the hour of the max "
          f"{generated[hmax]:+.3f} degC")
    print(f"  largest hourly RISE between {h0:02d} and {hmax:02d} AWST: "
          f"{peak:+.4f} degC/h at {hpeak:02d} AWST")
    if peak <= a.rate_tol:
        print(f"  -> the difference only ever decays between the handover "
              f"and the maximum.\n     Nothing is added during the day, so "
              f"the afternoon difference is what\n     survived the night. "
              f"That is the manuscript's claim, now measured.")
    else:
        print(f"  -> the difference RISES during the day, so a daytime "
              f"process is adding\n     heat. 'Residual of the night' is "
              f"not the whole story and the paper\n     has to name what "
              f"else is acting (the later sea breeze is a candidate).")

    # ------------------------------------------------------- (b), (c) pairs
    print("\nreading daily extremes")
    tx_r, tn_r, days = daily_extremes(a.run, a)
    tx_f, tn_f, _ = daily_extremes(a.ref, a)
    dTx, dTn = tx_r - tx_f, tn_r - tn_f      # (member, day, y, x)

    # Pool the 5 start dates before fitting. The members are stored
    # RCM-major, phy x date, so a reshape puts the dates on their own
    # axis. Averaging them first is the honest default: the five runs of
    # one RCM on one day are not five independent observations, they are
    # five noisy copies of the same day, and leaving them separate
    # inflates the sample count without adding information.
    nphy, ndate = len(wp.PHYS), len(wp.DATES)
    Xr = dTn.reshape(nphy, ndate, *dTn.shape[1:])
    Yr = dTx.reshape(nphy, ndate, *dTx.shape[1:])
    if a.pool == "member":
        Xp, Yp = dTn, dTx                         # 10 x 30
        plabel = f"{nphy * ndate} members x {dTn.shape[1]} days"
    elif a.pool == "init":
        Xp = Xr.mean(axis=1).reshape(nphy, *dTn.shape[1:])
        Yp = Yr.mean(axis=1).reshape(nphy, *dTx.shape[1:])
        plabel = f"{nphy} RCMs x {dTn.shape[1]} days (start dates averaged)"
    else:                                          # all
        Xp = Xr.mean(axis=(0, 1))[None]
        Yp = Yr.mean(axis=(0, 1))[None]
        plabel = f"{dTn.shape[1]} days (all 10 members averaged)"

    # T2MIN falls near dawn and T2MAX that same afternoon, so the pair
    # sits inside one local day: no lag is needed.
    x = np.array([np.nanmean(d[band]) for m in Xp for d in m])
    y = np.array([np.nanmean(d[band]) for m in Yp for d in m])
    grp = np.concatenate([[i] * Xp.shape[1] for i in range(Xp.shape[0])])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, grp = x[ok], y[ok], grp[ok]
    slope, intercept = np.polyfit(x, y, 1)
    rr = np.corrcoef(x, y)[0, 1]

    print(f"\n--- (b) does last night predict this afternoon? ---")
    print(f"  pooling: {plabel}")
    print(f"  {x.size} points")
    print(f"  mean d{TNNAME} {x.mean():+.3f} degC, "
          f"mean d{TNAME} {y.mean():+.3f} degC")
    print(f"  slope      {slope:+.3f}   "
          f"(fraction of the night's warmth left at {TNAME})")
    print(f"  intercept  {intercept:+.3f} degC")
    print(f"  r          {rr:+.3f}      r2 {rr ** 2:.3f}")
    print(f"  explained by the night: {100 * rr ** 2:.0f}% of the "
          f"{TNAME} variance")
    if rr ** 2 >= 0.4 and 0 < slope < 1:
        print("  -> a large share of the daytime warming is carried over "
              "from the night,\n     and it is damped on the way, which is "
              "the residual signature.")
    else:
        print("  -> the night explains little of the daytime warming. "
              "The residual\n     wording is not supported and should be "
              "softened.")

    # the map uses the same pooling, so panel (c) and panel (b) are the
    # same fit at different scales rather than two different analyses
    nsamp = Xp.shape[0] * Xp.shape[1]
    sl_map, r_map, _ = fit_cellwise(Xp.reshape(nsamp, *Xp.shape[2:]),
                                    Yp.reshape(nsamp, *Yp.shape[2:]))
    sl_map = np.where(land > 0.5, sl_map, np.nan)
    print(f"\n--- (c) carry-over slope at every land cell ---")
    print(f"  {nsamp} samples per cell, "
          f"{int(np.isfinite(sl_map).sum())} land cells")
    print(f"  land median slope {np.nanmedian(sl_map):+.3f}, "
          f"band median {np.nanmedian(sl_map[band]):+.3f}")

    # ---------------------------------------------------------------- figure
    # Two panels by default. The slope map is a third claim, and a third
    # claim is a third thing to defend; it stays available for the
    # supplement but does not crowd the main argument.
    if a.map:
        print("\ncolour scale:")
        smax = nice_levels(sl_map, "carry-over slope", hard=a.smax)
        fig = plt.figure(figsize=(13.0, 5.2))
        gs = fig.add_gridspec(1, 3, wspace=0.30)
    else:
        fig = plt.figure(figsize=(9.6, 4.6))
        gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axvspan(0, SUNRISE, color="0.90", zorder=0)
    ax1.axvspan(SUNSET, 23, color="0.90", zorder=0)
    ax1.fill_between(hours, d_cyc.min(axis=0), d_cyc.max(axis=0),
                     color="#2c7fb8", alpha=0.14, lw=0)
    ax1.plot(hours, mu, "-", color="#1a1a1a", lw=2.4,
             label=f"total $\\Delta$T2")
    ax1.axhline(inherited, color="#6a1b7a", ls="--", lw=1.8,
                label=f"$\\Delta$T2 at {h0:02d}:00")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.axvline(hmax, color="0.45", ls=":", lw=1.3)
    ax1.text(hmax, ax1.get_ylim()[1], f" {TNAME} ", rotation=90, va="top",
             ha="right", fontsize=8, color="0.35")
    ax1.set_xlim(0, 23)
    # y from zero, so the curve is read as a fraction of itself rather
    # than against a floating baseline. Warn if the band would be cut.
    lo = float(np.nanmin(d_cyc.min(axis=0)))
    if lo < 0:
        print(f"  NOTE: panel (a) y axis starts at 0, but the member "
              f"spread reaches {lo:+.2f} degC and is cut there")
    ax1.set_ylim(bottom=0)
    ax1.set_xticks([0, 6, 12, 18])
    ax1.set_xlabel("Hour (AWST)")
    ax1.set_ylabel(f"{wp.rlab(a.run)} $-$ {wp.rlab(a.ref)} ($^\\circ$C)")
    ax1.set_title(f"(a) Diurnal cycle of $\\Delta$T2",
                  fontsize=10, weight="bold")
    ax1.legend(fontsize=8, frameon=False, loc="lower left")
    ax1.grid(alpha=0.3, lw=0.5)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    if a.pool == "init":
        cols = {0: "#2c7fb8", 1: "#d6604d"}
        for g, name in enumerate(wp.PHYS):
            k = grp == g
            ax2.scatter(x[k], y[k], s=34, alpha=0.75, lw=0,
                        color=cols.get(g, "0.4"), label=name)
    else:
        ax2.scatter(x, y, s=34 if x.size < 120 else 20,
                    alpha=0.6, color="#2c7fb8", lw=0)
    xs = np.linspace(x.min(), x.max(), 50)
    ax2.plot(xs, slope * xs + intercept, "k-", lw=2.2,
             label=f"slope {slope:+.2f}")
    ax2.plot(xs, xs, color="0.6", ls=":", lw=1.2, label="1:1")
    ax2.set_xlabel(f"$\\Delta${TNNAME} ($^\\circ$C)")
    ax2.set_ylabel(f"$\\Delta${TNAME} ($^\\circ$C)")
    ax2.set_title(f"(b) $\\Delta${TNAME} vs $\\Delta${TNNAME}",
                  fontsize=10, weight="bold")

    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.grid(alpha=0.3, lw=0.5)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    if a.map:
        ax3 = fig.add_subplot(gs[0, 2], projection=wp.ROTATED_POLE)
        ax3.pcolormesh(lon, lat, np.where(land > 0.5, 1.0, 0.0),
                       cmap="Greys", vmin=0, vmax=4, shading="auto",
                       transform=ccrs.PlateCarree(), zorder=0)
        p = wp.draw(ax3, lon, lat, sl_map, -smax, smax, label="c", land=land)
        ax3.set_title("(c) Carry-over slope", fontsize=10,
                      weight="bold")
        cb = fig.colorbar(p, ax=ax3, orientation="horizontal",
                          fraction=0.055, pad=0.04)
        cb.set_label(f"$\\Delta${TNAME} per $\\Delta${TNNAME}",
                     fontsize=8.5)

    fig.tight_layout()
    wp.save(fig, os.path.join(a.outdir, "residual_warming.jpg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("mask", help="CoastalLandMask_N12.nc")
    ap.add_argument("--run", default="CNTL")
    ap.add_argument("--ref", default="EXP-SSTClim")
    ap.add_argument("--tmax-template", default=TMAX_UNIT,
                    help="daily-extremes file under DERIVED, with "
                         "{phy} {run} {date} placeholders")
    ap.add_argument("--tmax-var", default="TMAX")
    ap.add_argument("--tmin-var", default="TMIN")
    ap.add_argument("--pool", default="init",
                    choices=["member", "init", "all"],
                    help="how to pool before fitting. 'init' (default) "
                         "averages the 5 start dates and keeps the two "
                         "RCMs apart, 60 points; 'member' keeps all 10 "
                         "members, 300 points, which overstates the sample "
                         "because runs of one RCM on one day are copies, "
                         "not independent; 'all' averages everything into "
                         "30 daily points")
    ap.add_argument("--mask-var", default="auto")
    ap.add_argument("--smax", type=float, default=None,
                    help="colour limit for the slope map; omit to pick a "
                         "round number from the data")
    ap.add_argument("--rate-tol", type=float, default=0.005,
                    help="an hourly rise above this, between the handover "
                         "hour and the hour of the maximum, counts as a "
                         "daytime heat source (degC/h, default 0.005)")
    ap.add_argument("--map", action="store_true",
                    help="add the carry-over slope map as a third panel; "
                         "off by default, two panels carry the argument")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
