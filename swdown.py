#!/usr/bin/env python
"""
Supplementary figure: daytime downwelling shortwave radiation difference.

    columns  20SST, 40SST, 60SST, 80SST, CNTL   (each minus ClimSST)

The manuscript states that the simulations "did not show an increase in
downwelling shortwave radiation during the day (not shown)". This draws
it, so that claim stops resting on an unshown result. A reviewer who
asks to see it is asking a fair question, and the answer is one panel
row.

It is deliberately NOT a sixth row of Figure 7. That figure explains the
nighttime warming and is already five rows deep; shortwave belongs to
the daytime argument and would need a sixth colour bar on a figure that
has four.

  --when day     DEFAULT. 07-18 AWST, the window the claim is about.
  --when all     all hours. Half of them are dark, so the mean is close
                 to half the daytime value and means little on its own.
  --when night   a null check: SWDOWN should be ~0 everywhere, so this
                 run should be blank. If it is not, the hour indexing is
                 wrong and the daytime panel cannot be trusted either.
  --when diff    day minus night, which for this variable is almost the
                 same as --when day.

  --rcm mean     R3 and R5 pooled, 10 members. One row.
  --rcm R3 / R5  one configuration, 5 members. One row.
  --rcm split    three rows: R3, R5, and R3 minus R5, as in S10-S14.

  --var SWDOWN   DEFAULT. Downwelling shortwave at the surface, which is
                 what the manuscript sentence claims.
  --var SWNET    SWDOWN * (1 - ALBEDO), the part actually absorbed. This
                 is the term that enters the surface energy budget, and
                 it can differ in magnitude from SWDOWN wherever the
                 albedo responds. Worth checking once; quote whichever
                 one the sentence in the text is about.

LIMITS come from the data and are printed as a --vmax line, because
unlike the Figure 7 variables there is no submitted version of this
figure to match. Pin the value once, so the per-RCM runs share a scale.

Usage:
  python fig_swdown.py <coast_geometry.nc> [CoastalLandMask_N12.nc]
        [--when day|all|night|diff] [--rcm mean|R3|R5|split]
        [--var SWDOWN|SWNET] [--vmax 5] [--ocean] [--stipple]
        [--table] [--outdir figs]
"""
import argparse
import os

import numpy as np
import xarray as xr

import dvar_daynight as D
import wrfplot as wp

GROUP = "energy"

VARS = {
    "SWDOWN": dict(label="$\\Delta$SWDOWN", unit="W m$^{-2}$",
                   title="downwelling shortwave at the surface"),
    "SWNET": dict(label="$\\Delta$SW$_{net}$", unit="W m$^{-2}$",
                  title="net shortwave at the surface, "
                        "SWDOWN $\\times$ (1 $-$ albedo)"),
}

ROW_LABELS = ["R3", "R5", "R3$-$R5"]


def hourly(phy, run, date, var):
    """One member's diurnal cycle, with SWNET formed hour by hour."""
    if var == "SWDOWN":
        return wp.member_cycle(GROUP, phy, run, date, "SWDOWN")
    # Form the product per hour and per member, never from window means:
    # the mean of a product is not the product of the means, and the
    # albedo varies through the day.
    sw = wp.member_cycle(GROUP, phy, run, date, "SWDOWN")
    al = wp.member_cycle(GROUP, phy, run, date, "ALBEDO")
    return sw * (1.0 - al)


def window(da, when):
    if when == "diff":
        return window(da, "day") - window(da, "night")
    hours = wp.DAY_HOURS if when == "day" else (
        wp.NIGHT_HOURS if when == "night" else list(range(24)))
    return da.sel(hour=[h for h in hours if h in da["hour"].values]).mean(
        dim="hour")


def stack(var, run, when, phys, ref=None):
    """Member-wise anomaly, (member, y, x)."""
    ref = ref or wp.REF_RUN
    out = []
    for phy in phys:
        for date in wp.DATES:
            a = window(hourly(phy, run, date, var), when)
            b = window(hourly(phy, ref, date, var), when)
            out.append(a - b)
    return xr.concat(out, dim="member")


def load_band(path, land):
    if path is None:
        print("no mask given: band means over ALL land points")
        return land > 0.5
    with xr.open_dataset(path) as ds:
        v = [k for k in ds.data_vars if ds[k].ndim == 2][0]
        m = ds[v].isel(**wp.CROP).values
    band = np.isfinite(m) & (np.nan_to_num(m) > 0.5) & (land > 0.5)
    print(f"band '{v}': {int(band.sum())} coastal land points")
    return band


def table(a, band):
    """The number that replaces '(not shown)' in the text."""
    phys = D.phys_for("mean" if a.rcm == "split" else a.rcm)
    print(f"\n{VARS[a.var]['title']}, {len(phys) * len(wp.DATES)} members")
    print("mean +/- spread across members, then how many agree on the sign\n")
    for when in ("all", "day", "night", "diff"):
        cells = []
        for run in wp.SST_RUNS:
            st = stack(a.var, run, when, phys)
            per = np.array([np.nanmean(np.where(band, v, np.nan))
                            for v in np.asarray(st)])
            pos, neg = int((per > 0).sum()), int((per < 0).sum())
            cells.append(f"{np.nanmean(per):+.3f}+/-{np.nanstd(per):.3f} "
                         f"{max(pos, neg)}/{len(per)}")
        print(f"  {when:<6s}" + "".join(f"{c:>22s}" for c in cells))
    print("  " + " " * 6 + "".join(f"{t:>22s}" for t in wp.SST_TITLES))
    print("\nNighttime should be ~0 for a shortwave term. A value far from "
          "zero means\nthe hour indexing is wrong and the daytime row is "
          "not trustworthy either.")


def main(a):
    land, lat, lon = wp.grid(a.coast)
    keep = np.ones_like(land, dtype=bool) if a.ocean else (land > 0.5)

    if a.table:
        table(a, load_band(a.mask, land))
        return

    meta = VARS[a.var]
    split = a.rcm == "split"
    phys = D.phys_for("mean" if split else a.rcm)
    band = load_band(a.mask, land) if a.mask else (land > 0.5)

    print(f"{meta['title']}, {D.WHEN_TITLE[a.when]}, --rcm {a.rcm}")

    cols = []
    for run, title in zip(wp.SST_RUNS, wp.SST_TITLES):
        if split:
            r3 = stack(a.var, run, a.when, ["R3"])
            r5 = stack(a.var, run, a.when, ["R5"])
            m3 = np.asarray(r3.mean(dim="member"))
            m5 = np.asarray(r5.mean(dim="member"))
            maps, stacks = [m3, m5, m3 - m5], [r3, r5, None]
        else:
            st = stack(a.var, run, a.when, phys)
            maps, stacks = [np.asarray(st.mean(dim="member"))], [st]
        cols.append(dict(title=title, maps=maps, stacks=stacks))
        bm = np.nanmean(np.where(band, maps[0], np.nan))
        print(f"  {title:<16s} band mean {bm:+8.3f} {meta['unit']}"
              f"   range {np.nanmin(np.where(keep, maps[0], np.nan)):+8.3f}"
              f" to {np.nanmax(np.where(keep, maps[0], np.nan)):+8.3f}")

    allv = np.concatenate([np.where(keep, m, np.nan).ravel()
                           for c in cols for m in c["maps"]])
    allv = allv[np.isfinite(allv)]
    if a.vmax is not None:
        vmax, why = a.vmax, "set by --vmax"
    else:
        vmax = D.nice_round(float(np.percentile(np.abs(allv), 98)))
        why = "98th pct of the data, rounded"
    clipped = 100.0 * np.mean(np.abs(allv) > vmax)
    print(f"\ncolour limit +/-{vmax:g} {meta['unit']} ({why}), "
          f"{clipped:.2f}% clipped")
    if a.vmax is None:
        print(f"  pin it across the per-RCM runs with:  --vmax {vmax:g}")

    nrow = 3 if split else 1
    fig, axs = wp.panel_grid(
        nrow, len(cols),
        col_labels=[c["title"] for c in cols],
        row_labels=(ROW_LABELS if split else [meta["label"]]),
        suptitle=(None if a.no_title else
                  f"{meta['label']}  ({D.WHEN_TITLE[a.when]}, March 2011)"))

    p = None
    for j, c in enumerate(cols):
        for r in range(nrow):
            st = c["stacks"][r]
            mask = None
            if a.stipple and st is not None:
                frac = 0.8 if not split else 0.8     # 8/10 pooled, 4/5 split
                mask = np.asarray(wp.agreement_mask(st, fraction=frac))
            p = wp.draw(axs[r, j], lon, lat,
                        np.where(keep, c["maps"][r], np.nan),
                        -vmax, vmax, cmap=a.cmap,
                        label=chr(97 + r * len(cols) + j),
                        land=land, stipple=mask)

    wp.cbar(fig, p, list(axs.ravel()),
            f"{meta['label']} ({meta['unit']})")

    tag = "rcmmean" if a.rcm == "mean" else a.rcm
    name = (f"swrad_{a.var.lower()}_{a.when}_{tag}"
            f"{'_ocean' if a.ocean else ''}")
    wp.save(fig, os.path.join(a.outdir, f"{name}.jpg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("mask", nargs="?", default=None,
                    help="CoastalLandMask_N12.nc, for the band means")
    ap.add_argument("--when", default="day",
                    choices=["day", "all", "night", "diff"],
                    help="day is the window the manuscript claim is about")
    ap.add_argument("--rcm", default="mean",
                    choices=["mean", "R3", "R5", "split"],
                    help="split draws R3, R5 and R3-R5 as three rows")
    ap.add_argument("--var", default="SWDOWN", choices=sorted(VARS),
                    help="SWDOWN is the downwelling flux the text claims "
                         "about; SWNET is the absorbed part")
    ap.add_argument("--vmax", type=float, default=None,
                    help="colour limit; omitted, it is taken from the data "
                         "and printed so it can be pinned")
    ap.add_argument("--cmap", default=D.DEFAULT_CMAP)
    ap.add_argument("--ocean", action="store_true",
                    help="draw the ocean too; land only by default")
    ap.add_argument("--stipple", action="store_true",
                    help="sign agreement: 8 of 10 pooled, 4 of 5 per RCM")
    ap.add_argument("--table", action="store_true",
                    help="print band means for every window and experiment "
                         "instead of drawing; this is the number that "
                         "replaces '(not shown)' in the text")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
