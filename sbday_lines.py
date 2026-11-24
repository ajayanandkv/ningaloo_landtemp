#!/usr/bin/env python
"""
Sea-breeze days as line plots, averaged over the paper's coastal band.

Two panels, both using CoastalLandMask_N12.nc so the numbers match the
ones s2_sbday_classify.py prints:

  (a) Across the imposed SST gradient
      ClimSST -> 20 -> 40 -> 60 -> 80 -> CNTL. A monotonic decline is
      dose-response evidence that the SST forcing is what changes the
      sea breeze, which is much harder to argue with than a single
      CNTL - ClimSST difference. Drawn only if at least three of the
      six experiments are present; missing ones are reported and
      skipped.

  (b) Along the coast
      Band-mean sea-breeze days against latitude, so the stretch of
      coast that loses breezes can be compared with the coastal bands
      in the T2 and wind figures.

Lines are the ensemble mean over the 5 initialisation dates, one colour
per RCM plus their mean; shading is the spread across members.

The band wraps the south-west corner, so its southern end faces south
rather than west. Panel (b) marks that latitude, below which "along the
coast" stops meaning "north to south".

Usage:
  python fig_sbday_lines.py <coast_geometry.nc> <CoastalLandMask_N12.nc>
         [--outdir figs] [--ref EXP-SSTClim] [--run CNTL] [--dlat 0.3]
"""
import argparse
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import wrfplot as wp

GROUP = "sbday"


def suffix(a):
    """Keep one method's figures from overwriting another's."""
    return f"-{a.tag}" if a.tag else ""

# Left to right: the full marine heatwave, then the imposed SST stepped
# back towards climatology. CNTL first, matching the map figures.
SST_ORDER = ["CNTL", "EXP-SST80", "EXP-SST60",
             "EXP-SST40", "EXP-SST20", "EXP-SSTClim"]
SST_LABEL = ["CNTL", "80SST", "60SST", "40SST", "20SST", "ClimSST"]

COLORS = {"R3": "#2c7fb8", "R5": "#d6604d", "mean": "#1a1a1a"}


def band_mask(mask_file, var="auto"):
    with xr.open_dataset(mask_file) as ds:
        if var == "auto":
            cands = [v for v in ds.data_vars if ds[v].ndim == 2]
            if not cands:
                raise SystemExit(f"ERROR: no 2-D variable in {mask_file}")
            var = cands[0]
        m = ds[var].isel(**wp.CROP).values
    band = np.isfinite(m) & (np.nan_to_num(m) > 0.5)
    print(f"coastal band '{var}': {band.sum()} points after crop")
    return band


def ens_file(phy, run, tag):
    """The -freq.nc for one method variant written by s2."""
    p = wp.ens_path(GROUP, phy, run)
    return p.replace("-freq.nc", f"-freq-{tag}.nc") if tag else p


def load_days(run, tag="", say=False):
    """sb_days per member (phy-major) for one run, or None if absent."""
    out = []
    for phy in wp.PHYS:
        p = ens_file(phy, run, tag)
        if not os.path.exists(p):
            return None
        with xr.open_dataset(p) as ds:
            if say and phy == wp.PHYS[0] and "method" in ds.attrs:
                print(f"  method '{ds.attrs['method']}', afternoon window "
                      f"'{ds.attrs.get('pm_window', '?')}' "
                      f"({ds.attrs.get('pm_window_hours', '?')})")
            out.append(ds["sb_days"].isel(**wp.CROP).values)
    return np.concatenate(out)


def by_phy(days):
    """Split the member stack back into one array per RCM."""
    return dict(zip(wp.PHYS, np.split(days, len(wp.PHYS))))


SPREAD_NOTE = {
    "init": "min-max across the 5 initialisation dates within each RCM",
    "member": "$\\pm$1 s.d. across members of the area mean",
    "sem": "$\\pm$1 standard error of the 10-member mean",
    "within": "$\\pm$1 s.d. across initialisation dates, pooled within each RCM",
    "grid": "$\\pm$1 s.d. across members at a grid point, band-averaged",
    "spatial": "$\\pm$1 s.d. across grid points within the band",
}


def spread_band(vals, band, how, per_phy=None):
    """
    Mean and the two edges of the shaded band, for one experiment.

    vals is a list of per-member (y, x) fields. The four options answer
    DIFFERENT questions and differ by a lot, so the choice belongs in
    the caption:

      init / member  how uncertain the AREA MEAN is. Smallest, because
                     averaging over hundreds of grid points removes most
                     of the noise before the spread is taken.
      grid           how uncertain a SINGLE GRID POINT is. Larger, and
                     the honest answer when the map panels are what the
                     reader is being asked to trust.
      spatial        how much the field VARIES across the zone. Largest,
                     and not an uncertainty at all: a perfectly
                     reproducible field with a strong coastal gradient
                     has a wide spatial spread and zero member spread.
    """
    b = np.array([v[band].mean() for v in vals])
    mu = float(b.mean())
    if how == "within":
        # Spread of the initialisation dates ONLY, with the systematic
        # R3-minus-R5 offset removed: the within-RCM variances are
        # pooled rather than the members thrown into one pot. On the
        # mean series, "member" and "sem" both fold that offset into the
        # bar, which makes a model difference look like noise.
        if not per_phy:
            raise SystemExit("ERROR: --spread within needs the per-RCM "
                             "members; it is not available here")
        var = []
        for mem in per_phy.values():
            bb = np.array([v[band].mean() for v in mem])
            if len(bb) > 1:
                var.append(bb.var(ddof=1))
        s = float(np.sqrt(np.mean(var))) if var else 0.0
        return mu, mu - s, mu + s
    if how == "init":
        return mu, float(b.min()), float(b.max())
    if how == "member":
        s = float(b.std())
    elif how == "sem":
        # Spread of the MEAN rather than of the members: s / sqrt(n).
        # Treat it as a lower bound. The 10 members are 2 RCMs x 5
        # initialisation dates, and the RCM offset is systematic rather
        # than random, so they are not 10 independent draws and this
        # overstates confidence somewhat.
        s = float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else 0.0
    elif how == "grid":
        # spread across members AT each point, then averaged over the
        # band -- not the spread of the band mean
        s = float(np.nanmean(np.stack(vals).std(axis=0)[band]))
    elif how == "spatial":
        s = float(np.nanstd(np.stack(vals).mean(axis=0)[band]))
    else:
        raise SystemExit(f"ERROR: unknown --spread '{how}'; choose from "
                         f"{sorted(SPREAD_NOTE)}")
    return mu, mu - s, mu + s


def panel_gradient(ax, band, found, spread_all=False, legend_loc="lower right",
                   spread="init", style="line", series_only="all", note=True):
    """
    The gradient panel: sea-breeze days against experiment.

    --style line      filled band behind each line. Reads as a continuous
                      envelope, which suits a shaded range.
    --style errorbar  capped bars at each point, as in fig_sbstrength.
                      The three series are dodged slightly in x so their
                      bars do not overlap, and all three carry bars: a
                      bar does not swamp the panel the way a filled band
                      does, so the mean no longer has to go bare.

    Shading defaults to the spread across the 5 initialisation dates
    WITHIN each RCM -- the noise the trend has to beat. Pooling all 10
    members into one band instead would fold in the systematic offset
    between R3 and R5, which is a model difference, not noise. In line
    style the mean therefore carries no band unless --spread-all is
    passed; the two coloured lines already bound it.
    """
    x = np.arange(len(found))
    # --series mean drops the per-RCM lines. The mean then carries the
    # panel alone, so there is nothing for a legend to distinguish and
    # no reason to dodge in x.
    series = ["mean"] if series_only == "mean" else wp.PHYS + ["mean"]
    # dodge only in errorbar style, so bars at the same experiment are
    # readable; in line style the points must stay on the tick
    step = 0.13 if style == "errorbar" else 0.0
    off = {p: (i - (len(series) - 1) / 2.0) * step
           for i, p in enumerate(series)}

    for phy in series:
        mu, lo, hi = [], [], []
        for _, days in found:
            per = by_phy(days)
            vals = (np.concatenate(list(per.values())) if phy == "mean"
                    else per[phy])
            m, l, h = spread_band(list(vals), band, spread,
                                  per_phy=(per if phy == "mean"
                                           else {phy: per[phy]}))
            mu.append(m); lo.append(l); hi.append(h)
        mu, lo, hi = np.array(mu), np.array(lo), np.array(hi)
        lead = phy == "mean"
        z = 5 if lead else 3
        lab = "R3 and R5 mean" if lead else phy

        if style == "errorbar":
            # No connecting line: the x axis is categorical, so a line
            # between experiments implies a continuum that is not there.
            # The same reason the panel carries no vertical gridlines.
            # yerr is asymmetric, because --spread init is a min-max
            # range and need not be centred on the mean.
            ax.errorbar(x + off[phy], mu, yerr=[mu - lo, hi - mu],
                        marker="o", ms=6.5 if lead else 5.0,
                        ls="none", color=COLORS[phy],
                        capsize=3.5, capthick=1.2, elinewidth=1.2,
                        mec="white", mew=1.1, zorder=z, label=lab)
            continue

        if not lead or spread_all:
            ax.fill_between(x, lo, hi, color=COLORS[phy], alpha=0.13,
                            lw=0, zorder=z - 2)
        ax.plot(x, mu, "-", color=COLORS[phy], lw=2.6 if lead else 1.7,
                zorder=z, label=lab, alpha=1.0 if lead else 0.9)
        ax.plot(x, mu, "o", color=COLORS[phy], ms=6.5 if lead else 5.0,
                mec="white", mew=1.3, zorder=z + 1)

    ax.set_xticks(x)
    ax.set_xticklabels([SST_LABEL[SST_ORDER.index(r)] for r, _ in found])
    ax.set_xlim(-0.35, len(found) - 0.65)
    ax.set_xlabel("Experiments", labelpad=8)
    ax.set_ylabel("Sea-breeze days in March", labelpad=8)

    # horizontal rules only: the x axis is categorical, so vertical
    # gridlines suggest a continuous scale that is not there
    ax.grid(axis="y", alpha=0.35, lw=0.6, color="0.7")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.4")
    ax.tick_params(length=3.5, color="0.4", labelsize=10)

    if len(series) > 1:
        ax.legend(fontsize=9, frameon=False, loc=legend_loc,
                  handlelength=1.8, borderaxespad=0.8)

    # Say on the panel which spread is drawn. The four differ by a lot,
    # and a reader cannot tell them apart by eye.
    # The note names which of the four spreads is drawn; they differ by
    # a lot and cannot be told apart by eye. Dropping it moves that duty
    # to the caption, which then MUST carry it.
    if note:
        word = "bars" if style == "errorbar" else "shading"
        ax.text(0.015, 0.015, f"{word}: {SPREAD_NOTE[spread]}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, color="0.4")

    first = np.array([v[band].mean() for v in found[0][1]]).mean()
    last = np.array([v[band].mean() for v in found[-1][1]]).mean()
    print(f"  {found[0][0]} {first:.2f} days -> {found[-1][0]} {last:.2f} "
          f"days, change {last - first:+.2f}")


def panel_alongcoast(ax, band, lat, runs, dlat, turn_lat):
    edges = np.arange(np.floor(lat[band].min() / dlat) * dlat,
                      lat[band].max() + dlat, dlat)
    centres = 0.5 * (edges[:-1] + edges[1:])

    for label, days, color, ls in runs:
        prof = []
        for v in days:
            row = [v[band & (lat >= a) & (lat < b)].mean()
                   if (band & (lat >= a) & (lat < b)).any() else np.nan
                   for a, b in zip(edges[:-1], edges[1:])]
            prof.append(row)
        prof = np.array(prof)
        mu = np.nanmean(prof, axis=0)
        ax.fill_between(centres, np.nanmin(prof, axis=0),
                        np.nanmax(prof, axis=0), color=color, alpha=0.12, lw=0)
        ax.plot(centres, mu, ls, color=color, lw=2.0, label=wp.rlab(label))

    if turn_lat is not None:
        ax.axvline(turn_lat, color="0.45", ls=":", lw=1.3)
        ax.text(turn_lat, ax.get_ylim()[1], " coast turns ", rotation=90,
                va="top", ha="right", fontsize=8, color="0.35")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Sea-breeze days in March")
    ax.set_title("(b) Along the coastal band", fontsize=11, weight="bold")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower left")


def inset_band(fig, coast_file, band, rect):
    """Small map in the corner showing which points the line averages over."""
    with xr.open_dataset(coast_file) as c:
        lat = c["XLAT"].isel(**wp.CROP).values
        lon = c["XLONG"].isel(**wp.CROP).values
        land = c["landmask"].isel(**wp.CROP).values
    ax = fig.add_axes(list(rect), projection=wp.ROTATED_POLE)
    tr = ccrs.PlateCarree()
    ax.pcolormesh(lon, lat, land, cmap="Greys", vmin=0, vmax=2.4,
                  shading="auto", transform=tr, zorder=0)
    ax.contourf(lon, lat, band.astype(float), levels=[0.5, 1.5],
                colors=["#6a1b7a"], transform=tr, zorder=1)
    ax.contour(lon, lat, land, [0.5], colors="k", linewidths=0.4,
               transform=tr, zorder=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    return ax


def find_turn(coast_file, band, lat, dlat):
    """
    Latitude below which the band faces south rather than west.

    Counting how many points fall in a north-ish angle range is fragile:
    the angle wraps at 0/360, so scattered points on a cleanly
    west-facing stretch can still push the count over half. Averaging
    the onshore unit VECTOR over each latitude band avoids the wrap
    entirely -- if its north component dominates its east component, the
    coast there faces south. Returns None when no band qualifies, i.e.
    the coast does not turn inside the masked region.
    """
    with xr.open_dataset(coast_file) as c:
        if "onshore_n" not in c:
            return None
        oe = c["onshore_e"].isel(**wp.CROP).values
        on = c["onshore_n"].isel(**wp.CROP).values

    # The south coast is at the SOUTHERN end of the band, so walk up from
    # there while the coast is still south-facing and stop where it is
    # not. Scanning for the highest qualifying latitude instead would
    # latch onto any noise at the northern end.
    edges = np.arange(np.floor(lat[band].min() / dlat) * dlat,
                      lat[band].max() + dlat, dlat)
    turn = None
    for a, b in zip(edges[:-1], edges[1:]):
        sel = band & (lat >= a) & (lat < b) & np.isfinite(on)
        if sel.sum() < 5:
            continue
        if abs(np.nanmean(on[sel])) > abs(np.nanmean(oe[sel])):
            turn = 0.5 * (a + b)
        else:
            break                      # the south-facing stretch has ended
    if turn is None:
        print("  no majority south-facing latitudes in the band "
              "-- the coast does not turn inside it")
    else:
        print(f"  band turns south-facing below {turn:.2f} deg")
    return turn


def main(a):
    band = band_mask(a.mask)
    if a.tag:
        print(f"reading the '{a.tag}' variant")
    with xr.open_dataset(a.coast) as c:
        lat = c["XLAT"].isel(**wp.CROP).values

    found, missing = [], []
    for r in SST_ORDER:
        d = load_days(r, a.tag, say=not found)
        (found.append((r, d)) if d is not None else missing.append(r))
    if missing:
        print(f"not present, skipped: {', '.join(missing)}")
    if not found:
        raise SystemExit("ERROR: no sbday ensemble files found")

    if a.order == "clim-first":
        # ClimSST on the left, CNTL on the right. The panels then read as
        # the MHW being built up, which is the direction the text
        # describes and the direction fig_sbstrength is drawn in. Only
        # the drawing order changes; nothing is recomputed. The caption
        # must name whichever direction is used.
        found = found[::-1]
        print("  x axis reversed: ClimSST -> CNTL")

    if len(found) < 3:
        want = f"-freq-{a.tag}.nc" if a.tag else "-freq.nc"
        raise SystemExit(
            f"ERROR: only {len(found)} experiment(s) present, need at least "
            f"3 for the SST-gradient figure.\n"
            f"       Missing: {', '.join(missing)}\n"
            f"       Looking for files ending '{want}'. Stage 2 writes only "
            f"its --run and --ref, so each experiment needs its own call:\n"
            f"         for r in {' '.join(missing)}; do\n"
            f"           python s2_sbday_classify.py <mask.nc> --run $r "
            f"--ref {a.ref}{' --tag-matching flags' if a.tag else ''}\n"
            f"         done")

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.6))
    panel_gradient(ax, band, found, spread_all=a.spread_all,
                   spread=a.spread,
                   style=a.style,
                   series_only=a.series, note=not a.no_note,
                   legend_loc=a.legend)

    if a.alongcoast:
        runs = []
        for label, color, ls in [(a.ref, COLORS["R3"], "--"),
                                 (a.run, COLORS["R5"], "-")]:
            d = load_days(label, a.tag)
            if d is None:
                raise SystemExit(f"ERROR: {label} not found")
            runs.append((label, d, color, ls))
        turn = find_turn(a.coast, band, lat, a.dlat)
        f2, a2 = plt.subplots(1, 1, figsize=(7.2, 5.0))
        panel_alongcoast(a2, band, lat, runs, a.dlat, turn)
        a2.set_title("Along the coastal band", fontsize=11, weight="bold")
        f2.tight_layout()
        wp.save(f2, os.path.join(a.outdir, f"sbday_alongcoast{suffix(a)}.png"))

    fig.suptitle("March 2011 sea-breeze days across experiments",
                 fontsize=12.5, weight="bold", y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    # after tight_layout: a manually placed axes confuses it
    inset_band(fig, a.coast, band, a.inset)
    wp.save(fig, os.path.join(a.outdir, f"sbday_lines{suffix(a)}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("mask", help="CoastalLandMask_N12.nc")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--run", default="CNTL")
    ap.add_argument("--ref", default="EXP-SSTClim")
    ap.add_argument("--tag", default="",
                    help="method variant to read, e.g. hours or maxmean")
    ap.add_argument("--alongcoast", action="store_true",
                    help="also write the along-coast profile as its own figure")
    ap.add_argument("--inset", type=float, nargs=4,
                    default=[0.155, 0.660, 0.110, 0.175],
                    help="inset map position: x0 y0 width height, 0-1")
    ap.add_argument("--legend", default="lower right",
                    help="legend corner, e.g. 'upper left' (default: "
                         "lower right)")
    ap.add_argument("--series", default="all", choices=["all", "mean"],
                    help="mean draws only the R3 and R5 mean in the "
                         "gradient panel, dropping the per-RCM lines")
    ap.add_argument("--no-note", action="store_true",
                    help="drop the small label naming which spread is "
                         "drawn. The CAPTION must then say it instead")
    ap.add_argument("--order", default="cntl-first",
                    choices=["cntl-first", "clim-first"],
                    help="x-axis direction. clim-first runs ClimSST to "
                         "CNTL, matching fig_sbstrength and the paper's "
                         "own 'build up' framing; the caption must say "
                         "whichever is used")
    ap.add_argument("--style", default="line",
                    choices=["line", "errorbar"],
                    help="gradient panel: filled band behind lines, or "
                         "capped error bars as in fig_sbstrength")
    ap.add_argument("--spread", default="init",
                    choices=["init", "member", "sem", "within", "grid", "spatial"],
                    help="what the shading in the gradient panel means. "
                         "init/member: uncertainty of the AREA MEAN. "
                         "grid: uncertainty at a SINGLE grid point, which "
                         "is larger. spatial: variation ACROSS the zone, "
                         "which is not an uncertainty. Name it in the "
                         "caption; they differ by a lot")
    ap.add_argument("--spread-all", action="store_true",
                    help="also shade the pooled 10-member spread around the "
                         "mean; off by default because it folds the R3-R5 "
                         "offset into what reads as noise")
    ap.add_argument("--dlat", type=float, default=0.3,
                    help="latitude bin width for panel (b)")
    main(ap.parse_args())
