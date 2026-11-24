#!/usr/bin/env python
"""
Sea-breeze days: the manuscript figure, all four panels in one file.

    (a) CNTL              (b) ClimSST            sea-breeze days per March
    (c) CNTL - ClimSST    (d) every experiment, CNTL through ClimSST,
        difference, days       averaged over the coastal band

Panel (d) is drawn by fig_sbday_lines.panel_gradient, the same routine
that makes the standalone line figure, so the two cannot diverge. The
shading there is the spread across the five initialisation dates WITHIN
each RCM, which is the noise the trend has to beat; pooling all ten
would fold in the systematic R3/R5 offset, which is a model difference
rather than noise.

The y-axis is not forced to zero. From zero, an 18-to-22-day range
collapses to nothing. Read the change against the shaded spread, not
against the axis.

This script only draws. All loading comes from the two existing
scripts, so the maps here and the ones from fig_sbday.py cannot drift
apart.

Usage:
  python fig_sbday_combined.py <coast_geometry.nc> <CoastalLandMask_N12.nc>
        [--tag core] [--spread sd|range] [--outdir figs]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import sbday as S
import sbday_lines as L
import wrfplot as wp



def main(a):
    land, lat, lon = wp.grid(a.coast)
    keep = S.plot_mask(a)
    band = L.band_mask(a.mask)

    found = []
    for run in L.SST_ORDER:
        d = L.load_days(run, a.tag, say=(run == L.SST_ORDER[0]))
        if d is None:
            # name the path, otherwise a wrong --tag looks like missing data
            print(f"  {run}: not found, skipped\n"
                  f"      tried {L.ens_file(wp.PHYS[0], run, a.tag)}")
            continue
        found.append((run, d))
    if not found:
        raise SystemExit(
            "ERROR: no experiments found.\n"
            "       The paths above are what was tried. If your files carry "
            "a method tag\n"
            "       (for example sbday_d01_2011-03-freq-core.nc) pass it "
            "with --tag core.")

    if a.order == "clim-first":
        # ClimSST on the left, CNTL on the right. The panels then read as
        # the MHW being built up, which is the direction the text
        # describes and the direction fig_sbstrength is drawn in. Only
        # the drawing order changes; nothing is recomputed. The caption
        # must name whichever direction is used.
        found = found[::-1]
        print("  x axis reversed: ClimSST -> CNTL")

    runs = dict(found)
    for need in (a.run, a.ref):
        if need not in runs:
            raise SystemExit(f"ERROR: {need} is needed for the maps "
                             f"but was not found")

    def cut(f):
        return np.where(keep, f, np.nan)

    d_run = cut(runs[a.run].mean(axis=0))
    d_ref = cut(runs[a.ref].mean(axis=0))
    dd = runs[a.run] - runs[a.ref]              # member-paired
    d_diff = cut(dd.mean(axis=0))
    stip = None if a.no_stipple else (S.agreement(dd) & keep)

    vmax = np.nanpercentile(np.concatenate([d_run.ravel(), d_ref.ravel()]), 99)
    dmax = max(abs(np.nanpercentile(d_diff, 2)),
               abs(np.nanpercentile(d_diff, 98))) or 1.0

    print(f"\nmaps: {wp.rlab(a.run)} mean {np.nanmean(d_run):.2f}, "
          f"{wp.rlab(a.ref)} mean {np.nanmean(d_ref):.2f}, "
          f"difference {np.nanmean(d_diff):+.2f} days")

    # 2 x 2: the two runs on top, the difference and the line panel below.
    # hspace is wide because the shared colour bar for (a) and (b) sits
    # between the rows -- at a smaller value it lands on the titles of
    # (c) and (d).
    # panel_grid is not used here, so set the font scale by hand; this
    # figure is about three panels wide, like the advection one.
    wp.set_fscale(0.6)

    fig = plt.figure(figsize=(11.5, 10.8))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.12)

    cells = [gs[0, 0], gs[0, 1], gs[1, 0]]
    axs = [fig.add_subplot(c, projection=wp.ROTATED_POLE) for c in cells]
    for ax in axs:
        ax.coastlines()
        ax.add_feature(wp.cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(wp.cfeature.LAND, facecolor="lightgray")

    for ax, lab in zip(axs, [wp.rlab(a.run), wp.rlab(a.ref),
                             f"{wp.rlab(a.run)} $-$ {wp.rlab(a.ref)}"]):
        ax.set_title(lab, fontsize=12, weight="bold")

    p_tot = None
    for j, f in enumerate((d_run, d_ref)):
        p_tot = wp.draw(axs[j], lon, lat, f, 0, vmax,
                        cmap=plt.cm.viridis, label=chr(97 + j), land=land)
    p_d = wp.draw(axs[2], lon, lat, d_diff, -dmax, dmax,
                  label="c", land=land, stipple=stip)

    wp.cbar(fig, p_tot, axs[:2], "Sea-breeze days per March",
            orientation="horizontal", fraction=0.045, pad=0.03, shrink=0.72)
    wp.cbar(fig, p_d, axs[2], "Difference (days)",
            orientation="horizontal", fraction=0.045, pad=0.03, shrink=0.72)

    axd = fig.add_subplot(gs[1, 1])
    print("\n(d) sea-breeze days over the band")
    L.panel_gradient(axd, band, found, spread_all=a.spread_all,
                     spread=a.spread,
                     style=a.style,
                     series_only=a.series, note=not a.no_note,
                     legend_loc=a.legend)
    axd.set_title("(d) Sea-breeze days by experiment",
                  fontsize=11, weight="bold")

    fig.suptitle("March 2011 sea-breeze days, R3 and R5 mean",
                 fontsize=15, weight="bold", y=0.955)
    wp.save(fig, os.path.join(a.outdir, f"sbday_combined{S.suffix(a)}.jpg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("mask", help="CoastalLandMask_N12.nc")
    ap.add_argument("--run", default="CNTL")
    ap.add_argument("--ref", default="EXP-SSTClim")
    ap.add_argument("--tag", default="",
                    help="method variant written by s2, e.g. core")
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
                    help="also shade the spread of the pooled 10 members "
                         "around the mean line; off by default, since it "
                         "includes the R3/R5 offset")
    ap.add_argument("--legend", default="lower right",
                    help="legend position in panel (d)")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="limit the maps to this distance from the coast, km")
    ap.add_argument("--land-only", action="store_true")
    ap.add_argument("--no-stipple", action="store_true")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
