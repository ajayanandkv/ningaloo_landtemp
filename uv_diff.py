#!/usr/bin/env python
"""
Figure 10 of the manuscript, by period and by RCM.

    one row, five columns:
      20SST-ClimSST  40SST-ClimSST  60SST-ClimSST  80SST-ClimSST  CNTL-ClimSST

    vectors   the 10 m wind DIFFERENCE from ClimSST
    shading   the 10 m wind SPEED difference, over the whole domain

The submitted Figure 10 is this layout for the 24-hour mean. Reviewer 2
comment 1 asks for the analysis to be partitioned into day and night, so
this script takes the window as a flag and writes one figure per run:

  --when daily   all 24 hours, reproduces the submitted figure
  --when day     07-18 AWST
  --when night   19-06 AWST

  --rcm mean     R3 and R5 pooled, 10 members
  --rcm R3 / R5  one configuration, 5 members

SHADING IS THE DIFFERENCE OF SPEEDS, not the speed of the difference
vector. The first says whether the wind got stronger or weaker; the
second is positive even for a wind of unchanged speed that merely turns,
and would make a pure change of direction look like a change of
strength. It is taken per member and only then averaged, because the
speed of a mean vector is not the mean of the speeds.

Day, night and daily SHARE A COLOUR SCALE by default, so the three
figures can be read against each other. Pass --dmax to change it, and
change it for all three together or the comparison is lost.

The loaders come from fig_seabreeze.py, so this figure and that one
cannot drift apart.

Usage:
  python fig_wind_experiments.py <coast_geometry.nc>
        [--when daily|day|night] [--rcm mean|R3|R5]
        [--dmax 1.0] [--step 18] [--stipple] [--outdir figs]
"""
import argparse
import os

import numpy as np

import seabreeze as SB
import wrfplot as wp

WHEN_TITLE = {"daily": "all hours",
              "day": "daytime, 07-18 AWST",
              "night": "nighttime, 19-06 AWST"}


def main(a):
    land, lat, lon = wp.grid(a.coast)
    phys = list(wp.PHYS) if a.rcm == "mean" else [a.rcm]
    nmem = len(phys) * len(wp.DATES)

    print(f"{WHEN_TITLE[a.when]}, {a.rcm} ({nmem} members)")
    print(f"shading: 10 m wind speed difference from {wp.rlab(wp.REF_RUN)}")

    cols = []
    for run, title in zip(wp.SST_RUNS, wp.SST_TITLES):
        f, u, v, mask, n = SB.anomaly(a.coast, phys, run, a.when,
                                      wp.REF_RUN, "speed")
        cols.append(dict(title=title, f=f, u=u, v=v,
                         mask=(mask if a.stipple else None)))
        print(f"  {title:<16s} {np.nanmin(f):+6.2f} to {np.nanmax(f):+6.2f}"
              f"  mean {np.nanmean(f):+.3f} m/s  ({n} members)")

    dmax = a.dmax
    if dmax is None:
        allv = np.concatenate([c["f"].ravel() for c in cols])
        allv = allv[np.isfinite(allv)]
        dmax = float(np.ceil(np.percentile(np.abs(allv), 98) * 10) / 10)
        print(f"\ncolour limit +/-{dmax:g} m/s (98th pct, rounded). "
              f"Pass --dmax {dmax:g} to\npin the same scale on the other "
              f"windows, or the three cannot be compared.")
    else:
        print(f"\ncolour limit +/-{dmax:g} m/s (set by --dmax)")

    fig, axs = wp.panel_grid(
        1, len(cols),
        col_labels=[c["title"] for c in cols],
        suptitle=(a.title_text or "$\\Delta$10 m wind") if a.title else None)

    # ONE scale for every panel, set from the reference arrow: an arrow
    # of a.key m/s is drawn as a.arrow_frac of the panel width, in every
    # panel. Without this each panel autoscales to its own vectors and
    # the weakening from 20SST to CNTL cannot be read off the arrows.
    scale = a.scale if a.scale is not None else a.key / a.arrow_frac
    print(f"arrow scale {scale:.3g} (a {a.key:g} m/s arrow spans "
          f"{a.arrow_frac:.0%} of each panel, the same in all of them)")

    p = None
    for j, c in enumerate(cols):
        ax = axs[0, j]
        p = wp.draw(ax, lon, lat, c["f"], -dmax, dmax,
                    cmap=wp.CMAP, label=chr(97 + j), land=land,
                    stipple=c["mask"], label_pos=a.label_pos)
        # One reference arrow for the figure, under the last panel,
        # because the scale is now common to all of them.
        last = j == len(cols) - 1
        wp.quiver(ax, lon, lat, c["u"], c["v"], step=a.step, scale=scale,
                  key=a.key if last else None,
                  key_label=f"{a.key:g} m s$^{{-1}}$",
                  key_xy=tuple(a.key_xy), key_box=a.key_box)

    wp.hcolorbar_span(fig, p, "m s$^{-1}$", 0.3, 0.7)

    tag = "rcmmean" if a.rcm == "mean" else a.rcm
    wp.save(fig, os.path.join(a.outdir, f"wind_experiments_{a.when}_{tag}.jpg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("--when", default="daily",
                    choices=["daily", "day", "night"],
                    help="daily reproduces the submitted Figure 10")
    ap.add_argument("--rcm", default="mean", choices=["mean", "R3", "R5"])
    ap.add_argument("--arrow-frac", type=float, default=0.14,
                    help="length of the reference arrow as a fraction of "
                         "one panel's width. Sets the shared scale, so "
                         "raising it lengthens every arrow in the figure")
    ap.add_argument("--scale", type=float, default=None,
                    help="set the quiver scale directly (data units per "
                         "axes width) instead of deriving it from --key "
                         "and --arrow-frac. Use the SAME value on the "
                         "day, night and daily figures or the three "
                         "cannot be compared")
    ap.add_argument("--key-xy", type=float, nargs=2, default=[0.52, -0.10],
                    metavar=("X", "Y"),
                    help="where the reference arrow sits, in the last "
                         "panel's axes coordinates. A negative Y puts it "
                         "below the panel, which is the default")
    ap.add_argument("--key-box", action="store_true",
                    help="draw a white box behind the reference arrow. "
                         "Only needed when --key-xy puts it inside a "
                         "panel, over the shading")
    ap.add_argument("--label-pos", default="right",
                    choices=["left", "right"],
                    help="corner for the (a), (b) panel labels")
    ap.add_argument("--title-text", default=None,
                    help="replace the heading; default is "
                         '"$\\Delta$10 m wind"')
    ap.add_argument("--dmax", type=float, default=None,
                    help="colour limit, m/s. Omit and it is taken from the "
                         "data and printed; pin the same value on day, "
                         "night and daily so the three are comparable")
    ap.add_argument("--step", type=int, default=18,
                    help="vector thinning; raise it if the arrows crowd")
    ap.add_argument("--key", type=float, default=1.0,
                    help="reference arrow magnitude, m/s")
    ap.add_argument("--stipple", action="store_true",
                    help="hatch where at least 80%% of members agree on the "
                         "sign of the speed change")
    ap.add_argument("--title", action="store_true")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
