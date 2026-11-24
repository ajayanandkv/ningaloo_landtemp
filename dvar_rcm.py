#!/usr/bin/env python
"""
Supplementary figures S10-S14: one variable, R3 and R5 and their difference.

    row 1   R3                5 experiment columns
    row 2   R5
    row 3   R3 - R5

This is the layout of the submitted S10-S14, reproduced so the same
figure can be made for any window. Reviewer 2 asks for the analysis to
be partitioned into day and night; the main-text Figure 7 becomes the
nighttime field, so the per-RCM supplements have to follow it.

    --when all     reproduces the submitted supplements
    --when night   the nighttime window, to match the revised Figure 7
    --when day     the daytime window
    --when diff    day anomaly minus night anomaly

One figure per variable. --var all writes all five in one run.

WHAT THE THIRD ROW IS. R3 minus R5 of the ANOMALIES, member-paired
within each configuration:

    (R3 CNTL - R3 ClimSST) - (R5 CNTL - R5 ClimSST)

so it shows where the two PBL schemes disagree about the RESPONSE to the
SSTs, not where they disagree about the climate. Those are different
questions and the second one is not what the supplement is for.

ONE COLOUR SCALE FOR ALL THREE ROWS by default, as in the submitted
figures, so the scheme difference can be read against the response
itself rather than against a stretched scale of its own. --split-cbar
gives the difference row its own scale and its own bar; use it only when
the difference would otherwise be invisible, and say so in the caption,
because a reader will assume one scale.

LIMITS come from fig_dvar_daynight, so this figure and the main-text one
share them and can be compared panel for panel. --vmax overrides.

Usage:
  python fig_dvar_rcm.py <coast_geometry.nc>
        [--var all|DLR|LOWCF|MIDCF|Q2|PBLH] [--when all|day|night|diff]
        [--vmax 10] [--split-cbar] [--stipple] [--land-only]
        [--outdir figs]
"""
import argparse
import os

import numpy as np

import dvar_daynight as D
import wrfplot as wp

ROW_LABELS = ["R3", "R5", "R3$-$R5"]


# These supplements show the ocean, where the response is much larger
# than over land, so the main-text limits clip badly here. The submitted
# S10 runs to +/-10 W m-2 against the main figure's +/-5, which is this
# factor. With --land-only the main-text limit is used unchanged, so the
# two figures can then be compared panel for panel.
OCEAN_WIDEN = 2.0


def limit_for(spec, a, fields):
    """Colour limit: --vmax, else the main figure's, else from the data."""
    if a.vmax is not None:
        return a.vmax
    if not a.auto_limits and a.when in ("all", "day", "night"):
        base = D.DEFAULT_LIMITS[spec["cbar"]]
        return base if a.land_only else base * OCEAN_WIDEN
    v = np.nanpercentile(np.abs(np.concatenate(
        [f.ravel() for f in fields])), 99)
    return D.nice_round(float(v))


def one_figure(a, spec, land, lat, lon, keep):
    key = spec["key"]
    print(f"\n=== {key}, {D.WHEN_TITLE[a.when]}")

    # (row, column) -> member-mean anomaly map. R3 and R5 are loaded
    # separately and differenced here, so the third row is a difference
    # of anomalies rather than of raw fields.
    cols = []
    for run, title in zip(wp.SST_RUNS, wp.SST_TITLES):
        r3 = D.var_stack(spec, run, a.when, ["R3"])
        r5 = D.var_stack(spec, run, a.when, ["R5"])
        a3 = np.asarray(r3).mean(axis=0)
        a5 = np.asarray(r5).mean(axis=0)
        # keep the xarray stacks for stippling: agreement_mask needs the
        # named 'member' dimension, which np.asarray throws away
        cols.append(dict(title=title, maps=[a3, a5, a3 - a5],
                         stacks=[r3, r5, None]))
        print(f"  {title:<16s} R3 {np.nanmean(np.where(keep, a3, np.nan)):+8.3f}"
              f"   R5 {np.nanmean(np.where(keep, a5, np.nan)):+8.3f}"
              f"   R3-R5 {np.nanmean(np.where(keep, a3 - a5, np.nan)):+8.3f}")

    top = [c["maps"][r] for c in cols for r in (0, 1)]
    bot = [c["maps"][2] for c in cols]
    vtop = limit_for(spec, a, top)
    vbot = limit_for(spec, a, bot) if a.split_cbar else vtop
    print(f"  colour limit +/-{vtop:g} {spec['unit']}"
          + (f", difference row +/-{vbot:g}" if a.split_cbar else
             " (all three rows)"))

    clipped = 100.0 * np.mean([np.nanmean(np.abs(f) > vtop) for f in top])
    if clipped > 2.0:
        print(f"  WARNING: {clipped:.1f}% of the top two rows is beyond the "
              f"colour limit;\n           the panels are hiding part of the "
              f"response. Consider --vmax.")

    fig, axs = wp.panel_grid(
        3, len(cols),
        col_labels=[c["title"] for c in cols],
        row_labels=ROW_LABELS,
        suptitle=(None if a.no_title else
                  f"{spec['label']}  ({D.WHEN_TITLE[a.when]}, March 2011)"))

    handles = [None, None]
    for j, c in enumerate(cols):
        for r in range(3):
            v = vbot if r == 2 else vtop
            st = c["stacks"][r]
            # Agreement is within one configuration, so it is 4 of 5, not
            # 8 of 10. The difference row gets none: there is no ensemble
            # of scheme differences to agree about.
            mask = None
            if a.stipple and st is not None:
                mask = np.asarray(wp.agreement_mask(st, fraction=0.8))
            p = wp.draw(axs[r, j], lon, lat,
                        np.where(keep, c["maps"][r], np.nan) if a.land_only
                        else c["maps"][r],
                        -v, v, cmap=a.cmap,
                        label=chr(97 + r * len(cols) + j),
                        land=land, stipple=mask)
            handles[1 if r == 2 else 0] = p

    lab = f"{spec['label']} ({spec['unit']})"
    if a.split_cbar:
        wp.cbar(fig, handles[0], list(axs[:2, :].ravel()), lab)
        wp.cbar(fig, handles[1], list(axs[2, :].ravel()),
                f"{spec['label']} difference ({spec['unit']})")
    else:
        wp.cbar(fig, handles[0], list(axs.ravel()), lab)

    name = f"dvar_rcm_{key.lower()}_{a.when}{'_land' if a.land_only else ''}"
    wp.save(fig, os.path.join(a.outdir, f"{name}.jpg"))


def main(a):
    land, lat, lon = wp.grid(a.coast)
    keep = land > 0.5

    which = ([s for s in D.VARS] if a.var == "all"
             else [s for s in D.VARS if s["key"] == a.var])
    if not which:
        raise SystemExit(f"ERROR: --var {a.var} unknown; choose from "
                         f"all, {', '.join(s['key'] for s in D.VARS)}")

    print(f"{D.WHEN_TITLE[a.when]}; R3 and R5 as 5 members each")
    print("row 3 is the difference of the two ANOMALIES, not of the raw "
          "fields")
    for spec in which:
        one_figure(a, spec, land, lat, lon, keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("--var", default="all",
                    help="all, or one of DLR LOWCF MIDCF Q2 PBLH")
    ap.add_argument("--when", default="night",
                    choices=["all", "day", "night", "diff"],
                    help="all reproduces the submitted S10-S14; night "
                         "matches the revised Figure 7")
    ap.add_argument("--vmax", type=float, default=None,
                    help="colour limit. Omitted, it is taken from "
                         "fig_dvar_daynight so this figure and the "
                         "main-text one share one scale")
    ap.add_argument("--auto-limits", action="store_true",
                    help="limits from the data instead of the shared ones")
    ap.add_argument("--split-cbar", action="store_true",
                    help="give the R3-R5 row its own scale and colour bar. "
                         "Say so in the caption if you use it")
    ap.add_argument("--cmap", default=D.DEFAULT_CMAP,
                    help="default puts positive in red")
    ap.add_argument("--stipple", action="store_true",
                    help="4-of-5 sign agreement within each configuration; "
                         "the difference row is never stippled")
    ap.add_argument("--land-only", action="store_true",
                    help="mask the ocean out; the submitted supplements "
                         "show both")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
