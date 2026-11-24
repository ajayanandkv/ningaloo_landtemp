#!/usr/bin/env python
"""
Figure 4 of the manuscript, split into daytime and nighttime.

    rows     DLR, Low CF, Mid CF, Q2, PBLH
    columns  20SST, 40SST, 60SST, 80SST, CNTL   (each minus ClimSST)

Reviewer 2 comment 1 asks for the analysis to be partitioned into day and
night rather than presented as a daily mean. This script draws the same
five-variable panel as the submitted Figure 4 but for a chosen window.

  --when all     reproduces the submitted figure, for cross-checking
  --when day     07-18 AWST
  --when night   19-06 AWST

  --rcm mean     R3 and R5 pooled, 10 members  (the submitted figure)
  --rcm R3       R3 only, 5 members
  --rcm R5       R5 only, 5 members

WHERE EACH ROW COMES FROM. DLR, Q2 and PBLH are hourly (hour=24), so the
day and night means are taken over the hour axis. The cloud fractions are
NOT hourly: the derived files already hold cloud_fraction_day and
cloud_fraction_night with a leading band axis (0 low, 1 mid). Day is 07-18
and night 19-06, twelve hours each, so the all-hours cloud fraction is the
plain average of the two and --when all is exact, not an approximation.

Differences are formed member by member against ClimSST and only then
averaged, so a sign-agreement mask is available (--stipple). The submitted
Figure 4 has no stippling, so it is off by default and the cross-check
matches.

Colour limits are taken from the data across all five columns, which is
what makes the columns comparable. Override them to match the submitted
figure exactly with, for example:

    --limits DLR=5,CF=8,Q2=0.5,PBLH=60

LAND ONLY by default, as in the submitted Figure 4. --ocean keeps the
water too, which is the submitted Fig. S9, and widens the fixed colour
limits to suit: the SSTs there are prescribed, so the ocean response is
far larger than the land response and a land-tuned scale saturates every
ocean point. The two go to different filenames, since they are different
figures in the paper.

Usage:
  python fig_dvar_daynight.py <coast_geometry.nc> [--when all|day|night]
        [--rcm mean|R3|R5] [--ocean] [--limits DLR=5,CF=8] [--stipple]
        [--probe] [--outdir figs]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import wrfplot as wp

# key, label, colourbar group, unit, scale, how to read it
VARS = [
    dict(key="DLR",   var="GLW",  kind="hourly", cbar="DLR",
         label="$\\Delta$DLR",    unit="W m$^{-2}$",     scale=1.0),
    dict(key="LOWCF", band=0,     kind="cloud",  cbar="CF",
         label="$\\Delta$Low CF", unit="%",              scale=100.0),
    dict(key="MIDCF", band=1,     kind="cloud",  cbar="CF",
         label="$\\Delta$Mid CF", unit="%",              scale=100.0),
    dict(key="Q2",    var="Q2",   kind="hourly", cbar="Q2",
         label="$\\Delta$Q2",     unit="g kg$^{-1}$",    scale=1000.0),
    dict(key="PBLH",  var="PBLH", kind="hourly", cbar="PBLH",
         label="$\\Delta$PBLH",   unit="m",              scale=1.0),
]

# which rows share a colourbar, in drawing order
CBAR_ROWS = {"DLR": [0], "CF": [1, 2], "Q2": [3], "PBLH": [4]}

# Fixed symmetric limits, matching the submitted Figure 4. These are the
# default rather than data-driven limits for two reasons: the all-hours
# figure then reproduces the submitted one exactly, and day, night and
# all-hours share one scale so the three can be compared by eye. Pass
# --auto-limits to take them from the data instead.
DEFAULT_LIMITS = {"DLR": 5.0, "CF": 8.0, "Q2": 0.5, "PBLH": 60.0}

# With --ocean the limits above do not work: the response over the water
# is far larger than over land, because the SSTs there are prescribed, so
# a land-tuned scale saturates every ocean point. The submitted Fig. S9
# handles this with a wider range, which is this factor. It is applied
# only to the fixed defaults, never to --limits or --auto-limits, which
# were set deliberately.
OCEAN_WIDEN = 2.0

# Steps used to round the day-minus-night limits to a readable number.
NICE_STEPS = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0,
              10.0, 20.0, 25.0, 50.0, 100.0)

# Positive (warming, moistening, deeper, more cloud) is RED.
#DEFAULT_CMAP = "RdBu_r"
DEFAULT_CMAP = "seismic"

WHEN_TITLE = {"day": "Daytime (07-18 AWST)",
              "night": "Nighttime (19-06 AWST)",
              "all": "All hours",
              "diff": "Day minus night (07-18 minus 19-06 AWST)"}


def nice_round(x, target=5):
    """Round a colour limit up to a readable value."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    step = next((s for s in NICE_STEPS if x / s <= target), NICE_STEPS[-1])
    return float(np.ceil(x / step) * step)


def phys_for(rcm):
    return list(wp.PHYS) if rcm == "mean" else [rcm]


def parse_limits(text):
    """'DLR=5,CF=8' -> {'DLR': 5.0, 'CF': 8.0}."""
    out = {}
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"ERROR: --limits entry '{part}' is not NAME=VALUE")
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in CBAR_ROWS:
            raise SystemExit(f"ERROR: --limits name '{name}' unknown; "
                             f"use one of {sorted(CBAR_ROWS)}")
        out[name] = abs(float(val))
    return out


def cloud_stack(run, when, phys, scale):
    """
    Member-wise cloud-fraction anomaly, (member, band, y, x).

    The cloud files store the two windows as separate variables, so 'all'
    is built here. Day and night are both 12 hours, so the plain mean of
    the two is the true 24-hour mean.
    """
    if when in ("day", "night"):
        return wp.stack_anomaly("cloud", f"cloud_fraction_{when}", run,
                                phys=phys, scale=scale)
    # 'all' reads the stored daily field, which is what the submitted
    # Figure 4 used, so the cross-check compares like with like. Day and
    # night are 12 h each, so (day+night)/2 should equal it; check_split()
    # below reports the discrepancy rather than assuming it.
    return wp.stack_anomaly("cloud", "cloud_fraction_daily", run,
                            phys=phys, scale=scale)


def check_split(run, phys, scale=100.0):
    """
    Does (day + night)/2 reproduce the stored daily cloud fraction?

    Day is 07-18 and night 19-06, twelve hours each, so it should. A
    non-zero answer means the three fields were not built from the same
    hours, and the day/night figures would not be a clean partition of
    the submitted one.
    """
    def m(v):
        return np.asarray(wp.stack_anomaly("cloud", v, run, phys=phys,
                                           scale=scale).mean(dim="member"))
    resid = m("cloud_fraction_daily") - 0.5 * (m("cloud_fraction_day")
                                               + m("cloud_fraction_night"))
    worst = np.nanmax(np.abs(resid))
    print(f"\ncloud split check ({run}): "
          f"max |daily - (day+night)/2| = {worst:.4g} %")
    if worst > 0.05:
        print("  WARNING: the day and night cloud fields do not average "
              "back to the daily field.\n           The windows used to "
              "build them may differ. Check stage 1 before\n           "
              "trusting the day/night cloud panels.")
    else:
        print("  day and night partition the daily field cleanly")


def var_stack(spec, run, when, phys):
    """Member-wise anomaly stack for one row and one column, (member, y, x)."""
    if when == "diff":
        # Difference of the two ANOMALIES, not of the two raw fields:
        #   (day CNTL - day Clim) - (night CNTL - night Clim)
        # Positive therefore means the response to the SSTs is larger by
        # day than by night. Taken member by member, so the sign-agreement
        # mask still applies.
        return (var_stack(spec, run, "day", phys)
                - var_stack(spec, run, "night", phys))
    if spec["kind"] == "cloud":
        st = cloud_stack(run, when, phys, spec["scale"])
        # expect (member, band, y, x). Guard it: slicing axis 1 on a file
        # without a band axis would silently take a row of the map.
        if st.ndim != 4:
            raise SystemExit(
                f"ERROR: cloud_fraction_{when} stacked to dims {st.dims}; "
                f"expected (member, band, y, x). Run --probe to see what "
                f"the file actually holds.")
        return st.isel({st.dims[1]: spec["band"]})
    group = wp.resolve_group(spec["var"], phys=phys)
    return wp.stack_anomaly_daynight(group, spec["var"], run, when,
                                     how="mean", phys=phys,
                                     scale=spec["scale"])


def load_band(path, land, var="auto"):
    """The averaging region: the coastal mask if given, else all land."""
    if path is None:
        print("no --mask given: averaging over ALL land points in the domain")
        return land > 0.5
    with xr.open_dataset(path) as ds:
        if var == "auto":
            var = [v for v in ds.data_vars if ds[v].ndim == 2][0]
        m = ds[var].isel(**wp.CROP).values
    band = np.isfinite(m) & (np.nan_to_num(m) > 0.5) & (land > 0.5)
    print(f"averaging over '{var}': {int(band.sum())} coastal land points")
    return band


def table(phys, band):
    """
    Area means per variable, window and experiment.

    Each member's band mean is formed first, so the spread and the
    sign-agreement count are across members and mean what they say. This
    is the numeric backing for the day/night paragraph -- do not read
    these off the maps.
    """
    nmem = len(phys) * len(wp.DATES)
    windows = ["all", "day", "night", "diff"]
    print(f"\nArea means over {int(band.sum())} grid points, {nmem} members.")
    print("Each cell: mean +/- spread across members, then how many "
          "members agree on the sign.")
    print("'diff' is day minus night, so positive means a larger response "
          "by day.\n")

    for spec in VARS:
        print(f"{spec['key']}  ({spec['unit']})")
        print("  " + f"{'window':<9s}"
              + "".join(f"{t:>23s}" for t in wp.SST_TITLES))
        cache = {}
        for when in windows:
            cells = []
            for run in wp.SST_RUNS:
                if when == "diff":
                    st = cache[("day", run)] - cache[("night", run)]
                else:
                    st = var_stack(spec, run, when, phys)
                    if when in ("day", "night"):
                        cache[(when, run)] = st
                per = np.array([np.nanmean(np.where(band, v, np.nan))
                                for v in np.asarray(st)])
                pos, neg = int((per > 0).sum()), int((per < 0).sum())
                cells.append(f"{np.nanmean(per):+.3f}+/-{np.nanstd(per):.3f}"
                             f" {max(pos, neg)}/{len(per)}")
            print("  " + f"{when:<9s}" + "".join(f"{c:>23s}" for c in cells))
        print()


def probe(phys):
    """Report what is on disk before drawing anything."""
    print("=== probe ===")
    print(f"DERIVED = {wp.DERIVED}")
    print(f"members = {len(phys)} RCM(s) x {len(wp.DATES)} dates "
          f"= {len(phys) * len(wp.DATES)}")
    ok = True
    for spec in VARS:
        if spec["kind"] == "cloud":
            p = wp.unit_path("cloud", phys[0], wp.REF_RUN, wp.DATES[0])
            print(f"\n{spec['key']:<6s} group 'cloud'  band {spec['band']}")
            print(f"       {p}")
            if not os.path.exists(p):
                print("       MISSING")
                ok = False
                continue
            with xr.open_dataset(p) as ds:
                for v in ("cloud_fraction_day", "cloud_fraction_night"):
                    if v in ds:
                        print(f"       {v}: dims {ds[v].dims} "
                              f"shape {ds[v].shape}")
                    else:
                        print(f"       {v}: NOT PRESENT")
                        ok = False
                print(f"       file holds: {sorted(ds.data_vars)}")
            continue

        print(f"\n{spec['key']:<6s} variable '{spec['var']}'")
        try:
            g = wp.resolve_group(spec["var"], phys=phys)
        except KeyError as e:
            print(f"       NOT FOUND: {e}")
            ok = False
            continue
        p = wp.unit_path(g, phys[0], wp.REF_RUN, wp.DATES[0])
        print(f"       group '{g}'")
        print(f"       {p}")
        if not os.path.exists(p):
            print("       MISSING")
            ok = False
            continue
        with xr.open_dataset(p) as ds:
            da = ds[spec["var"]]
            print(f"       dims {da.dims} shape {da.shape}")
            if "hour" not in da.dims:
                print("       NO 'hour' DIMENSION -- day/night split "
                      "is not possible for this variable")
                ok = False
    print("\n=== probe", "OK" if ok else "FAILED", "===")
    if not ok:
        raise SystemExit("Fix the missing inputs above before plotting.")


def main(a):
    phys = phys_for(a.rcm)
    if a.probe:
        probe(phys)
        return

    land, lat, lon = wp.grid(a.coast)

    if a.table:
        print(f"{a.rcm}: {len(phys)} RCM(s) x {len(wp.DATES)} dates")
        table(phys, load_band(a.mask, land))
        return

    overrides = parse_limits(a.limits)
    cmap = plt.get_cmap(a.cmap)
    nmem = len(phys) * len(wp.DATES)

    print(f"{WHEN_TITLE[a.when]}, {a.rcm} ({nmem} members)")
    if a.when == "all":
        check_split(wp.SST_RUNS[-1], phys)      # CNTL, the strongest column

    # fields[row][col] = 2-D anomaly; masks likewise
    fields, masks = {}, {}
    for i, spec in enumerate(VARS):
        print(f"\n{spec['key']}")
        for j, run in enumerate(wp.SST_RUNS):
            st = var_stack(spec, run, a.when, phys)
            # Land only by default, as in the submitted figure: the ocean
            # response is much larger and would set the colour scale for
            # everything. --ocean keeps it, and widens the limits to suit.
            f = np.asarray(st.mean(dim="member"))
            if not a.ocean:
                f = np.where(land > 0.5, f, np.nan)
            fields[(i, j)] = f
            masks[(i, j)] = (np.asarray(wp.agreement_mask(st))
                             if a.stipple else None)
            print(f"  {wp.SST_TITLES[j]:<18s} "
                  f"{np.nanmin(f):+9.3f} to {np.nanmax(f):+9.3f} "
                  f"{spec['unit']}")

    # one symmetric limit per colourbar group, over all five columns
    print("\ncolour limits:")
    lims = {}
    for name, rows in CBAR_ROWS.items():
        vals = np.concatenate([np.asarray(fields[(i, j)]).ravel()
                               for i in rows for j in range(len(wp.SST_RUNS))])
        vals = vals[np.isfinite(vals)]
        p98 = float(np.percentile(np.abs(vals), 98)) if vals.size else 1.0
        peak = float(np.nanmax(np.abs(vals))) if vals.size else 1.0

        if name in overrides:
            lims[name], why = overrides[name], "set by --limits"
        elif a.auto_limits:
            lims[name], why = p98, "98th pct of |data|"
        elif a.when == "diff":
            # The difference is smaller than the fields, so the Figure 4
            # limits would flatten it. Take a readable value from the data
            # and print it so it can be pinned for the final version.
            lims[name], why = nice_round(p98), "from data, rounded"
        elif a.ocean:
            lims[name] = DEFAULT_LIMITS[name] * OCEAN_WIDEN
            why = f"fixed x{OCEAN_WIDEN:g}, as submitted Fig. S9"
        else:
            lims[name], why = DEFAULT_LIMITS[name], "fixed, as submitted Fig. 4"

        clipped = 100.0 * np.mean(np.abs(vals) > lims[name]) if vals.size else 0.0
        where = "all" if a.ocean else "land"
        print(f"  {name:<5s} +/-{lims[name]:<6g} ({why})"
              f"   data: 98th pct {p98:.4g}, peak {peak:.4g}, "
              f"{clipped:.2f}% of {where} points clipped")

    fig, axs = wp.panel_grid(
        len(VARS), len(wp.SST_RUNS),
        row_labels=[s["label"] for s in VARS],
        col_labels=list(wp.SST_TITLES),
        suptitle=(f"{WHEN_TITLE[a.when]} - {a.rcm} ({nmem} members)"
                  if a.title else None))

    handles = {}
    for i, spec in enumerate(VARS):
        v = lims[spec["cbar"]]
        for j in range(len(wp.SST_RUNS)):
            p = wp.draw(axs[i, j], lon, lat, fields[(i, j)], -v, v,
                        cmap=cmap,
                        label=chr(97 + i * len(wp.SST_RUNS) + j),
                        stipple=masks[(i, j)], land=land)
            handles[spec["cbar"]] = p

    for name, rows in CBAR_ROWS.items():
        unit = next(s["unit"] for s in VARS if s["cbar"] == name)
        cax = [axs[i, j] for i in rows for j in range(len(wp.SST_RUNS))]
        wp.cbar(fig, handles[name], cax, unit, fraction=0.02, pad=0.015)

    if a.when == "diff" and not overrides and not a.auto_limits:
        pin = ",".join(f"{k}={v:g}" for k, v in lims.items())
        print(f"\nto pin these limits across all three RCM runs:\n"
              f"  --limits {pin}")

    tag = "rcmmean" if a.rcm == "mean" else a.rcm
    # the suffix keeps the land-only and with-ocean versions apart, since
    # they are different figures in the paper (Fig. 7 and Fig. S9)
    wp.save(fig, os.path.join(
        a.outdir, f"dvar_{a.when}_{tag}{'_ocean' if a.ocean else ''}.jpg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("--when", default="all",
                    choices=["all", "day", "night", "diff"],
                    help="all reproduces the submitted Figure 4 (default); "
                         "diff is the daytime anomaly minus the nighttime "
                         "anomaly, so positive means a larger response by day")
    ap.add_argument("--rcm", default="mean", choices=["mean", "R3", "R5"])
    ap.add_argument("--limits", default=None,
                    help="override the fixed colour limits, e.g. "
                         "DLR=5,CF=8,Q2=0.5,PBLH=60")
    ap.add_argument("--auto-limits", action="store_true",
                    help="take the limits from the data instead of the "
                         "fixed defaults; day, night and all-hours then "
                         "no longer share a scale")
    ap.add_argument("--cmap", default=DEFAULT_CMAP,
                    help="colour map; the default puts positive in red, "
                         "as in the submitted figure")
    ap.add_argument("--ocean", action="store_true",
                    help="draw the ocean as well as the land, as in the "
                         "submitted Fig. S9. The fixed colour limits are "
                         f"widened by x{OCEAN_WIDEN:g} to suit, since the "
                         "prescribed SSTs make the ocean response much "
                         "larger; --limits and --auto-limits are left alone")
    ap.add_argument("--stipple", action="store_true",
                    help="hatch where at least 80%% of members agree on the "
                         "sign; off by default, as in the submitted figure")
    ap.add_argument("--title", action="store_true",
                    help="add a suptitle; off by default, as in the "
                         "submitted figure, which also saves the strip "
                         "of white space it would need")
    ap.add_argument("--table", action="store_true",
                    help="print area means for every variable, window and "
                         "experiment instead of drawing")
    ap.add_argument("--mask", default=None,
                    help="CoastalLandMask_N12.nc, the region --table "
                         "averages over; without it, all land is used")
    ap.add_argument("--probe", action="store_true",
                    help="report the input files and variable dimensions, "
                         "then stop")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
