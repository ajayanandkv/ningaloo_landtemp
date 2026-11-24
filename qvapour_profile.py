#!/usr/bin/env python
"""
Figure 9 of the manuscript, split into daytime and nighttime.

Vertical profile of the water-vapour mixing ratio difference between two
experiments, averaged over the WA coastal zone.

  --when all     reproduces the submitted figure, for cross-checking
  --when night   nighttime only, 20/23/02/05 AWST
  --when day     daytime only, 08/11/14/17 AWST
  --when both    DEFAULT. Day and night as two lines on one axes, which
                 is the figure worth publishing: the comparison is the
                 point, and two separate panels make the reader do the
                 subtraction by eye.

CROSS-CHECK FIRST. Run --when all and compare against the submitted
figure. The stage-3 file averages on MODEL levels and stores the mean
pressure of each level as the vertical axis. If the published figure
interpolated each column to fixed pressure levels before averaging,
expect small differences near the surface, where the levels are least
flat. Settle that before changing any number in the text.

WHY THE WINDOWS ARE ONLY FOUR SAMPLES EACH. wrfout is 3-hourly, so the
night mean is 20, 23, 02 and 05 AWST and the day mean 08, 11, 14 and 17.
These are cf3's windows exactly, so this profile and the cloud panels of
Figure 7 partition the month the same way. It is not an hourly mean and
should not be described as one.

THE SPREAD is across the ten members, each member's own band-mean
profile formed first. The shaded band is therefore ensemble spread, not
spatial spread across the coastal zone, which is much larger.

Usage:
  python fig_qvapor_profile.py <coast_geometry.nc> <CoastalLandMask_N12.nc>
        [--when both|all|day|night] [--rcm mean|R3|R5]
        [--run CNTL] [--ref EXP-SSTClim] [--pmin 400] [--spread sd|range]
        [--table] [--probe] [--outdir figs]
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr

import diurnal_cycle as DC
import wrfplot as wp

UNIT = "profile/{phy}/{run}/{date}/qvapor_profile_d01_2011-03.nc"

WHEN_TITLE = {"all": "All hours",
              "day": "Daytime (08, 11, 14, 17 AWST)",
              "night": "Nighttime (20, 23, 02, 05 AWST)"}

STYLE = {"day": dict(color="tab:red", ls="-", label="Daytime"),
         "night": dict(color="tab:blue", ls="--", label="Nighttime"),
         "all": dict(color="k", ls="-", label="All hours")}

G_PER_KG = 1000.0

_PBL_WARNED = []


def once_pbl():
    if not _PBL_WARNED:
        _PBL_WARNED.append(1)
        print("note: PBLTOP not in the stage-3 files, so the PBL top is "
              "not drawn.\n      Re-run submit_s3_profile to add it, or "
              "pass --pbl none to silence this.")


def path(phy, run, date):
    return os.path.join(wp.DERIVED, UNIT.format(phy=phy, run=run, date=date))


def read(phy, run, date, when):
    p = path(phy, run, date)
    if not os.path.exists(p):
        raise SystemExit(
            f"ERROR: profile file not found:\n       {p}\n"
            f"       Run submit_s3_profile on Setonix. No existing derived "
            f"product keeps\n       a vertical axis, so this figure cannot "
            f"be made without it.")
    with xr.open_dataset(p) as ds:
        qv, pv = f"QVAPOR_{when}", f"PRES_{when}"
        for v in (qv, pv):
            if v not in ds:
                raise SystemExit(f"ERROR: {v} not in {p}\n"
                                 f"       has: {sorted(ds.data_vars)}")
        q = ds[qv].isel(**wp.CROP).values * G_PER_KG
        pr = ds[pv].isel(**wp.CROP).values / 100.0        # Pa -> hPa
    return q, pr


def read_pbl(phy, run, date, when):
    """
    Band-ready PBL top (Pa) and PBL height (m) for one member.

    Returns (None, None) when the file predates the PBLTOP addition to
    stage 3, so an old derived tree still draws the profile without it
    rather than failing.
    """
    with xr.open_dataset(path(phy, run, date)) as ds:
        tv, hv = f"PBLTOP_{when}", f"PBLH_{when}"
        if tv not in ds:
            return None, None
        t = ds[tv].isel(**wp.CROP).values
        h = ds[hv].isel(**wp.CROP).values if hv in ds else None
    return t, h


def member_pbl(phys, run, when, band):
    """
    Mean PBL top over the coastal band, in hPa, and mean PBLH in m.

    The members are averaged after each one's band mean is taken, to
    match how the profiles themselves are formed.
    """
    tops, hs = [], []
    for phy in phys:
        for date in wp.DATES:
            t, h = read_pbl(phy, run, date, when)
            if t is None:
                return None, None
            tops.append(np.nanmean(t[band]) / 100.0)
            if h is not None:
                hs.append(np.nanmean(h[band]))
    return float(np.mean(tops)), (float(np.mean(hs)) if hs else None)


def member_profiles(phys, run, ref, when, band):
    """
    (member, level) anomaly profiles and the mean pressure axis.

    Each member's band mean is formed before the members are combined,
    so the spread across members means what it says.
    """
    dq, pr = [], []
    for phy in phys:
        for date in wp.DATES:
            qa, pa = read(phy, run, date, when)
            qb, _ = read(phy, ref, date, when)
            dq.append(np.nanmean((qa - qb)[:, band], axis=1))
            pr.append(np.nanmean(pa[:, band], axis=1))
    return np.stack(dq), np.stack(pr).mean(axis=0)


def load_band(mask_path, land):
    with xr.open_dataset(mask_path) as ds:
        v = [k for k in ds.data_vars if ds[k].ndim == 2][0]
        m = ds[v].isel(**wp.CROP).values
    band = np.isfinite(m) & (np.nan_to_num(m) > 0.5) & (land > 0.5)
    print(f"coastal zone '{v}': {int(band.sum())} land grid points")
    return band


def summarise(name, dq, pres, pmin):
    mu = dq.mean(axis=0)
    k = np.isfinite(mu) & (pres >= pmin)
    if not k.any():
        print(f"  {name}: nothing above {pmin:g} hPa")
        return
    i = int(np.nanargmax(np.abs(np.where(k, mu, np.nan))))
    # highest level at which the sign is still agreed by 8 of 10 members
    agree = (np.maximum((dq > 0).sum(axis=0), (dq < 0).sum(axis=0))
             >= int(np.ceil(0.8 * dq.shape[0])))
    top = pres[agree & k].min() if (agree & k).any() else np.nan
    print(f"  {name:<6s} surface {mu[0]:+.4f} g/kg, "
          f"peak {mu[i]:+.4f} at {pres[i]:.0f} hPa, "
          f"8/10 agreement up to {top:.0f} hPa")


def main(a):
    phys = list(wp.PHYS) if a.rcm == "mean" else [a.rcm]

    if a.probe:
        p = path(phys[0], a.ref, wp.DATES[0])
        print(f"DERIVED = {wp.DERIVED}\n{p}")
        if not os.path.exists(p):
            print("  MISSING -- run submit_s3_profile")
            return
        with xr.open_dataset(p) as ds:
            print(f"  dims {dict(ds.sizes)}")
            print(f"  vars {sorted(ds.data_vars)}")
            for k, v in ds.attrs.items():
                print(f"  {k}: {v}")
        return

    land, lat, lon = wp.grid(a.coast)
    band = load_band(a.mask, land)
    wins = ["day", "night"] if a.when == "both" else [a.when]
    nmem = len(phys) * len(wp.DATES)

    print(f"{wp.rlab(a.run)} minus {wp.rlab(a.ref)}, {a.rcm} "
          f"({nmem} members)\n")

    data = {}
    for w in wins:
        dq, pres = member_profiles(phys, a.run, a.ref, w, band)
        data[w] = (dq, pres)
        summarise(w, dq, pres, a.pmin)

    if a.when == "both":
        # The day and night profiles share a pressure axis to within the
        # thickness of a line, so differencing them level by level is
        # safe. Quote this rather than reading it off the figure.
        d, n = data["day"][0].mean(axis=0), data["night"][0].mean(axis=0)
        pres = data["day"][1]
        k = pres >= a.pmin
        i = int(np.nanargmax(np.abs(np.where(k, d - n, np.nan))))
        print(f"\n  day minus night: surface {d[0] - n[0]:+.4f} g/kg, "
              f"largest {d[i] - n[i]:+.4f} at {pres[i]:.0f} hPa")

    if a.table:
        print(f"\n{'hPa':>8s}" + "".join(f"{w:>12s}" for w in wins))
        pres = data[wins[0]][1]
        for k in range(len(pres)):
            if pres[k] < a.pmin:
                break
            print(f"{pres[k]:>8.1f}"
                  + "".join(f"{data[w][0].mean(axis=0)[k]:>12.4f}"
                            for w in wins))
        return

    fig, ax = plt.subplots(figsize=(6.0, 7.2))
    for w in wins:
        dq, pres = data[w]
        mu = dq.mean(axis=0)
        lo = dq.min(axis=0) if a.spread == "range" else mu - dq.std(axis=0)
        hi = dq.max(axis=0) if a.spread == "range" else mu + dq.std(axis=0)
        s = STYLE[w]
        ax.fill_betweenx(pres, lo, hi, color=s["color"], alpha=0.18, lw=0)
        ax.plot(mu, pres, lw=2.4, **s)

    ax.axvline(0, color="0.4", lw=0.9)

    # ---- PBL top ------------------------------------------------------
    # The concluding sentence of this paragraph is that the moistening is
    # confined to the boundary layer and the lower free troposphere. That
    # claim is only checkable if the reader can see where the boundary
    # layer ends, so the mean PBL top is drawn on the same pressure axis.
    # It is found by interpolating each model column's own pressure
    # against its own geopotential height at z = PBLH (stage 3), not from
    # an assumed scale height, so it is consistent with the profile.
    pbl_handles = []
    if a.pbl != "none":
        for w in wins:
            col = STYLE[w]["color"]
            for who, run, ls in (("ref", a.ref, ":"), ("run", a.run, "-.")):
                if a.pbl not in ("both", who):
                    continue
                top, hm = member_pbl(phys, run, w, band)
                if top is None:
                    once_pbl()
                    break
                ax.axhline(top, color=col, ls=ls, lw=1.5, alpha=0.85,
                           zorder=1)
                hs = f", {hm:.0f} m AGL" if hm is not None else ""
                print(f"  PBL top {w:<5s} {wp.rlab(run):<8s} "
                      f"{top:7.1f} hPa{hs}")
        if a.pbl == "both":
            pbl_handles = [
                Line2D([], [], color="0.35", ls=":", lw=1.5,
                       label=f"PBL top, {wp.rlab(a.ref)}"),
                Line2D([], [], color="0.35", ls="-.", lw=1.5,
                       label=f"PBL top, {wp.rlab(a.run)}")]
    ax.set_ylim(max(data[wins[0]][1]), a.pmin)      # pressure decreases up
    ax.set_xlabel("$\\Delta$Q$_{vapour}$ (g kg$^{-1}$)", fontsize=12)
    ax.set_ylabel("Pressure (hPa)", fontsize=12)
    ax.grid(alpha=0.25)
    if len(wins) > 1 or a.legend or pbl_handles:
        # keep the legend clear of the inset in the top right
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h + pbl_handles, fontsize=10, loc="lower left")
    if not a.no_title:
        # rlab gives the short names used in the manuscript text and in
        # every other figure's column labels ("ClimSST"). The submitted
        # Fig. 9 title used the raw run name ("EXP-SSTClim") instead;
        # --title-text sets it verbatim if that spelling is wanted back.
        ax.set_title(a.title_text or
                     (f"$\\Delta$QVAPOR vertical profile, "
                      f"{wp.rlab(a.run)} $-$ {wp.rlab(a.ref)}"), fontsize=13)

    sp = "full member range" if a.spread == "range" else "$\\pm$1 s.d."
    # Off by default. In the submitted version this note sat at the
    # bottom right and covered the lowest part of the profile, which is
    # where the largest difference is. The caption is the right place
    # for it; --note puts it back if a standalone figure needs it.
    NOTE_XY = {"left": (0.02, 0.50, "left"),
               "right": (0.98, 0.50, "right"),
               "bottom": (0.98, 0.02, "right")}
    if a.note != "none":
        x, y, ha = NOTE_XY[a.note]
        ax.text(x, y, f"shading: {sp} across {nmem} members",
                transform=ax.transAxes, ha=ha, va="top", fontsize=8,
                color="0.35")

    fig.tight_layout()

    # The inset goes on AFTER tight_layout: a hand-placed axes confuses
    # it. band_map comes from fig_diurnal_cycle, so the region is drawn
    # the same way in both figures and the two cannot disagree about
    # where the average was taken.
    if not a.no_inset:
        DC.band_map(fig, ax, a.coast, band, a.inset)
    tag = "rcmmean" if a.rcm == "mean" else a.rcm
    wp.save(fig, os.path.join(a.outdir, f"qvapor_profile_{a.when}_{tag}.jpg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("mask", help="CoastalLandMask_N12.nc")
    ap.add_argument("--when", default="both",
                    choices=["both", "all", "day", "night"],
                    help="both draws day and night together; all "
                         "reproduces the submitted figure")
    ap.add_argument("--rcm", default="mean", choices=["mean", "R3", "R5"])
    ap.add_argument("--run", default="CNTL")
    ap.add_argument("--ref", default=wp.REF_RUN)
    ap.add_argument("--pmin", type=float, default=400.0,
                    help="top of the plot in hPa; the submitted figure "
                         "stops around 600")
    ap.add_argument("--spread", default="sd", choices=["sd", "range"],
                    help="shading: one standard deviation across members, "
                         "or their full range")
    ap.add_argument("--pbl", default="both",
                    choices=["both", "ref", "run", "none"],
                    help="draw the mean PBL top as a horizontal line. "
                         "both (default) draws it for each run, so the "
                         "deepening is visible; ref or run draws one; "
                         "none drops it")
    ap.add_argument("--note", default="none",
                    choices=["none", "left", "right", "bottom"],
                    help="in-figure note naming what the shading is. "
                         "none (default) leaves it to the caption; "
                         "bottom reproduces the submitted placement, "
                         "which covered the lowest part of the profile")
    ap.add_argument("--legend", action="store_true")
    ap.add_argument("--title-text", default=None,
                    help="set the heading verbatim, e.g. "
                         '"$\\Delta$QVAPOR vertical profile, '
                         'CNTL $-$ EXP-SSTClim"')
    ap.add_argument("--no-title", action="store_true",
                    help="drop the heading; it is on by default, matching "
                         "the submitted figure")
    ap.add_argument("--no-inset", action="store_true",
                    help="drop the map of the averaging region")
    ap.add_argument("--inset", type=float, nargs=4,
                    default=[0.60, 0.60, 0.38, 0.38],
                    metavar=("X0", "Y0", "W", "H"),
                    help="inset position inside the axes, 0-1. Move it if "
                         "the profile runs under it")
    ap.add_argument("--table", action="store_true",
                    help="print the profile level by level instead of "
                         "drawing it")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
