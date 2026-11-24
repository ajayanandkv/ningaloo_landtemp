#!/usr/bin/env python
"""
Reviewer 2 comment 3: horizontal temperature advection.

Reads the stage-5 files and draws one figure per period:

    rows     zonal term, meridional term, total
    columns  CNTL, ClimSST, CNTL - ClimSST
    land only

    ADV   = -( u dT/dx + v dT/dy )
    ADV_U = -( u dT/dx )
    ADV_V = -( v dT/dy )

POSITIVE means advection WARMS the cell. A sea breeze carrying cool
marine air inland therefore appears as NEGATIVE daytime advection along
the coast, and a weaker or later breeze in CNTL should appear as a
POSITIVE difference there -- less cooling than in ClimSST.

The split into zonal and meridional matters on this coast. The west
coast runs roughly north-south, so onshore flow there is mostly ZONAL
and the sea-breeze signature should sit in ADV_U. The south coast runs
east-west, so its onshore flow is MERIDIONAL and shows in ADV_V. Looking
only at the total would blur the two.

The two terms are drawn on the SAME colour scale as each other so they
can be compared by eye; the total gets its own, since it can be larger.

UNITS. Stage 5 writes K h-1. This is a temperature CHANGE per hour, and
a change of 1 K is a change of 1 degC exactly, so the figures are
labelled degC h-1 with NO conversion applied. Do not add 273.15.

SHORELINE MASK (--erode). The gradient at a land cell touching the sea
is taken across the land-sea temperature jump, which is several degrees
over one grid cell. That gives advection of order 10 degC h-1 in the
first land row -- far off any sensible colour scale -- and, because the
jump is smaller in CNTL than in ClimSST, the difference there is
positive by construction rather than by physics. The outermost land
rows are therefore dropped before plotting. The coastline and the grey
land background are still drawn from the full mask, so only the data
are trimmed.

Usage:
  python fig_advect.py <coast_geometry.nc> <CoastalLandMask_N12.nc>
        [--when day|night|all] [--erode 2] [--cmap RdYlBu_r]
        [--run CNTL] [--ref EXP-SSTClim] [--outdir figs]
"""
import argparse
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import wrfplot as wp

ADV_UNIT = "advect/{phy}/{run}/{date}/advect_d01_2011-03.nc"
NICE_STEPS = (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)

# Positive (warming) is RED, negative (cooling) is BLUE.
DEFAULT_CMAP = "RdYlBu_r"

# The difference column gets its OWN colour map. It is a different kind
# of quantity from the two mean panels beside it -- a change, not a
# rate -- and on a shared palette the eye reads the three panels as one
# continuous field. PuOr_r keeps warm = positive and is colour-blind
# safe. Try BrBG_r or PRGn_r if it clashes with the figure around it.
DEFAULT_DCMAP = "PuOr_r"

ALL_ROWS = [("ADV_U", "Zonal,  $-u\\,\\partial T/\\partial x$"),
            ("ADV_V", "Meridional,  $-v\\,\\partial T/\\partial y$"),
            ("ADV", "Total")]

# --terms total keeps only the total row, which is what the manuscript
# shows; the component split stays available for the response letter.
ROWS = list(ALL_ROWS)

WHEN_TITLE = {"day": "Daytime", "night": "Nighttime", "all": "All-hours"}


def load_mask(path, var="auto"):
    with xr.open_dataset(path) as ds:
        if var == "auto":
            var = [v for v in ds.data_vars if ds[v].ndim == 2][0]
        m = ds[var].isel(**wp.CROP).values
    band = np.isfinite(m) & (np.nan_to_num(m) > 0.5)
    print(f"coastal band '{var}': {band.sum()} grid points after crop")
    return band


def shrink_land(land, n, ocean=False):
    """
    The cells whose advection is trustworthy.

    Land only by default. With ocean=True the whole domain is kept, as
    in the supplementary version of Figure 4 that includes the ocean.

    Either way the cells next to the coastline are dropped, because
    their gradient stencil straddles the land-sea temperature jump and
    the result is dominated by that rather than by any atmospheric
    process. With the ocean included the jump is seen from BOTH sides,
    so the band is removed from the sea as well as the land.
    """
    sea = land <= 0.5
    keep = np.ones_like(sea) if ocean else (land > 0.5)
    if n <= 0:
        print("shoreline mask: not applied (--erode 0)")
        return keep
    try:
        from scipy.ndimage import binary_erosion
    except ImportError:
        raise SystemExit(
            "ERROR: --erode needs scipy. Either install it or pass "
            "--erode 0 and say in the caption that the first land row "
            "is contaminated by the land-sea temperature jump.")
    # border_value=1 so the array edge is treated as interior. Without it
    # the domain boundary erodes too, and this domain's land runs to the
    # east and north edges -- those rows are fine and must be kept.
    if ocean:
        # strip the band from both sides of the coastline
        out = (binary_erosion(land > 0.5, iterations=n, border_value=1)
               | binary_erosion(sea, iterations=n, border_value=1))
        side = "either side of"
    else:
        out = binary_erosion(keep, iterations=n, border_value=1)
        side = "inland from"
    print(f"shoreline mask: dropped {int(keep.sum() - out.sum())} of "
          f"{int(keep.sum())} points ({n} row(s) {side} the coast)")
    return out


def unit_fields(ds, when, path):
    """
    The three advection fields for one unit, for the requested period.

    'all' is the hour-weighted combination of the stored day and night
    means, using the hour counts written by stage 5. Combining here is
    exact: the two periods partition the month and the weights are the
    number of hours that went into each.
    """
    if when in ("day", "night"):
        out = {}
        for k, _ in ROWS:
            v = f"{k}_{when}"
            if v not in ds:
                raise SystemExit(f"ERROR: '{v}' not in {path}\n"
                                 f"       file has: {sorted(ds.data_vars)}")
            out[k] = ds[v].isel(**wp.CROP).values
        return out

    try:
        nd = float(ds.attrs["n_day_hours"])
        nn = float(ds.attrs["n_night_hours"])
    except KeyError:
        raise SystemExit(
            f"ERROR: {path} has no n_day_hours / n_night_hours attribute, "
            f"so --when all cannot weight the two periods. Re-run "
            f"submit_sbday_s5, or use --when day / --when night.")
    out = {}
    for k, _ in ROWS:
        for v in (f"{k}_day", f"{k}_night"):
            if v not in ds:
                raise SystemExit(f"ERROR: '{v}' not in {path}\n"
                                 f"       file has: {sorted(ds.data_vars)}")
        d = ds[f"{k}_day"].isel(**wp.CROP).values
        n = ds[f"{k}_night"].isel(**wp.CROP).values
        out[k] = (nd * d + nn * n) / (nd + nn)
    return out


def members(run, when, a):
    """Every member's three advection fields: dict of (member, y, x)."""
    out = {k: [] for k, _ in ROWS}
    for phy in wp.PHYS:
        for date in wp.DATES:
            p = os.path.join(wp.DERIVED, a.template.format(
                phy=phy, run=run, date=date))
            if not os.path.exists(p):
                raise SystemExit(
                    f"ERROR: not found: {p}\n"
                    f"       has submit_sbday_s5 been run for "
                    f"{phy}/{run}/{date}?")
            with xr.open_dataset(p) as ds:
                f = unit_fields(ds, when, p)
            for k, _ in ROWS:
                out[k].append(f[k])
    return {k: np.stack(v) for k, v in out.items()}


def agreement(stack, fraction=0.8):
    """
    True where at least `fraction` of the members agree on the sign.

    wp.agreement_mask wants a DataArray with a 'member' dimension;
    members() here returns plain numpy stacks, so this is the same test
    on axis 0. Same 8-of-10 threshold as the rest of the paper.
    """
    n = stack.shape[0]
    thresh = int(np.ceil(fraction * n))
    return (((stack > 0).sum(axis=0) >= thresh) |
            ((stack < 0).sum(axis=0) >= thresh))


def land_only(f, land):
    return np.where(land, f, np.nan)


def nice_levels(field, what, pct=98, target=8, hard=None):
    v = np.asarray(field)
    f = v[np.isfinite(v)]
    if f.size == 0:
        return 1.0
    lo, hi = np.percentile(f, [100 - pct, pct])
    raw = hard if hard is not None else max(abs(lo), abs(hi))
    step = next((s for s in NICE_STEPS if raw / s <= target), NICE_STEPS[-1])
    lim = np.ceil(raw / step) * step
    print(f"  {what:<34s} data {lo:+.4f} to {hi:+.4f}, full {f.min():+.4f} "
          f"to {f.max():+.4f}  ->  scale +/-{lim:g}")
    return lim


def build_rows(a):
    """
    One entry per figure row: (label, term key, period).

    With --when both the rows are the two periods rather than the three
    terms, since day and night of a single term is what the manuscript
    figure shows and six rows would be unreadable.
    """
    if a.when == "both":
        return [("Daytime", "ADV", "day"), ("Nighttime", "ADV", "night")]
    terms = ALL_ROWS[-1:] if a.terms == "total" else ALL_ROWS
    return [(lab, key, a.when) for key, lab in terms]


def main(a):
    global ROWS
    # which fields have to be read: the union over the rows
    rows = build_rows(a)
    keys = list(dict.fromkeys(k for _, k, _ in rows))
    ROWS = [r for r in ALL_ROWS if r[0] in keys]

    band = load_mask(a.mask, a.mask_var)
    land, lat, lon = wp.grid(a.coast)
    land_data = shrink_land(land, a.erode, ocean=a.ocean)
    if a.ocean and a.vmax is None and a.dmax is None:
        print("NOTE: --ocean without --vmax/--dmax. Advection over the sea "
              "is much larger\n      than over land, so the colour scale "
              "will be set by the ocean and the\n      land response may "
              "look flat. Fix the limits to the land values if you\n"
              "      want the two comparable.")
    cmap = plt.get_cmap(a.cmap)
    dcmap = plt.get_cmap(a.dcmap)

    whens = list(dict.fromkeys(w for _, _, w in rows))
    R, F = {}, {}
    for w in whens:
        print(f"\nreading {WHEN_TITLE[w].lower()} advection")
        R[w] = members(a.run, w, a)
        F[w] = members(a.ref, w, a)
        print(f"  {R[w][keys[0]].shape[0]} members")

    print("\nband means, degC h-1 (shoreline rows excluded):")
    print(f"  {'row':<14s}{wp.rlab(a.run):>12s}{wp.rlab(a.ref):>12s}"
          f"{'difference':>13s}")
    panels = []
    for lab, key, when in rows:
        t = land_only(R[when][key].mean(axis=0), land_data)
        f = land_only(F[when][key].mean(axis=0), land_data)
        d = t - f
        # member-wise difference, so the sign test means what it says
        stip = (agreement(R[when][key] - F[when][key]) & land_data
                if not a.no_stipple else None)
        panels.append(dict(label=lab, key=key, when=when, run=t, ref=f,
                           dif=d, stip=stip))
        name = lab if a.when == "both" else key
        print(f"  {name:<14s}{np.nanmean(t[band]):>+12.4f}"
              f"{np.nanmean(f[band]):>+12.4f}{np.nanmean(d[band]):>+13.4f}")

    # Rows showing the SAME term share a colour scale, so day and night
    # can be read against each other. The zonal and meridional terms
    # share one with each other; the total gets its own, since it can
    # exceed either component.
    print("\ncolour scales:")
    groups = {}
    for p in panels:
        groups.setdefault("comp" if p["key"] != "ADV" else "total",
                          []).append(p)
    vlim, dlim = {}, {}
    for g, ps in groups.items():
        what = "zonal and meridional" if g == "comp" else "total"
        vlim[g] = nice_levels(
            np.concatenate([p[k].ravel() for p in ps for k in ("run", "ref")]),
            f"{what} (degC/h)", hard=a.vmax)
        dlim[g] = nice_levels(
            np.concatenate([p["dif"].ravel() for p in ps]),
            f"difference, {what} (degC/h)", hard=a.dmax)

    fig, axs = wp.panel_grid(
        len(panels), 3,
        row_labels=([""] * len(panels) if len(panels) == 1 else
                    [p["label"] for p in panels]),
        col_labels=[wp.rlab(a.run), wp.rlab(a.ref),
                    f"{wp.rlab(a.run)} $-$ {wp.rlab(a.ref)}"],
        suptitle=(f"March 2011 horizontal advection" if a.when == "both"
                  else f"March 2011 {WHEN_TITLE[a.when].lower()} "
                       f"horizontal advection"))

    k = 0
    handles = {}
    for i, p in enumerate(panels):
        g = "comp" if p["key"] != "ADV" else "total"
        for j, f in enumerate((p["run"], p["ref"])):
            ax = axs[i, j]
            if not a.ocean:
                ax.pcolormesh(lon, lat, np.where(land > 0.5, 1.0, 0.0),
                              cmap="Greys", vmin=0, vmax=4, shading="auto",
                              transform=ccrs.PlateCarree(), zorder=0)
            handles[("v", g)] = wp.draw(ax, lon, lat, f, -vlim[g], vlim[g],
                                        cmap=cmap, label=chr(97 + k),
                                        land=land)
            k += 1
        ax = axs[i, 2]
        if not a.ocean:
            ax.pcolormesh(lon, lat, np.where(land > 0.5, 1.0, 0.0),
                          cmap="Greys", vmin=0, vmax=4, shading="auto",
                          transform=ccrs.PlateCarree(), zorder=0)
        handles[("d", g)] = wp.draw(ax, lon, lat, p["dif"],
                                    -dlim[g], dlim[g], cmap=dcmap,
                                    label=chr(97 + k), land=land,
                                    stipple=p["stip"])
        k += 1

    deg = "$\\mathrm{^{\\circ}C}$ h$^{-1}$"
    for g, ps in groups.items():
        idx = [panels.index(p) for p in ps]
        frac = 0.08 if len(panels) <= 2 else 0.05
        what = "Advection, components" if g == "comp" else "Advection"
        wp.cbar(fig, handles[("v", g)], axs[idx, :2].ravel().tolist(),
                f"{what} ({deg})", orientation="horizontal",
                fraction=frac, pad=0.03, shrink=0.75)
        wp.cbar(fig, handles[("d", g)], axs[idx, 2].ravel().tolist(),
                f"Difference ({deg})", orientation="horizontal",
                fraction=frac, pad=0.03)

    tag = "_total" if a.terms == "total" and a.when != "both" else ""
    tag += "_ocean" if a.ocean else ""
    wp.save(fig, os.path.join(a.outdir, f"advection_{a.when}{tag}.jpg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("mask", help="CoastalLandMask_N12.nc")
    ap.add_argument("--when", default="day",
                    choices=["day", "night", "all", "both"],
                    help="day 07-18, night 19-06, all 24 hours, or both, "
                         "which puts daytime and nighttime in one figure "
                         "as two rows of the total term (default: day)")
    ap.add_argument("--terms", default="all", choices=["all", "total"],
                    help="total draws only the total advection row, as in "
                         "the manuscript; all adds the zonal and "
                         "meridional components (default: all)")
    ap.add_argument("--dcmap", default=DEFAULT_DCMAP,
                    help="colour map for the difference column only, so it "
                         "is not mistaken for a third mean panel")
    ap.add_argument("--ocean", action="store_true",
                    help="plot the whole domain instead of land only; the "
                         "band means still use the coastal land mask")
    ap.add_argument("--no-stipple", action="store_true",
                    help="drop the 8-of-10 sign-agreement hatching on the "
                         "difference column; on by default, as on the other "
                         "difference maps in the paper")
    ap.add_argument("--erode", type=int, default=2,
                    help="land rows to drop at the shoreline, where the "
                         "gradient straddles the coast; 0 keeps them "
                         "(default: 2)")
    ap.add_argument("--cmap", default=DEFAULT_CMAP,
                    help="colour map; the default puts warming in red and "
                         "cooling in blue. Use RdYlBu for the reverse.")
    ap.add_argument("--run", default="CNTL")
    ap.add_argument("--ref", default="EXP-SSTClim")
    ap.add_argument("--template", default=ADV_UNIT,
                    help="stage-5 file under DERIVED, with {phy} {run} "
                         "{date} placeholders")
    ap.add_argument("--mask-var", default="auto")
    ap.add_argument("--vmax", type=float, default=None,
                    help="colour limit for the mean fields, degC/h; omit to "
                         "pick a round number from the data")
    ap.add_argument("--dmax", type=float, default=None,
                    help="colour limit for the differences, degC/h")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
