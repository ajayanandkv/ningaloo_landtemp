#!/usr/bin/env python
"""
When during the day does the extra rain or cloud appear? Day by day.

Reviewer 2 asks whether the extra rain and cloud occur at night, which
supports the nighttime longwave argument, or at dawn, which does not.
The published figures cannot answer it: they show a March total or a
single March-mean cycle, and a month in which every event fell at dawn
looks the same as a month spread evenly.

Replaces fig_rain_dayhour.py, which did rain only.

VARIABLES

  --var rain       hourly, 24 points      stage-3 rain_dayhour files
  --var low        3-hourly, 8 points     stage-2 cloud_dayhour files
  --var mid        3-hourly, 8 points
  --var high       3-hourly, 8 points

WHICH CLOUD BAND. The files carry wrf-python's three bands: 0 is surface
to 970 hPa, 1 is 970 to 800, 2 is 800 to 450. The manuscript's "Low CF,
970-800" and "Mid CF, 800-450" are therefore bands 1 and 2, while the
map figures plot bands 0 and 1. --var low/mid/high here name the
MANUSCRIPT's bands, so low is band 1 and mid is band 2. --raw-bands
switches to wrf-python's numbering, for checking against the map
figures. Settle which the paper means before quoting anything from this.

EIGHT SAMPLES IS NOT AN HOURLY CYCLE. wrfout is 3-hourly, so the cloud
panels have eight points at 02, 05, 08, 11, 14, 17, 20 and 23 AWST. They
are drawn as markers joined by straight lines, deliberately, so they are
not mistaken for a resolved diurnal cycle.

MODES

  --mode all      every complete day in one figure
                    (a) one faint line per day, with the month mean bold
                    (b) the same data as a day-by-step map, which is what
                        settles the timing question
                    (c) the box, on the March difference field

  --mode events   only the days the paper names, one panel each, with
                  CNTL and ClimSST drawn separately rather than
                  differenced, plus the box map as a final panel

WHERE TO AVERAGE. Three choices.

  --select zone  DEFAULT. The coastal zone, exactly as defined in the
                 Methods and used for every other area mean in the paper.
                 One region, fixed, nothing to justify to a reviewer.
                 Needs the coastal mask file as a second argument.

  --select wet   coastal-zone grid points at which the DAILY rain total
                 passes --wet-thresh, recomputed for each day. Sharper on
                 the event days, but it is a conditional composite and
                 needs the care set out below.

  --select box   a fixed rectangle, from --box or placed by --auto-box.
                 --auto-box takes one or two sizes in grid points:
                 `--auto-box 30 50` is 30 in latitude by 50 in longitude.
                 It PRINTS the corners as a --box line; pin them once
                 settled or the box moves when anything upstream changes.

THE SELECTION TRAP, which applies to --select wet only. If
CNTL alone defines which points are wet, the average runs over points
chosen for raining in CNTL, and CNTL minus ClimSST is positive by
construction. The same in reverse for ClimSST.

  --wet-on both      DEFAULT. Points wet in BOTH runs. The comparison is
                     then conditioned on something neither run alone
                     decided, so a difference in TIMING is a real result.
  --wet-on either    points wet in either run: more samples, but it
                     reintroduces the bias, since the extra points are
                     mostly wet in one run only.
  --wet-on run/ref   one run decides. For diagnosis only. The amount
                     difference from these is not interpretable.

Selection is always on RAIN, even when a cloud variable is plotted,
because the question is when cloud appeared on the days it rained. So
--var mid --select wet reads both the cloud and the rain files.

Days on which no point passes are left out and reported. Most of March
will drop; that is the answer to a different question the paper already
answers, namely that the response is two events.

AN AREA MEAN IS NOT AN EVENT TOTAL. The table prints the mean over the
selected points, the maximum among them, and how many there were.

TIMING, for rain: hour h is the rain that fell between h-1 and h AWST.
A peak at 07 is DAWN, not night.

Usage:
  python fig_dayhour.py <coast_geometry.nc> [CoastalLandMask_N12.nc]
        [--var rain|low|mid|high] [--mode all|events]
        [--select zone|wet|box] [--wet-thresh 5] [--wet-on both|either|run|ref]
        [--days 4,5,14,15,16] [--box -35 -31 114 119] [--auto-box 30 50]
        [--run CNTL] [--ref EXP-SSTClim] [--rcm mean|R3|R5]
        [--probe] [--outdir figs]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import diurnal_cycle as DC
import wrfplot as wp

RAIN_UNIT = "rain_dayhour/{phy}/{run}/{date}/rain_dayhour_d01_2011-03.nc"
CLOUD_UNIT = "cloud_dayhour/{phy}/{run}/{date}/cloud_dayhour_d01_2011-03.nc"

# band: index into low_mid_high. 'paper' is what the manuscript calls it,
# 'raw' is wrf-python's own numbering.
VARS = {
    # TO ADD A VARIABLE: copy a line below and change it. Nothing else in
    # the script needs touching, as long as the file it comes from is one
    # of the two already read AND is shaped (day, hour, y, x).
    #   src      which file: "rain" or "cloud"
    #   var      the netCDF variable name (rain only; default PREC_ACC_TOT)
    #   paper    band index in the manuscript's naming (cloud only)
    #   raw      band index in wrf-python's naming (cloud only)
    #   label    axis and colour-bar text
    #   unit     printed after the label, LaTeX allowed
    #   scale    multiplier into display units (1000 for kg/kg -> g/kg)
    #   cmap     diverging colour map for the day-by-hour panel
    #   submit   the Setonix script named in the "file not found" error
    "rain": dict(src="rain", var="PREC_ACC_TOT", label="rain",
                 unit="mm h$^{-1}$", tot="mm", scale=1.0, cmap="BrBG",
                 submit="submit_rain_s3"),
    "rain_nc": dict(src="rain", var="PREC_ACC_NC", label="rain, grid-scale",
                    unit="mm h$^{-1}$", tot="mm", scale=1.0, cmap="BrBG",
                    submit="submit_rain_s3"),
    "rain_c": dict(src="rain", var="PREC_ACC_C", label="rain, convective",
                   unit="mm h$^{-1}$", tot="mm", scale=1.0, cmap="BrBG",
                   submit="submit_rain_s3"),
    "low":  dict(src="cloud", paper=1, raw=0, label="Low CF",
                 unit="%", tot="%", scale=100.0, cmap="BrBG",
                 submit="submit_s2_dayhour"),
    "mid":  dict(src="cloud", paper=2, raw=1, label="Mid CF",
                 unit="%", tot="%", scale=100.0, cmap="BrBG",
                 submit="submit_s2_dayhour"),
    "high": dict(src="cloud", paper=2, raw=2, label="High CF",
                 unit="%", tot="%", scale=100.0, cmap="BrBG",
                 submit="submit_s2_dayhour"),
}

NIGHT = (19, 24, 0, 7)          # 19-23 and 00-06 AWST
DAWN = (5, 8)                   # 05-07 AWST, inside night


def path(tmpl, phy, run, date):
    return os.path.join(wp.DERIVED, tmpl.format(phy=phy, run=run, date=date))


def load(spec, raw_bands, phy, run, date):
    """(dates, hours, (day, hour, y, x), (y, x) March difference field)."""
    tmpl = RAIN_UNIT if spec["src"] == "rain" else CLOUD_UNIT
    p = path(tmpl, phy, run, date)
    if not os.path.exists(p):
        raise SystemExit(
            f"ERROR: per-day file not found:\n       {p}\n"
            f"       Run {spec['submit']} on Setonix. The existing stage-1 "
            f"files hold only\n       a March mean and cannot answer this "
            f"question.")
    with xr.open_dataset(p) as ds:
        if spec["src"] == "rain":
            v = spec.get("var", "PREC_ACC_TOT")
            if v not in ds:
                raise SystemExit(f"ERROR: no '{v}' in {p}\n"
                                 f"       has: {sorted(ds.data_vars)}")
            da = ds[v]
        else:
            if "cloud_fraction" not in ds:
                raise SystemExit(
                    f"ERROR: no 'cloud_fraction' in {p}\n"
                    f"       has: {sorted(ds.data_vars)}\n"
                    f"       This is the stage-1 day/night file, not "
                    f"stage 2.")
            b = spec["raw"] if raw_bands else spec["paper"]
            da = ds["cloud_fraction"].isel(low_mid_high=b)
        if "day" not in da.dims:
            raise SystemExit(f"ERROR: {p} has dims {da.dims} and no 'day' "
                             f"axis -- wrong stage.")
        arr = da.isel(**wp.CROP).values * spec["scale"]
        hours = np.asarray(ds["hour"].values, dtype=int)
        days = np.array([str(v)[:10] for v in ds["day"].values])
    # the "total" field the box is placed on: the month mean of the
    # variable itself, which for rain is proportional to the total
    return days, hours, arr, arr.mean(axis=(0, 1))


def stack(spec, raw_bands, phys, run):
    per, keys, hours = [], None, None
    for phy in phys:
        for date in wp.DATES:
            k, h, a, t = load(spec, raw_bands, phy, run, date)
            keys = k if keys is None else np.intersect1d(keys, k)
            hours = h if hours is None else hours
            per.append((k, a, t))
    arr = np.stack([a[np.isin(k, keys)] for k, a, _ in per])
    tot = np.stack([t for _, _, t in per])
    return keys, hours, arr, tot


def box_mask(lat, lon, box):
    la0, la1, lo0, lo1 = box
    return ((lat >= min(la0, la1)) & (lat <= max(la0, la1))
            & (lon >= min(lo0, lo1)) & (lon <= max(lo0, lo1)))


def load_mask(mask_path, land, why="--select zone"):
    """The coastal zone, the same mask every other area mean in the paper uses."""
    if mask_path is None:
        raise SystemExit(
            f"ERROR: {why} needs the coastal mask as a second argument,\n"
            f"       e.g.  fig_dayhour.py coast_geometry.nc "
            f"CoastalLandMask_N12.nc\n"
            f"       Pass --select box to use a rectangle instead.")
    with xr.open_dataset(mask_path) as ds:
        var = [v for v in ds.data_vars if ds[v].ndim == 2][0]
        m = ds[var].isel(**wp.CROP).values
    band = np.isfinite(m) & (np.nan_to_num(m) > 0.5) & (land > 0.5)
    print(f"coastal zone '{var}': {int(band.sum())} land grid points")
    return band


def wet_mask(rain_run, rain_ref, zone, thresh, how):
    """
    Per-day mask of coastal points whose DAILY rain total passes thresh.

    rain_* are (member, day, hour, y, x). The member mean is taken first,
    so the mask is one field per day rather than one per member, and the
    ensemble is averaged over the same points in both runs.

    Returns (day, y, x) boolean.
    """
    a = rain_run.mean(axis=0).sum(axis=1)          # (day, y, x), mm/day
    b = rain_ref.mean(axis=0).sum(axis=1)
    wa, wb = a > thresh, b > thresh
    sel = {"both": wa & wb, "either": wa | wb, "run": wa, "ref": wb}[how]
    sel = sel & zone[None, :, :]

    print(f"\nwet selection: daily rain > {thresh:g} mm, --wet-on {how}")
    print(f"  point-days passing in {wp.rlab('CNTL')} only: "
          f"{int((wa & ~wb & zone).sum())}, in the reference only: "
          f"{int((wb & ~wa & zone).sum())}, in both: "
          f"{int((wa & wb & zone).sum())}")
    if how in ("run", "ref"):
        print("  WARNING: one run defines the selection, so the AMOUNT "
              "difference is\n           biased by construction. Read the "
              "timing only.")
    return sel


def no_land_message(box, lat, lon, land):
    """
    Say WHY the box is empty, not just that it is.

    The usual cause is a dropped minus sign. Western Australia is in the
    southern hemisphere, so every latitude in this domain is negative,
    and `--box 31 33 ...` is a box in China. That is worth catching by
    name, because the bare message looks like a data problem.
    """
    la0, la1, lo0, lo1 = box
    lines = [f"ERROR: the box {box} contains no land grid points.",
             f"       domain covers "
             f"{float(lat.min()):.2f} to {float(lat.max()):.2f} N, "
             f"{float(lon.min()):.2f} to {float(lon.max()):.2f} E"]

    if min(la0, la1) > 0 and float(lat.max()) < 0:
        lines += ["",
                  "       Both latitudes are POSITIVE and this domain is "
                  "entirely in the",
                  "       southern hemisphere. Western Australia needs "
                  "negative latitudes.",
                  f"       Try:  --box {-max(la0, la1):g} "
                  f"{-min(la0, la1):g} {lo0:g} {lo1:g}"]
        return "\n".join(lines)

    if not box_mask(lat, lon, box).any():
        lines.append("       The box does not overlap the domain at all.")
    else:
        lines.append("       The box overlaps the domain but falls entirely "
                     "over ocean.")
    lines.append("       Omit --box to let --auto-box place one on the peak "
                 "and print its corners.")
    return "\n".join(lines)


def auto_box(dtot, lat, lon, land, ny, nx):
    f = np.where(land > 0.5, dtot, np.nan)
    if not np.isfinite(f).any():
        raise SystemExit("ERROR: no finite land points for --auto-box")
    j, i = np.unravel_index(np.nanargmax(f), f.shape)
    hy, hx = max(ny // 2, 1), max(nx // 2, 1)
    js = slice(max(j - hy, 0), min(j + hy + 1, f.shape[0]))
    iss = slice(max(i - hx, 0), min(i + hx + 1, f.shape[1]))
    sl, so = lat[js, iss], lon[js, iss]
    box = (float(sl.min()), float(sl.max()), float(so.min()), float(so.max()))
    print(f"--auto-box {ny} {nx}: peak at {lat[j, i]:.2f}, {lon[j, i]:.2f}")
    print(f"  pin it with:  --box {box[0]:.2f} {box[1]:.2f} "
          f"{box[2]:.2f} {box[3]:.2f}")
    return box


def hours_in(window):
    out = []
    for k in range(0, len(window), 2):
        out += list(range(window[k], window[k + 1]))
    return out


def reduce_sel(field, sel):
    """
    (day, hour, y, x) and a (day, y, x) mask -> (day, hour) mean and max.

    A day with no selected point gives NaN rather than zero, so it is
    dropped from the figure instead of being drawn as a flat line.
    """
    nd, nh = field.shape[0], field.shape[1]
    out = np.full((nd, nh), np.nan)
    mx = np.full((nd, nh), np.nan)
    npts = np.zeros(nd, dtype=int)
    for i in range(nd):
        m = sel[i]
        npts[i] = int(m.sum())
        if npts[i]:
            v = field[i][:, m]
            out[i] = np.nanmean(v, axis=1)
            mx[i] = np.nanmax(v, axis=1)
    return out, mx, npts


def table(spec, days, hours, dmean, dmax, npts=None):
    night = [i for i, h in enumerate(hours) if h in hours_in(NIGHT)]
    dayi = [i for i, h in enumerate(hours) if i not in night]
    dawn = [i for i, h in enumerate(hours) if h in hours_in(DAWN)]
    agg = "sum" if spec["src"] == "rain" else "mean"

    def red(v, idx):
        if not idx:
            return 0.0
        return float(np.nansum(v[idx]) if agg == "sum"
                     else np.nanmean(v[idx]))

    print(f"\n{'date':<12s}{'npts':>6s}{'peak hr':>8s}{'peak':>10s}"
          f"{'max':>10s}{'night':>9s}{'day':>9s}{'dawn':>9s}"
          f"    ({agg} over window)")
    tn = td = tw = 0.0
    nused = 0
    for i, d in enumerate(days):
        c = dmean[i]
        np_i = "-" if npts is None else str(npts[i])
        if not np.isfinite(c).any():
            print(f"{d:<12s}{np_i:>6s}{'':>8s}{'  no points selected':<30s}")
            continue
        nused += 1
        pk = int(np.nanargmax(np.abs(c)))
        n, dd, w = red(c, night), red(c, dayi), red(c, dawn)
        tn, td, tw = tn + n, td + dd, tw + w
        print(f"{d:<12s}{np_i:>6s}{hours[pk]:>8d}{c[pk]:>10.4f}"
              f"{np.nanmax(dmax[i]):>10.3f}{n:>9.3f}{dd:>9.3f}{w:>9.3f}")

    k = max(nused, 1) if agg == "mean" else 1
    print(f"\n{'TOTAL':<12s}{nused:>6d}{'':>8s}{'':>10s}{'':>10s}"
          f"{tn / k:>9.3f}{td / k:>9.3f}{tw / k:>9.3f}   "
          f"({'days used' if npts is not None else 'days'})")
    if agg == "sum" and abs(tn + td) > 1e-9:
        print(f"night share of the difference: "
              f"{100.0 * tn / (tn + td):.0f}%   "
              f"(0500-0700 alone: {100.0 * tw / (tn + td):.0f}%)")
    print(f"night = 19-23 and 00-06 AWST; dawn = 05-07, counted inside night")
    if spec["src"] == "rain":
        print("hour h is the rain that fell between h-1 and h AWST")
    else:
        print(f"3-hourly samples at {', '.join(f'{h:02d}' for h in hours)} "
              f"AWST -- eight points, not a resolved cycle")


def shade_night(ax, hours):
    ax.axvspan(hours[0] - 0.6, NIGHT[3] - 0.5, color="0.88", zorder=0)
    ax.axvspan(NIGHT[0] - 0.5, hours[-1] + 0.6, color="0.88", zorder=0)


def region_panel(fig, cell, spec, lat, lon, land, dtot, box, sel, label,
                 plain=True, name="Core rainfall zone"):
    """
    Where the average was taken.

    sel is None for --select box (a rectangle is drawn), 2-D for
    --select zone (the zone outlined) and 3-D for --select wet, where it
    becomes the COUNT of qualifying days per grid point. That last one
    is the honest picture for a composite: the region is not one shape,
    it moves with the weather, and the map shows how often each point
    took part.

    plain=True (the default) draws land only: no variable shaded
    underneath and no colour bar, so the panel says where the average
    was taken and nothing else. The March difference field is already a
    figure of its own in the paper, and a second, differently scaled
    copy of it here only invites the reader to compare the two.
    --region-field puts it back.
    """
    ax = fig.add_subplot(cell, projection=wp.ROTATED_POLE)
    if wp._features_ok():
        ax.coastlines()
        ax.add_feature(wp.cfeature.BORDERS, linewidth=0.5)

    # --select wet: the count map IS the panel, so it is drawn whether
    # or not a background field was asked for.
    if sel is not None and sel.ndim == 3:
        n = sel.sum(axis=0).astype(float)
        n[n == 0] = np.nan
        p = wp.draw(ax, lon, lat, np.where(land > 0.5, n, np.nan),
                    0, max(float(np.nanmax(n)) if np.isfinite(n).any()
                           else 1.0, 1.0), cmap="YlGnBu", land=land)
        ax.set_title(f"({label}) Days selected", fontsize=11, weight="bold")
        cb = fig.colorbar(p, ax=ax, fraction=0.040, pad=0.03, shrink=0.82)
        cb.set_label("days", fontsize=9)
        return ax

    if plain:
        ax.pcolormesh(lon, lat, np.where(land > 0.5, 1.0, np.nan),
                      cmap="Greys", vmin=0, vmax=4, shading="auto",
                      transform=wp.ccrs.PlateCarree(), zorder=0)
    else:
        v = float(np.nanpercentile(np.abs(dtot), 99)) or 1.0
        p = wp.draw(ax, lon, lat, np.where(land > 0.5, dtot, np.nan),
                    -v, v, cmap=spec["cmap"], land=land)
        cb = fig.colorbar(p, ax=ax, fraction=0.040, pad=0.03, shrink=0.82)
        cb.set_label(f"$\\Delta${spec['label']} ({spec['unit']})",
                     fontsize=9)

    if sel is not None:
        ax.contour(lon, lat, sel.astype(float), levels=[0.5],
                   colors="red", linewidths=2.0,
                   transform=wp.ccrs.PlateCarree(), zorder=6)
    else:
        la0, la1, lo0, lo1 = box
        ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0],
                color="red", lw=2.2, transform=wp.ccrs.PlateCarree(),
                zorder=6)
    ax.set_title(f"({label}) {name}", fontsize=11, weight="bold")
    return ax


def fig_all(a, spec, days, hours, dmean, run_c, ref_c,
            lat, lon, land, dtot, box, sel):
    # Three rows, not two. The map used to share a row with the day-by-
    # hour panel, which left both of them narrow and the cycle above them
    # squeezed. Each now gets the full width, and the map sits alone on
    # the bottom row at one third of it, which is all a reference map
    # needs.
    fig = plt.figure(figsize=(12.0, 13.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.35, 1.25],
                          hspace=0.34, wspace=0.22)
    mk = "o" if len(hours) <= 12 else None

    ax = fig.add_subplot(gs[0, :])
    shade_night(ax, hours)
    # nanmean throughout, because --select wet leaves a NaN row for
    # every day on which no point qualified; a plain mean returns NaN.
    if a.values == "actual":
        # The two runs as they stand, not their difference. The faint
        # per-day lines are dropped here: twenty of each, in two
        # colours, is unreadable, and the difference figure already
        # shows the day-to-day spread.
        ax.plot(hours, np.nanmean(run_c, axis=0), color="tab:red", lw=2.6,
                marker=mk, label=wp.rlab(a.run))
        ax.plot(hours, np.nanmean(ref_c, axis=0), color="tab:blue", lw=2.6,
                ls="--", marker=mk, label=wp.rlab(a.ref))
        ax.set_ylabel(f"{spec['label']} ({spec['unit']})", fontsize=11)
        ax.set_title(f"(a) March mean, {wp.rlab(a.run)} and "
                     f"{wp.rlab(a.ref)}", fontsize=12, weight="bold")
    else:
        for i in range(len(days)):
            ax.plot(hours, dmean[i], color="tab:blue", lw=0.8, alpha=0.30)
        ax.plot(hours, np.nanmean(dmean, axis=0), color="k", lw=2.6,
                marker=mk, label="March 2011")
        ax.set_ylabel(f"$\\Delta${spec['label']} ({spec['unit']})",
                      fontsize=11)
        ax.set_title(f"(a) {wp.rlab(a.run)} $-$ {wp.rlab(a.ref)}, "
                     f"one line per day", fontsize=12, weight="bold")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlim(hours[0] - 0.6, hours[-1] + 0.6)
    ax.set_xticks(hours)
    ax.set_xlabel("Hour, AWST", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")

    axh = fig.add_subplot(gs[1, :])
    v = float(np.nanpercentile(np.abs(dmean), 99)) or 1.0
    edges = np.append(hours - 1.5, hours[-1] + 1.5)
    im = axh.pcolormesh(edges, np.arange(len(days) + 1), dmean,
                        cmap=spec["cmap"], vmin=-v, vmax=v, shading="flat")
    axh.set_xticks(hours)
    axh.set_yticks(np.arange(len(days)) + 0.5)
    axh.set_yticklabels([d[5:] for d in days], fontsize=7)
    axh.set_xlabel("Hour, AWST", fontsize=11)
    axh.set_title("(b) Day by hour", fontsize=12, weight="bold")
    for h in (NIGHT[3] - 0.5, NIGHT[0] - 0.5):
        axh.axvline(h, color="k", lw=1.0, ls="--")
    cb = fig.colorbar(im, ax=axh, fraction=0.05, pad=0.03)
    cb.set_label(f"$\\Delta${spec['label']} ({spec['unit']})", fontsize=10)

    region_panel(fig, gs[2, 1], spec, lat, lon, land, dtot, box,
                 sel, "c", plain=not a.region_field, name=a.region_name)
    tag = "actual" if a.values == "actual" else "all"
    wp.save(fig, os.path.join(a.outdir, f"dayhour_{a.var}_{tag}.png"))


def fig_events(a, spec, days, hours, run_c, ref_c, picks,
               lat, lon, land, dtot, box, sel):
    idx = [i for i, d in enumerate(days) if int(d[8:10]) in picks]
    if not idx:
        raise SystemExit(f"ERROR: none of the days {sorted(picks)} are in "
                         f"the file ({days[0]} to {days[-1]})")
    n = len(idx)
    mk = "o" if len(hours) <= 12 else None

    # The day panels wrap into a grid of --ncol columns and the map takes
    # the next free cell, so five days at --ncol 3 give a 3 x 2 block
    # with no gap, and eight days give a full 3 x 3. One long row put
    # every cycle into a third of the width it needed, which is where
    # the crowding came from.
    #
    # Panels share one y range, so the days can be read against each
    # other; a per-panel range would make a quiet day look like a wet
    # one. Tick labels are kept on the first column and on the last
    # panel of each column only.
    ncol = max(1, min(a.ncol, n + 1))
    nrow = int(np.ceil((n + 1) / ncol))
    fig = plt.figure(figsize=(3.3 * ncol, 3.5 * nrow))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.42, wspace=0.16)

    axs = [fig.add_subplot(gs[k // ncol, k % ncol]) for k in range(n)]
    for k, (ax, i) in enumerate(zip(axs, idx)):
        shade_night(ax, hours)
        ax.plot(hours, run_c[i], color="tab:red", lw=2.0, marker=mk,
                label=wp.rlab(a.run))
        ax.plot(hours, ref_c[i], color="tab:blue", lw=2.0, ls="--",
                marker=mk, label=wp.rlab(a.ref))
        ax.set_xlim(hours[0] - 0.6, hours[-1] + 0.6)
        ax.set_xticks(hours[::2])
        ax.set_title(f"({chr(97 + k)}) {days[i]}", fontsize=11, weight="bold")
        # the x label goes on the lowest panel in this column, which is
        # this one if the cell below it holds the map or nothing
        if k + ncol >= n:
            ax.set_xlabel("Hour, AWST", fontsize=10)
        if k % ncol:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(f"{spec['label']} ({spec['unit']})", fontsize=10)

    lo = min(x.get_ylim()[0] for x in axs)
    hi = max(x.get_ylim()[1] for x in axs)
    for ax in axs:
        ax.set_ylim(lo, hi)
    axs[0].legend(fontsize=9)

    region_panel(fig, gs[n // ncol, n % ncol], spec, lat, lon, land,
                 dtot, box, sel, chr(97 + n),
                 plain=not a.region_field, name=a.region_name)
    wp.save(fig, os.path.join(a.outdir, f"dayhour_{a.var}_events.png"))

    print(f"\n{'date':<12s}{wp.rlab(a.run):>10s}{wp.rlab(a.ref):>12s}"
          f"   peak hour AWST")
    for i in idx:
        print(f"{days[i]:<12s}{hours[int(np.argmax(run_c[i]))]:>10d}"
              f"{hours[int(np.argmax(ref_c[i]))]:>12d}")


def series(a, spec, phys, land, lat, lon):
    """
    One variable, reduced to area-mean cycles over the selected region.

    Pulled out of main() so --mode twin can ask for a second variable
    without a second copy of the selection logic. Returns a dict rather
    than a long tuple: the twin figure needs four of these fields and
    the other modes need all of them.
    """
    days, hours, run_s, run_t = stack(spec, a.raw_bands, phys, a.run)
    d2, _, ref_s, ref_t = stack(spec, a.raw_bands, phys, a.ref)
    keep = np.isin(days, d2)
    days, run_s = days[keep], run_s[:, keep]
    ref_s = ref_s[:, np.isin(d2, days)]
    print(f"{spec['label']}: {len(days)} days, {days[0]} to {days[-1]}; "
          f"{len(hours)} samples per day")

    dtot = (run_t - ref_t).mean(axis=0)
    box, sel, npts = None, None, None

    if a.select == "zone":
        sel = load_mask(a.mask, land)
        run_c = run_s.mean(axis=0)[..., sel].mean(axis=-1)
        ref_c = ref_s.mean(axis=0)[..., sel].mean(axis=-1)
        dmean = run_c - ref_c
        dmax = (run_s - ref_s).mean(axis=0)[..., sel].max(axis=-1)

    elif a.select == "wet":
        # Selection is always on RAIN, whatever is being plotted, so a
        # cloud cycle is read on the days and points where it rained.
        # When --var is already rain these are the same arrays.
        if spec["src"] == "rain":
            rr, rf = run_s, ref_s
        else:
            rspec = VARS["rain"]
            rdays, _, rr, _ = stack(rspec, False, phys, a.run)
            rd2, _, rf, _ = stack(rspec, False, phys, a.ref)
            common = np.intersect1d(np.intersect1d(rdays, rd2), days)
            rr = rr[:, np.isin(rdays, common)]
            rf = rf[:, np.isin(rd2, common)]
            k2 = np.isin(days, common)
            days, run_s, ref_s = days[k2], run_s[:, k2], ref_s[:, k2]
            print(f"rain and cloud share {len(days)} days")

        zone = load_mask(a.mask, land, "--select wet")
        sel = wet_mask(rr, rf, zone, a.wet_thresh, a.wet_on)
        run_c, _, npts = reduce_sel(run_s.mean(axis=0), sel)
        ref_c, _, _ = reduce_sel(ref_s.mean(axis=0), sel)
        dmean = run_c - ref_c
        _, dmax, _ = reduce_sel((run_s - ref_s).mean(axis=0), sel)
        print(f"{int((npts > 0).sum())} of {len(days)} days have at least "
              f"one qualifying point")
        if not (npts > 0).any():
            raise SystemExit(
                f"ERROR: no coastal grid point reached {a.wet_thresh:g} mm "
                f"on any day.\n       Lower --wet-thresh, or check the "
                f"units in the stage-3 file.")
    else:
        box = (tuple(a.box) if a.box is not None else
               auto_box(dtot, lat, lon, land,
                        a.auto_box[0], a.auto_box[-1]))
        m = box_mask(lat, lon, box) & (land > 0.5)
        if not m.any():
            raise SystemExit(no_land_message(box, lat, lon, land))
        print(f"box: {int(m.sum())} land grid points, "
              f"{box[0]:.2f} to {box[1]:.2f} N, "
              f"{box[2]:.2f} to {box[3]:.2f} E")
        run_c = run_s.mean(axis=0)[..., m].mean(axis=-1)
        ref_c = ref_s.mean(axis=0)[..., m].mean(axis=-1)
        dmean = run_c - ref_c
        dmax = (run_s - ref_s).mean(axis=0)[..., m].max(axis=-1)

    return dict(days=days, hours=hours, run_c=run_c, ref_c=ref_c,
                dmean=dmean, dmax=dmax, dtot=dtot, box=box, sel=sel,
                npts=npts, spec=spec)


def cycle_panel(fig, cell, spec, hours, dmean, label, color="tab:red",
                ls="--", share=None):
    """
    The March-mean diurnal cycle of one variable, as a panel.

    Used as the last cell of --mode twin-events, where the day panels
    beside it are single days: this one is the month. It carries its own
    y range by default, because the month mean is much weaker than an
    event day and sharing the day panels' range would flatten it to a
    line. Pass share=(lo, hi) to force the comparison instead.
    """
    ax = fig.add_subplot(cell)
    shade_night(ax, hours)
    ax.plot(hours, np.nanmean(dmean, axis=0), color=color, lw=2.4, ls=ls,
            marker="o" if len(hours) <= 12 else None)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlim(hours[0] - 0.6, hours[-1] + 0.6)
    ax.set_xticks(hours[::3] if len(hours) > 12 else hours[::2])
    ax.set_xlabel("Hour, AWST", fontsize=10)
    ax.set_ylabel(f"$\\Delta${spec['label']} ({spec['unit']})",
                  fontsize=10, color=color)
    ax.tick_params(axis="y", labelcolor=color)
    if share is not None:
        ax.set_ylim(*share)
    ax.set_title(f"({label}) $\\Delta${spec['label']}, March 2011",
                 fontsize=11, weight="bold")
    return ax


def field_panel(fig, cell, spec, lat, lon, land, dtot, label):
    """
    The March-mean difference field of one variable, as a map.

    Unlike region_panel this draws no outline: it is the field itself,
    not a statement about where an average was taken. Used as the last
    cell of --mode twin-events, where the day panels already say where
    the averages came from.
    """
    ax = fig.add_subplot(cell, projection=wp.ROTATED_POLE)
    if wp._features_ok():
        ax.coastlines()
        ax.add_feature(wp.cfeature.BORDERS, linewidth=0.5)
    v = float(np.nanpercentile(np.abs(dtot), 99)) or 1.0
    p = wp.draw(ax, lon, lat, np.where(land > 0.5, dtot, np.nan),
                -v, v, cmap=spec["cmap"], land=land)
    ax.set_title(f"({label}) $\\Delta${spec['label']}", fontsize=11,
                 weight="bold")
    cb = fig.colorbar(p, ax=ax, fraction=0.040, pad=0.03, shrink=0.82)
    cb.set_label(f"$\\Delta${spec['label']} ({spec['unit']})", fontsize=9)
    return ax


def region_2d(S, lat, lon):
    """
    One 2-D mask for the inset, whatever --select produced.

    --select wet gives a different set of points each day, so the inset
    shows the union: every point that took part on at least one day. The
    panel version of the map shows the per-point COUNT instead, which is
    the honest picture; the inset cannot carry a colour bar, so it shows
    the footprint only.
    """
    sel = S["sel"]
    if sel is None:
        return box_mask(lat, lon, S["box"])
    return sel.any(axis=0) if sel.ndim == 3 else sel


def fig_cycle(a, S, lat, lon, land):
    """
    One variable, one axes: the March-mean diurnal cycle of the
    difference, with the averaging region as an inset rather than a
    panel of its own.
    """
    spec, hours = S["spec"], S["hours"]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    shade_night(ax, hours)
    if a.per_day:
        for i in range(len(S["days"])):
            ax.plot(hours, S["dmean"][i], color="tab:blue", lw=0.8,
                    alpha=0.28)
    # nanmean: --select wet leaves a NaN row for any day with no
    # qualifying point, and a plain mean would return NaN throughout.
    ax.plot(hours, np.nanmean(S["dmean"], axis=0), color="k", lw=2.8,
            marker="o" if len(hours) <= 12 else None, label="March 2011")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlim(hours[0] - 0.6, hours[-1] + 0.6)
    ax.set_xticks(hours)
    ax.set_xlabel("Hour, AWST", fontsize=11)
    ax.set_ylabel(f"$\\Delta${spec['label']} ({spec['unit']})", fontsize=11)
    ax.set_title(f"{wp.rlab(a.run)} $-$ {wp.rlab(a.ref)}, March 2011",
                 fontsize=12, weight="bold")
    # With no per-day lines there is one line and the title already
    # names it, so a legend would only repeat itself.
    if a.per_day:
        ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    # AFTER tight_layout: a hand-placed axes confuses it.
    if not a.no_inset:
        DC.band_map(fig, ax, a.coast, region_2d(S, lat, lon), a.inset)
    wp.save(fig, os.path.join(a.outdir, f"dayhour_{a.var}_cycle.png"))


def fig_twin(a, SL, SR, lat, lon, land):
    """
    Two variables, one axes, two y scales.

    The left axis carries --var and the right --twin. They are NOT on
    the same sampling rate: rain is hourly (24 points) and cloud
    3-hourly (8), so each is drawn on its own hour axis and the cloud
    series is marked at its eight samples. It is not a resolved cycle.

    A second y axis invites the eye to read a correlation that the
    scaling put there. The only thing tied between the two scales is
    zero: both axes are made symmetric about it, so the SIGNS can be
    compared down the figure and the relative amplitudes cannot.
    """
    sl, sr = SL["spec"], SR["spec"]
    shared = np.intersect1d(SL["days"], SR["days"])
    if len(shared) == 0:
        raise SystemExit("ERROR: the two variables share no days")
    if len(shared) < max(len(SL["days"]), len(SR["days"])):
        print(f"twin: averaging over the {len(shared)} days both "
              f"variables have")
    ml = np.nanmean(SL["dmean"][np.isin(SL["days"], shared)], axis=0)
    mr = np.nanmean(SR["dmean"][np.isin(SR["days"], shared)], axis=0)

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    shade_night(ax, SL["hours"])
    l1, = ax.plot(SL["hours"], ml, color="tab:blue", lw=2.6,
                  marker="o" if len(SL["hours"]) <= 12 else None,
                  label=f"$\\Delta${sl['label']}")
    ax.set_ylabel(f"$\\Delta${sl['label']} ({sl['unit']})", fontsize=11,
                  color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    l2, = ax2.plot(SR["hours"], mr, color="tab:red", lw=2.6, ls="--",
                   marker="o" if len(SR["hours"]) <= 12 else None,
                   label=f"$\\Delta${sr['label']}")
    ax2.set_ylabel(f"$\\Delta${sr['label']} ({sr['unit']})", fontsize=11,
                   color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Zero on both axes at the same height, so a sign change in one
    # series lines up with the other instead of floating.
    for b, m in ((ax, ml), (ax2, mr)):
        v = float(np.nanmax(np.abs(m))) or 1.0
        b.set_ylim(-1.15 * v, 1.15 * v)
    ax.axhline(0, color="0.4", lw=0.8)

    allh = np.concatenate([SL["hours"], SR["hours"]])
    ax.set_xlim(allh.min() - 0.6, allh.max() + 0.6)
    ax.set_xticks(np.unique(np.concatenate([SL["hours"], SR["hours"]])))
    ax.set_xlabel("Hour, AWST", fontsize=11)
    ax.set_title(f"{wp.rlab(a.run)} $-$ {wp.rlab(a.ref)}, March 2011",
                 fontsize=12, weight="bold")
    ax.legend(handles=[l1, l2], fontsize=10, loc="upper left")

    fig.tight_layout()
    # The region goes inside the axes, not in a panel of its own, and
    # AFTER tight_layout, which a hand-placed axes confuses. twinx makes
    # ax2 share the same box, so anchoring on ax is enough.
    if not a.no_inset:
        DC.band_map(fig, ax, a.coast, region_2d(SL, lat, lon), a.inset)
    wp.save(fig, os.path.join(a.outdir,
                              f"dayhour_{a.var}_{a.twin}_twin.png"))

    print(f"\n{'hour':>6s}{sl['label']:>14s}{sr['label']:>14s}")
    for k, h in enumerate(SL["hours"]):
        j = int(np.argmin(np.abs(SR["hours"] - h)))
        other = f"{mr[j]:>14.3f}" if SR["hours"][j] == h else f"{'':>14s}"
        print(f"{h:>6d}{ml[k]:>14.3f}{other}")


def fig_twin_events(a, SL, SR, picks, lat, lon, land):
    """
    One panel per named day, each carrying both variables on two y axes.

    Same contract as fig_twin: only zero is tied between the two scales.
    Here the panels also share one y range per axis, so a quiet day
    cannot be rescaled into looking like a wet one -- which is the whole
    reason the event panels exist.
    """
    sl, sr = SL["spec"], SR["spec"]
    shared = np.intersect1d(SL["days"], SR["days"])
    pick_days = [d for d in shared if int(d[8:10]) in picks]
    if not pick_days:
        raise SystemExit(
            f"ERROR: none of the days {sorted(picks)} are present in both "
            f"variables.\n       Shared days run {shared[0]} to "
            f"{shared[-1]}.")
    if len(pick_days) < len(picks):
        missing = sorted(picks - {int(d[8:10]) for d in pick_days})
        print(f"not in both variables, skipped: {missing}")

    il = {d: i for i, d in enumerate(SL["days"])}
    ir = {d: i for i, d in enumerate(SR["days"])}
    n = len(pick_days)

    ncol = max(1, min(a.ncol, n + 1))
    nrow = int(np.ceil((n + 1) / ncol))
    fig = plt.figure(figsize=(3.9 * ncol, 3.6 * nrow))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.42, wspace=0.42)

    mkl = "o" if len(SL["hours"]) <= 12 else None
    mkr = "o" if len(SR["hours"]) <= 12 else None
    pairs, handles = [], None
    for k, d in enumerate(pick_days):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        shade_night(ax, SL["hours"])
        l1, = ax.plot(SL["hours"], SL["dmean"][il[d]], color="tab:blue",
                      lw=2.0, marker=mkl, label=f"$\\Delta${sl['label']}")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.axhline(0, color="0.4", lw=0.8)

        ax2 = ax.twinx()
        l2, = ax2.plot(SR["hours"], SR["dmean"][ir[d]], color="tab:red",
                       lw=2.0, ls="--", marker=mkr,
                       label=f"$\\Delta${sr['label']}")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        allh = np.concatenate([SL["hours"], SR["hours"]])
        ax.set_xlim(allh.min() - 0.6, allh.max() + 0.6)
        ax.set_xticks(SL["hours"][::3] if len(SL["hours"]) > 12
                      else SL["hours"][::2])
        ax.set_title(f"({chr(97 + k)}) {d}", fontsize=11, weight="bold")
        if k + ncol >= n:
            ax.set_xlabel("Hour, AWST", fontsize=10)
        pairs.append((ax, ax2))
        handles = handles or [l1, l2]

    # One y range per axis across every panel, set after all are drawn.
    for which, spec_, col in ((0, sl, "tab:blue"), (1, sr, "tab:red")):
        v = max(abs(x) for pr in pairs
                for x in pr[which].get_ylim()) or 1.0
        for k, pr in enumerate(pairs):
            pr[which].set_ylim(-1.1 * v, 1.1 * v)
            # Last COLUMN only, not "last panel". A short final row would
            # otherwise put a right-hand label next to the last cell,
            # which carries its own axis and its own label.
            first, last = k % ncol == 0, k % ncol == ncol - 1
            show = first if which == 0 else last
            pr[which].tick_params(labelleft=False) if (which == 0 and
                                                      not show) else None
            if which == 1 and not show:
                pr[which].tick_params(labelright=False)
            if show:
                pr[which].set_ylabel(
                    f"$\\Delta${spec_['label']} ({spec_['unit']})",
                    fontsize=10, color=col)

    pairs[0][0].legend(handles=handles, fontsize=9, loc="upper left")
    cell, lab = gs[n // ncol, n % ncol], chr(97 + n)
    if a.last_panel == "zone":
        region_panel(fig, cell, sl, lat, lon, land, SL["dtot"],
                     SL["box"], SL["sel"], lab,
                     plain=not a.region_field, name=a.region_name)
    elif a.last_panel.startswith("field"):
        F = SR if a.last_panel.endswith("right") else SL
        field_panel(fig, cell, F["spec"], lat, lon, land, F["dtot"], lab)
    else:
        # The month-mean cycle of one of the two variables. The day
        # panels beside it are single days, so this is the only panel in
        # the figure that speaks for March as a whole. Say that in the
        # caption: the sampling differs across the figure.
        right = a.last_panel.endswith("right")
        F = SR if right else SL
        share = (pairs[0][1 if right else 0].get_ylim()
                 if a.last_share else None)
        cycle_panel(fig, cell, F["spec"], F["hours"], F["dmean"], lab,
                    color="tab:red" if right else "tab:blue",
                    ls="--" if right else "-", share=share)
    wp.save(fig, os.path.join(
        a.outdir, f"dayhour_{a.var}_{a.twin}_twin_events.png"))

    print(f"\n{'date':<12s}{sl['label']:>12s} peak{sr['label']:>12s} peak")
    for d in pick_days:
        print(f"{d:<12s}"
              f"{SL['hours'][int(np.nanargmax(SL['dmean'][il[d]]))]:>17d}"
              f"{SR['hours'][int(np.nanargmax(SR['dmean'][ir[d]]))]:>17d}")


def main(a):
    spec = VARS[a.var]
    phys = list(wp.PHYS) if a.rcm == "mean" else [a.rcm]

    if a.probe:
        tmpl = RAIN_UNIT if spec["src"] == "rain" else CLOUD_UNIT
        p = path(tmpl, phys[0], a.ref, wp.DATES[0])
        print(f"DERIVED = {wp.DERIVED}\n{p}")
        if not os.path.exists(p):
            print(f"  MISSING -- run {spec['submit']}")
            return
        with xr.open_dataset(p) as ds:
            print(f"  dims {dict(ds.sizes)}")
            print(f"  vars {sorted(ds.data_vars)}")
            d = np.array([str(v)[:10] for v in ds["day"].values])
            print(f"  {len(d)} days, {d[0]} to {d[-1]}")
            print(f"  hours {np.asarray(ds['hour'].values).tolist()}")
            if "bands" in ds.attrs:
                print(f"  {ds.attrs['bands']}")
        return

    if spec["src"] == "cloud":
        b = spec["raw"] if a.raw_bands else spec["paper"]
        print(f"--var {a.var} -> band index {b} "
              f"({'wrf-python' if a.raw_bands else 'manuscript'} numbering)")

    if spec["src"] == "cloud":
        b = spec["raw"] if a.raw_bands else spec["paper"]
        print(f"--var {a.var} -> band index {b} "
              f"({'wrf-python' if a.raw_bands else 'manuscript'} numbering)")

    land, lat, lon = wp.grid(a.coast)
    print(f"{a.rcm}: {len(phys) * len(wp.DATES)} members")

    S = series(a, spec, phys, land, lat, lon)
    days, hours = S["days"], S["hours"]
    table(spec, days, hours, S["dmean"], S["dmax"], S["npts"])

    if a.mode in ("twin", "twin-events"):
        if a.twin is None:
            raise SystemExit("ERROR: --mode twin needs --twin VAR, for "
                             "example --var rain --twin mid")
        if a.select == "wet":
            raise SystemExit(
                "ERROR: --mode twin does not support --select wet.\n"
                "       The wet composite picks a different set of points "
                "for each day, so\n       the two variables would be "
                "averaged over different regions. Use\n       --select "
                "zone or --select box.")
        S2 = series(a, VARS[a.twin], phys, land, lat, lon)
        if a.mode == "twin-events":
            picks = {int(x) for x in a.days.split(",") if x.strip()}
            fig_twin_events(a, S, S2, picks, lat, lon, land)
        else:
            fig_twin(a, S, S2, lat, lon, land)
        return

    if a.mode == "cycle":
        fig_cycle(a, S, lat, lon, land)
    elif a.mode == "all":
        fig_all(a, spec, days, hours, S["dmean"], S["run_c"], S["ref_c"],
                lat, lon, land, S["dtot"], S["box"], S["sel"])
    else:
        picks = {int(x) for x in a.days.split(",") if x.strip()}
        fig_events(a, spec, days, hours, S["run_c"], S["ref_c"], picks,
                   lat, lon, land, S["dtot"], S["box"], S["sel"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coast")
    ap.add_argument("mask", nargs="?", default=None,
                    help="CoastalLandMask_N12.nc, required by --select wet")
    ap.add_argument("--select", default="zone",
                    choices=["zone", "wet", "box"],
                    help="zone: the coastal zone from the mask, as in the "
                         "Methods. wet: only its points whose daily rain "
                         "passes --wet-thresh. box: a fixed rectangle")
    ap.add_argument("--wet-thresh", type=float, default=5.0,
                    help="daily rain threshold in mm, default 5")
    ap.add_argument("--wet-on", default="both",
                    choices=["both", "either", "run", "ref"],
                    help="which run decides a point is wet. both is the "
                         "only one whose amount difference is unbiased")
    ap.add_argument("--var", default="rain", choices=sorted(VARS))
    ap.add_argument("--mode", default="all",
                    choices=["all", "cycle", "events", "twin", "twin-events"],
                    help="cycle: the March-mean diurnal cycle alone, one "
                         "axes, region as an inset. "
                         "all: every day plus the day-by-hour map. "
                         "events: the named days, each run drawn "
                         "separately. twin: --var and --twin as two "
                         "differences on one axes with two y scales. "
                         "twin-events: the same, one panel per day in --days")
    ap.add_argument("--twin", default=None, choices=sorted(VARS),
                    help="second variable for --mode twin and twin-events, on the right "
                         "y axis, e.g. --var rain --twin mid")
    ap.add_argument("--values", default="diff", choices=["diff", "actual"],
                    help="panel (a) of --mode all. diff: the difference, "
                         "one faint line per day. actual: the March mean "
                         "of each run as it stands, no difference")
    ap.add_argument("--per-day", action="store_true",
                    help="--mode cycle: draw one faint line per day "
                         "behind the mean, as --mode all does")
    ap.add_argument("--no-inset", action="store_true",
                    help="drop the inset region map from --mode cycle "
                         "and --mode twin")
    ap.add_argument("--inset", type=float, nargs=4,
                    default=[0.70, 0.56, 0.27, 0.40],
                    metavar=("X0", "Y0", "W", "H"),
                    help="inset position inside the axes, 0-1. Move it "
                         "if the cycle runs under it")
    ap.add_argument("--last-panel", default="cycle-right",
                    choices=["cycle-right", "cycle-left", "field-right",
                             "field-left", "zone"],
                    help="--mode twin-events: what the cell after the "
                         "last day holds. cycle-right (default) the "
                         "March-mean diurnal cycle of --twin, "
                         "cycle-left that of --var, field-* the "
                         "March-mean difference MAP instead, zone the "
                         "averaging-region map")
    ap.add_argument("--last-share", action="store_true",
                    help="give the last cycle panel the same y range as "
                         "the day panels. Off by default: the month mean "
                         "is much weaker than an event day and sharing "
                         "flattens it")
    ap.add_argument("--region-field", action="store_true",
                    help="shade the March difference field under the "
                         "region map. Off by default: the map is there "
                         "to say where the average was taken, and the "
                         "field already has a figure of its own")
    ap.add_argument("--region-name", default="Core rainfall zone",
                    help="title of the region panel (default: "
                         "Core rainfall zone)")
    ap.add_argument("--ncol", type=int, default=3,
                    help="columns in the --mode events grid; the map takes "
                         "the cell after the last day, so 5 days gives a "
                         "3 x 2 and 8 days a full 3 x 3")
    ap.add_argument("--days", default="4,5,14,15,16",
                    help="days of March for --mode events; the paper names "
                         "4-5 and 14-16")
    ap.add_argument("--box", type=float, nargs=4, default=None,
                    metavar=("LAT0", "LAT1", "LON0", "LON1"))
    ap.add_argument("--auto-box", type=int, nargs="+", default=[30, 50],
                    metavar="N",
                    help="box size in grid points: one value for a square, "
                         "two for latitude then longitude")
    ap.add_argument("--raw-bands", action="store_true",
                    help="use wrf-python's band numbering instead of the "
                         "manuscript's, to check against the map figures")
    ap.add_argument("--run", default="CNTL")
    ap.add_argument("--ref", default=wp.REF_RUN)
    ap.add_argument("--rcm", default="mean", choices=["mean", "R3", "R5"])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--outdir", default="figs")
    main(ap.parse_args())
