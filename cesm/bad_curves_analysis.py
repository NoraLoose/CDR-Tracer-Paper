#!/usr/bin/env python
# coding: utf-8
"""
Build a cache of per-polygon diagnostics used to explain which validation
curves have the largest relative error ("bad curves").

For every polygon (000..689) we collect three families of predictors, one per
hypothesis:

  1. sea-ice cover   -> area-weighted ANTITRACER_IFRAC over the polygon
                        footprint, for the injection January and for the
                        all-time maximum (poly_ifrac_inject_jan / poly_ifrac_max_all).
  2. efficiency      -> final and peak CDR efficiency (uptake / forcing) of the
                        true CESM-MARBL run.
  3. narrow passage  -> geometry metrics: polygon centroid, and the land
                        fraction in a halo around the footprint (a proxy for
                        being boxed in by coastlines / straits).

and the target:

  * relative error   -> (truth - CDR tracer) / truth of the efficiency curve:
                        final value, time-mean |.|, max |.|, and the summed
                        |.| "divergence" metric used in validation_curves_all.
                        The full relative-error time series is stored too.

Everything is written to ONE netCDF (see CACHE_PATH) with a `polygon` dim of
length 690 and a `mode` dim of length 2 (dor, oae) for the curve metrics, plus
2-D maps and the grid, so the notebooks can just open it:
  - sea ice:  ifrac_inject_jan / ifrac_max_all
  - MLD:      mld_inject_jan / mld_mean_all  (metres, from SMYLE-FOSI HMXL -
              the atlas output carries no mixed-layer depth of its own)

The two file-heavy steps (reading ANTITRACER_IFRAC and the 2x690 curve files)
are checkpointed next to CACHE_PATH; pass reuse_intermediate=False to force a
full re-read.

Run:  python bad_curves_analysis.py
"""

import glob
import os

import numpy as np
import xarray as xr
from scipy import ndimage

from analysis import (
    INVERSE_POLYGON_MAP,
    open_true_curve,
    open_deficit_tracer_curve,
    deficit_tracer_experiment_paths,
)

N_POLYGONS = 690
MODES = ["dor", "oae"]
SUFFIX = {"dor": "all-dor-daily", "oae": "all-oae-daily"}
FORCING_VAR = {"dor": "DELTADIC", "oae": "DELTAALK"}

CACHE_PATH = (
    "/global/cfs/cdirs/m4746/Users/nora/Ocean-CDR-Atlas-v0/"
    "data/analysis/bad_curves_cache.nc"
)

# halo width (in grid cells) for the "boxed in by land" metric
HALO_CELLS = 3

# SMYLE-FOSI reference run (the physics the whole atlas branches from) - the
# atlas output itself carries no mixed-layer depth.  Model year 347 == 1999,
# POP time stamp = end of the averaging month, so stamp 0347-02 == Jan 1999.
SMYLE_HMXL = (
    "/global/cfs/projectdirs/m4746/Datasets/SMYLE-FOSI/ocn/proc/tseries/month_1/"
    "g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005.pop.h.HMXL.030601-036812.nc"
)
INJECT_STAMP_YEAR, INJECT_STAMP_MONTH = 347, 2  # Jan 1999 mean
MLD_FIELDS = ["mld_inject_jan", "mld_mean_all"]


def _hist_files(mode):
    subdir, _ = deficit_tracer_experiment_paths(SUFFIX[mode])
    base = (
        f"/global/cfs/cdirs/m4746/Users/nora/Ocean-CDR-Atlas-v0/"
        f"data/archive/{subdir}/ocn/hist/"
    )
    return sorted(glob.glob(base + "*.pop.h.*.nc"))


def load_grid(mode="oae"):
    """TLAT, TLONG, TAREA, KMT from the first history file."""
    ds = xr.open_dataset(_hist_files(mode)[0], decode_timedelta=False)
    return ds[["TLAT", "TLONG", "TAREA", "KMT"]].load()


def load_forcing_footprints(mode):
    """(polygon, nlat, nlon) boolean footprint of each polygon's forcing."""
    ds = xr.open_dataset(_hist_files(mode)[0], decode_timedelta=False)
    fv = FORCING_VAR[mode]
    masks = np.zeros((N_POLYGONS, ds.sizes["nlat"], ds.sizes["nlon"]), dtype=bool)
    for i in range(N_POLYGONS):
        masks[i] = (ds[f"{fv}{i:03d}_FORCING"].isel(time=0).values > 0)
    return masks


ICE_FIELDS = ["ifrac_inject_jan", "ifrac_max_all"]


def load_ice_maps(mode="oae"):
    """
    Read ANTITRACER_IFRAC from every monthly history file (one time step per
    file) and reduce to two 2-D maps:

      ifrac_inject_jan : sea-ice concentration in the injection January
                         (the first history file)
      ifrac_max_all    : maximum sea-ice concentration over the whole run
                         (ice-covered at any time)
    """
    files = _hist_files(mode)
    frames = []
    for f in files:
        ds = xr.open_dataset(f, decode_timedelta=False)
        frames.append(ds["ANTITRACER_IFRAC"].isel(time=0).values.astype("float32"))
    arr = np.stack(frames)                      # (time, nlat, nlon)

    out = xr.Dataset(
        {
            "ifrac_inject_jan": (("nlat", "nlon"), arr[0]),
            "ifrac_max_all": (("nlat", "nlon"), np.nanmax(arr, axis=0)),
        }
    )
    out["n_months"] = arr.shape[0]
    return out


def load_mld_maps():
    """
    Mixed-layer depth (metres) from the SMYLE-FOSI reference run, reduced to:

      mld_inject_jan : MLD in the injection January (Jan 1999 == stamp 0347-02)
      mld_mean_all   : long-term (full SMYLE-FOSI record) time-mean MLD
    """
    ds = xr.open_dataset(SMYLE_HMXL, decode_timedelta=False)
    hmxl = ds["HMXL"] / 100.0  # cm -> m
    t = ds["time"].values
    jan = next(
        i for i, x in enumerate(t)
        if x.year == INJECT_STAMP_YEAR and x.month == INJECT_STAMP_MONTH
    )
    out = xr.Dataset(
        {
            "mld_inject_jan": (("nlat", "nlon"), hmxl.isel(time=jan).values.astype("float32")),
            "mld_mean_all": (("nlat", "nlon"), hmxl.mean("time").values.astype("float32")),
        }
    )
    out["n_months"] = hmxl.sizes["time"]
    return out


def _centroid(mask, tlat, tlong, tarea):
    """Area-weighted centroid of a footprint, robust to the dateline."""
    w = tarea[mask]
    w = w / w.sum()
    lat = float((tlat[mask] * w).sum())
    lon_rad = np.deg2rad(tlong[mask])
    x = float((np.cos(lon_rad) * w).sum())
    y = float((np.sin(lon_rad) * w).sum())
    lon = float(np.rad2deg(np.arctan2(y, x))) % 360.0
    return lat, lon


def geometry_metrics(masks, grid):
    """Per-polygon centroid, size and land-halo fraction."""
    tlat = grid.TLAT.values
    tlong = grid.TLONG.values
    tarea = grid.TAREA.values
    land = grid.KMT.values == 0

    struct = ndimage.generate_binary_structure(2, 2)
    lat = np.full(N_POLYGONS, np.nan)
    lon = np.full(N_POLYGONS, np.nan)
    ncells = np.zeros(N_POLYGONS, dtype=int)
    area = np.full(N_POLYGONS, np.nan)
    land_halo = np.full(N_POLYGONS, np.nan)

    for i in range(N_POLYGONS):
        m = masks[i]
        if not m.any():
            continue
        lat[i], lon[i] = _centroid(m, tlat, tlong, tarea)
        ncells[i] = int(m.sum())
        area[i] = float(tarea[m].sum())
        halo = ndimage.binary_dilation(m, structure=struct, iterations=HALO_CELLS) & ~m
        if halo.any():
            land_halo[i] = float(land[halo].mean())

    basin = [INVERSE_POLYGON_MAP[i][0] for i in range(N_POLYGONS)]
    pol = [INVERSE_POLYGON_MAP[i][1] for i in range(N_POLYGONS)]
    return xr.Dataset(
        {
            "centroid_lat": ("polygon", lat),
            "centroid_lon": ("polygon", lon),
            "n_cells": ("polygon", ncells),
            "area_m2": ("polygon", area),
            "land_halo_frac": ("polygon", land_halo),
            "basin": ("polygon", basin),
            "basin_polygon": ("polygon", pol),
        },
        coords={"polygon": np.arange(N_POLYGONS)},
    )


# a footprint cell counts as "marginal ice zone" when its concentration is
# between these bounds - neither open water nor a solid ice pack
MIZ_LO, MIZ_HI = 0.15, 0.85


def ice_over_footprints(masks, grid, ice_maps):
    """
    Per-polygon sea-ice descriptors over the forcing footprint:

      poly_ifrac_inject_jan : area-weighted mean SIC, injection January
      poly_ifrac_max_all    : area-weighted mean of the run-maximum SIC
      poly_ice_jan_miz_frac : area-weighted fraction of the footprint in the
                              marginal ice zone (MIZ_LO..MIZ_HI) in January
                              -> the "partially but not fully ice-covered" metric
      poly_ice_jan_std      : area-weighted std of January SIC across the
                              footprint (0 if uniform, large if it straddles
                              the ice edge)
    """
    tarea = grid.TAREA.values
    jan = ice_maps["ifrac_inject_jan"].values

    mean = {f: np.full(N_POLYGONS, np.nan) for f in ICE_FIELDS}
    miz = np.full(N_POLYGONS, np.nan)
    std = np.full(N_POLYGONS, np.nan)

    for i in range(N_POLYGONS):
        m = masks[i]
        if not m.any():
            continue
        w = tarea[m]
        w = w / w.sum()
        for f in ICE_FIELDS:
            mean[f][i] = float(np.nansum(ice_maps[f].values[m] * w))

        jm = jan[m]
        ok = np.isfinite(jm)
        if ok.any():
            ww = w[ok] / w[ok].sum()
            jv = jm[ok]
            miz[i] = float(ww[(jv >= MIZ_LO) & (jv <= MIZ_HI)].sum())
            mu = float((ww * jv).sum())
            std[i] = float(np.sqrt((ww * (jv - mu) ** 2).sum()))

    data = {"poly_" + k: ("polygon", v) for k, v in mean.items()}
    data["poly_ice_jan_miz_frac"] = ("polygon", miz)
    data["poly_ice_jan_std"] = ("polygon", std)
    return xr.Dataset(data, coords={"polygon": np.arange(N_POLYGONS)})


def mld_over_footprints(masks, grid, mld_maps):
    """Area-weighted mean mixed-layer depth (m) over each polygon footprint."""
    tarea = grid.TAREA.values
    out = {f: np.full(N_POLYGONS, np.nan) for f in MLD_FIELDS}
    for i in range(N_POLYGONS):
        m = masks[i]
        if not m.any():
            continue
        w = tarea[m]
        w = w / w.sum()
        for f in MLD_FIELDS:
            v = mld_maps[f].values[m]
            ok = np.isfinite(v)
            if ok.any():
                out[f][i] = float((v[ok] * (w[ok] / w[ok].sum())).sum())
    return xr.Dataset(
        {"poly_" + k: ("polygon", v) for k, v in out.items()},
        coords={"polygon": np.arange(N_POLYGONS)},
    )


def curve_metrics(nt=180):
    """
    Efficiency and relative-error scalars *and* the full time series for every
    polygon and mode.

    Time series are stored on a fixed length `nt` (padded with NaN) together
    with an `elapsed_years` coordinate, so the notebooks can plot the whole
    family of curves without touching the per-polygon files again.
    """
    shape = (len(MODES), N_POLYGONS)
    eff_final = np.full(shape, np.nan)
    eff_peak = np.full(shape, np.nan)
    relerr_final = np.full(shape, np.nan)
    relerr_meanabs = np.full(shape, np.nan)
    relerr_maxabs = np.full(shape, np.nan)
    relerr_divergence = np.full(shape, np.nan)   # sum |100*(truth-tracer)/truth|
    absdiv = np.full(shape, np.nan)              # sum |truth-tracer|  (efficiency units)

    ts_shape = (len(MODES), N_POLYGONS, nt)
    eff_true_ts = np.full(ts_shape, np.nan)
    eff_approx_ts = np.full(ts_shape, np.nan)
    relerr_ts = np.full(ts_shape, np.nan)
    elapsed_years = np.full(nt, np.nan)

    for mi, mode in enumerate(MODES):
        sign = -1.0 if mode == "dor" else 1.0
        for pid in range(N_POLYGONS):
            sid = f"{pid:03d}"
            try:
                ds_true = open_true_curve(sid, mode)
                ds_tilde = open_deficit_tracer_curve(sid, SUFFIX[mode])
            except (FileNotFoundError, OSError):
                continue

            eff = sign * (ds_true.uptake / ds_true.forcing).squeeze().values
            eff_t = sign * (ds_tilde.uptake / ds_tilde.forcing).squeeze().values
            n = min(len(eff), len(eff_t), nt)
            eff, eff_t = eff[:n], eff_t[:n]

            eff_final[mi, pid] = eff[-1]
            eff_peak[mi, pid] = np.nanmax(eff)

            # sign cancels in the ratio, so this is exactly the `relative=True`
            # divergence of compare_difference_curves in validation_curves_all
            rel = 100.0 * np.divide(
                eff - eff_t, eff, out=np.full(n, np.nan), where=eff != 0
            )
            relerr_final[mi, pid] = rel[-1]
            relerr_meanabs[mi, pid] = np.nanmean(np.abs(rel))
            relerr_maxabs[mi, pid] = np.nanmax(np.abs(rel))
            relerr_divergence[mi, pid] = np.nansum(np.abs(rel))
            # the `relative=False` divergence (default highlight in that notebook)
            absdiv[mi, pid] = np.nansum(np.abs(eff - eff_t))

            eff_true_ts[mi, pid, :n] = eff
            eff_approx_ts[mi, pid, :n] = eff_t
            relerr_ts[mi, pid, :n] = rel

            if np.isnan(elapsed_years[:n]).any():
                ey = (ds_true.elapsed_time.values[:n] / 365.0)
                elapsed_years[:n] = ey

    dims = ("mode", "polygon")
    tdims = ("mode", "polygon", "t")
    return xr.Dataset(
        {
            "eff_final": (dims, eff_final),
            "eff_peak": (dims, eff_peak),
            "relerr_final": (dims, relerr_final),
            "relerr_meanabs": (dims, relerr_meanabs),
            "relerr_maxabs": (dims, relerr_maxabs),
            "relerr_divergence": (dims, relerr_divergence),
            "absdiv": (dims, absdiv),
            "eff_true_ts": (tdims, eff_true_ts),
            "eff_approx_ts": (tdims, eff_approx_ts),
            "relerr_ts": (tdims, relerr_ts),
        },
        coords={
            "mode": MODES,
            "polygon": np.arange(N_POLYGONS),
            "elapsed_years": ("t", elapsed_years),
        },
    )


_ICE_MAPS_TMP = os.path.join(os.path.dirname(CACHE_PATH), "_ice_maps_tmp.nc")
_MLD_MAPS_TMP = os.path.join(os.path.dirname(CACHE_PATH), "_mld_maps_tmp.nc")
_CURVES_TMP = os.path.join(os.path.dirname(CACHE_PATH), "_curves_tmp.nc")


def build_cache(path=CACHE_PATH, reuse_intermediate=True):
    print("grid + footprints ...")
    grid = load_grid("oae")
    masks = load_forcing_footprints("oae")

    print("geometry metrics ...")
    geom = geometry_metrics(masks, grid)

    # the next two steps each read hundreds of files; checkpoint them so a
    # downstream bug does not force a full re-read
    if reuse_intermediate and os.path.exists(_ICE_MAPS_TMP):
        print("sea-ice maps ... (cached)")
        ice_maps = xr.open_dataset(_ICE_MAPS_TMP).load()
    else:
        print("sea-ice maps ...")
        ice_maps = load_ice_maps("oae")
        ice_maps.to_netcdf(_ICE_MAPS_TMP)

    print("sea ice over footprints ...")
    ice_poly = ice_over_footprints(masks, grid, ice_maps)

    if reuse_intermediate and os.path.exists(_MLD_MAPS_TMP):
        print("mixed-layer depth maps ... (cached)")
        mld_maps = xr.open_dataset(_MLD_MAPS_TMP).load()
    else:
        print("mixed-layer depth maps ...")
        mld_maps = load_mld_maps()
        mld_maps.to_netcdf(_MLD_MAPS_TMP)

    print("mld over footprints ...")
    mld_poly = mld_over_footprints(masks, grid, mld_maps)

    if reuse_intermediate and os.path.exists(_CURVES_TMP):
        print("curve metrics ... (cached)")
        curves = xr.open_dataset(_CURVES_TMP, decode_timedelta=False).load()
        # backfill scalars added after the checkpoint was written, straight
        # from the stored time series (no need to re-read the curve files)
        if "absdiv" not in curves:
            curves["absdiv"] = np.abs(
                curves.eff_true_ts - curves.eff_approx_ts
            ).sum("t", skipna=True)
    else:
        print("curve metrics ...")
        curves = curve_metrics()
        curves.to_netcdf(_CURVES_TMP)

    ds = xr.merge([geom, ice_poly, mld_poly, curves])
    # strip the POP 2-D coordinates (TLAT/TLONG reference each other) so the
    # assignment does not drag conflicting coordinates into the merge
    for name in ["TLAT", "TLONG", "TAREA", "KMT"]:
        ds[name] = (("nlat", "nlon"), grid[name].values)
    for v in ICE_FIELDS:
        ds[v] = (("nlat", "nlon"), ice_maps[v].values)
    for v in MLD_FIELDS:
        ds[v] = (("nlat", "nlon"), mld_maps[v].values)

    ds.attrs["halo_cells"] = HALO_CELLS
    ds.attrs["description"] = "Per-polygon predictors and relative-error targets for bad-curve analysis"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    ds.to_netcdf(path)
    print("wrote", path)
    return ds


def open_cache(path=CACHE_PATH):
    return xr.open_dataset(path, decode_timedelta=False)


if __name__ == "__main__":
    build_cache()
