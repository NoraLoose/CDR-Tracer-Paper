#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import cftime
import glob
import s3fs

_S3_FS = None

def _get_s3fs():
    global _S3_FS
    if _S3_FS is None:
        _S3_FS = s3fs.S3FileSystem(anon=True)
    return _S3_FS

# -----------------------------
# Polygon maps
# -----------------------------
BASINS = [
    "North_Atlantic_basin",
    "North_Pacific_basin",
    "South",
    "Southern_Ocean",
]

NPOLYGON = {
    "North_Atlantic_basin": 150,
    "North_Pacific_basin": 200,
    "South": 300,
    "Southern_Ocean": 40,
}

# Forward and inverse maps
POLYGON_MASTER_MAP = {}  # (basin, polygon) → index
INVERSE_POLYGON_MAP = {}  # index → (basin, polygon)
idx = 0
for b in BASINS:
    for p in range(NPOLYGON[b]):
        POLYGON_MASTER_MAP[(b, p)] = idx
        INVERSE_POLYGON_MAP[idx] = (b, p)
        idx += 1

# -----------------------------
# Time utilities
# -----------------------------
def _time_dim(ds):
    """Return the name of the time dimension ('time' or 'elapsed_time')."""
    if "time" in ds.dims:
        return "time"
    if "elapsed_time" in ds.dims:
        return "elapsed_time"
    raise ValueError("Dataset has neither 'time' nor 'elapsed_time' dimension")

def _squeeze_extra_dims(da, keep_dim):
    """Drop size-1 dims other than `keep_dim` (e.g. a size-1 'injection_date')."""
    extra_dims = [d for d in da.dims if d != keep_dim and da.sizes[d] == 1]
    return da.squeeze(dim=extra_dims, drop=True) if extra_dims else da

def compute_dt_seconds(ds):
    """Return time step length in seconds for each time index."""
    tdim = _time_dim(ds)
    dt = ds.time_bound.isel(d2=1) - ds.time_bound.isel(d2=0)
    # time_bound may carry extra size-1 dims (e.g. 'injection_date') beyond
    # the actual time dimension - drop them so dt is 1-D over tdim.
    dt = _squeeze_extra_dims(dt, tdim)
    # For a single time step, xarray/cftime returns timedelta64 instead of
    # datetime.timedelta objects, so total_seconds() isn't available.
    if np.issubdtype(dt.values.dtype, np.timedelta64):
        dt_seconds = dt.values / np.timedelta64(1, "s")
    else:
        dt_seconds = np.array([d.total_seconds() for d in np.atleast_1d(dt.values)])
    return xr.DataArray(dt_seconds, dims=tdim)

def compute_elapsed_days(ds):
    """
    Compute elapsed days relative to one month before the first time in ds.time.
    
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Must have a 'time' coordinate (cftime or datetime).
    
    Returns
    -------
    xarray.DataArray
        Elapsed days coordinate.
    """
    time = _squeeze_extra_dims(ds.time, _time_dim(ds))

    # First time in the dataset
    t0 = np.atleast_1d(time.values)[0]

    # Only works with cftime.DatetimeNoLeap
    if not isinstance(t0, cftime.DatetimeNoLeap):
        raise ValueError("ds.time must be cftime.DatetimeNoLeap for this function.")

    # Subtract one month safely
    month = t0.month - 1
    year = t0.year
    if month == 0:
        month = 12
        year -= 1
    # Use day=1, safe for all months
    start = cftime.DatetimeNoLeap(year, month, 1)

    # Compute elapsed days
    elapsed = np.array([(t - start).days for t in np.atleast_1d(time.values)])

    return xr.DataArray(elapsed, dims=_time_dim(ds), attrs={"units": "days"})

def finalize(arr, ds):
    """Attach elapsed_time coordinate and return cleaned DataArray."""
    elapsed_days = compute_elapsed_days(ds)
    return arr.assign_coords(elapsed_time=elapsed_days)

_NMOL_TO_MOL = 1e-9

def to_instantaneous(cumulative_da):
    """Convert cumulative DataArray (nmol) to per-timestep values (mol).

    cumsum[0] = value[0] (injection month, recovered exactly);
    cumsum[i] - cumsum[i-1] = value[i] for i >= 1.
    elapsed_time is set to interval midpoints.
    """
    tdim = _time_dim(cumulative_da)
    et = cumulative_da.elapsed_time.values
    first = cumulative_da.isel({tdim: slice(0, 1)})
    rest = cumulative_da.diff(tdim)
    inst = xr.concat([first, rest], dim=tdim)
    midpoints = np.concatenate([[et[0] / 2], (et[:-1] + et[1:]) / 2])
    return inst.assign_coords(elapsed_time=(tdim, midpoints)) * _NMOL_TO_MOL

# -----------------------------
# Experiment paths and dataset loaders
# -----------------------------
def dor_experiment_paths(
    polygon_id,
    intervention_month="01",
    intervention_year="1999",
    realization="001",
):
    """Return (subdir, file_glob) for true CESM-MARBL experiment."""
    base_path = "/global/cfs/projectdirs/m4746/Projects/Ocean-CDR-Atlas-v0/data/archive"

    b, p = INVERSE_POLYGON_MAP[int(polygon_id)]
    multiplier = f"{int(polygon_id) * 4:05d}"

    subdir = (
        f"smyle.cdr-atlas-v0.glb-dor_"
        f"{b}_{p:03d}_{intervention_year}-{intervention_month}-01_"
        f"{multiplier}.{realization}"
    )

    file_glob = f"{base_path}/{subdir}/ocn/hist/*.pop.h.*.nc"

    return subdir, file_glob

def oae_experiment_paths(
    polygon_id,
    intervention_month="01",
    intervention_year="1999",
    realization="001",
):
    """Return (subdir, s3_glob) for true CESM-MARBL experiment."""
    base_path = "s3://us-west-2.opendata.source.coop/cworthy/oae-efficiency-atlas/data"

    b, p = INVERSE_POLYGON_MAP[int(polygon_id)]
    multiplier = f"{int(polygon_id) * 4:05d}"

    subdir = (
        f"smyle.cdr-atlas-v0.glb-oae_"
        f"{b}_{p:03d}_{intervention_year}-{intervention_month}-01_"
        f"{multiplier}.{realization}"
    )

    pid_str = f"{int(polygon_id):03d}"
    file_glob = f"{base_path}/experiments/{pid_str}/{intervention_month}/alk-forcing.{pid_str}-{intervention_year}-{intervention_month}.pop.h.*.nc"

    return subdir, file_glob
    
def deficit_tracer_experiment_paths(
    suffix,
    intervention_month="01",
    intervention_year="1999",
    realization="001",
):
    """Return (subdir, file_glob) for deficit tracer experiment."""
    base_path = (
        "/global/cfs/cdirs/m4746/Users/nora/"
        "Ocean-CDR-Atlas-v0/data/archive"
    )

    subdir = (
        f"smyle.cdr-atlas-v0.glb-antitracer_"
        f"{intervention_year}-{intervention_month}_{suffix}.{realization}"
    )

    file_glob = (
        f"{base_path}/{subdir}/ocn/hist/"
        f"smyle.cdr-atlas-v0.glb-antitracer_"
        f"{intervention_year}-{intervention_month}_{suffix}.{realization}"
        ".pop.h.*.nc"
    )

    return subdir, file_glob

MODES = {
    "dor": dor_experiment_paths,
    "oae": oae_experiment_paths,
}

def _open_experiment(path, first_file=False, file_index=None):
    """
    Open the dataset(s) matching `path` (local glob or s3:// glob).

    Files are named by year-month and sorted chronologically, with one
    time step per file. By default all matching files are opened. Pass
    `file_index` (an int, or a slice for a contiguous range) to only read
    the file(s) needed instead of the whole experiment - much faster when
    only a specific time step (or a leading range, e.g. for a cumulative
    sum) is required.
    """
    if first_file:
        file_index = 0

    is_s3 = path.startswith("s3://")

    if is_s3:
        fs = _get_s3fs()
        files = sorted(fs.glob(path))
        if not files:
            raise FileNotFoundError(f"No files match: {path}")
        if file_index is None:
            open_files = [fs.open(f"s3://{f}") for f in files]
            return xr.open_mfdataset(open_files, engine="h5netcdf", decode_timedelta=False)
        selected = files[file_index]
        if isinstance(file_index, slice):
            open_files = [fs.open(f"s3://{f}") for f in selected]
            return xr.open_mfdataset(open_files, engine="h5netcdf", decode_timedelta=False)
        return xr.open_dataset(fs.open(f"s3://{selected}"), engine="h5netcdf", decode_timedelta=False)

    if file_index is None:
        return xr.open_mfdataset(path, decode_timedelta=False)

    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No files match: {path}")
    selected = files[file_index]
    if isinstance(file_index, slice):
        return xr.open_mfdataset(selected, decode_timedelta=False)
    return xr.open_dataset(selected, decode_timedelta=False)


def open_true_experiment(*args, mode, first_file=False, file_index=None, **kwargs):

    experiment_paths = MODES[mode]

    _, path = experiment_paths(*args, **kwargs)

    return _open_experiment(path, first_file=first_file, file_index=file_index)


def open_deficit_tracer_experiment(*args, first_file=False, file_index=None, **kwargs):
    _, path = deficit_tracer_experiment_paths(*args, **kwargs)

    return _open_experiment(path, first_file=first_file, file_index=file_index)

analysis_base = (
    "/global/cfs/projectdirs/m4746/Users/nora/"
    "Ocean-CDR-Atlas-v0/data/analysis"
)


def open_true_curve(
    pid,
    mode,
    intervention_month="01",
    intervention_year="1999",
    realization="001",
):

    experiment_paths = MODES[mode]
        
    subdir, _ = experiment_paths(
        pid,
        intervention_month=intervention_month,
        intervention_year=intervention_year,
        realization=realization,
    )

    output_file = f"{analysis_base}/{subdir}/integrated.nc"
    return xr.open_dataset(output_file, decode_timedelta=False)


def open_true_flux_map(
    pid,
    mode,
    intervention_month="01",
    intervention_year="1999",
    realization="001",
):
    """Load the precomputed cumulative air-sea CO2 flux map (see
    integrate_true_flux_map.py) for a true CESM-MARBL experiment."""

    experiment_paths = MODES[mode]

    subdir, _ = experiment_paths(
        pid,
        intervention_month=intervention_month,
        intervention_year=intervention_year,
        realization=realization,
    )

    output_file = f"{analysis_base}/{subdir}/flux_map.nc"
    return xr.open_dataset(output_file, decode_timedelta=False).flux


def open_truth_sensitivity(
    pid,
    mode,
    intervention_month="01",
    intervention_year="1999",
    realization="001",
):
    """Load the global monthly carbonate-sensitivity maps (dDICdCO2, dDICdALK,
    and their _altco2 baselines) for a true CESM-MARBL experiment (see
    compute_truth_sensitivities.py)."""

    experiment_paths = MODES[mode]

    subdir, _ = experiment_paths(
        pid,
        intervention_month=intervention_month,
        intervention_year=intervention_year,
        realization=realization,
    )

    output_file = f"{analysis_base}/{subdir}/carbonate_sensitivity.nc"
    return xr.open_dataset(output_file, decode_timedelta=False)


def open_approx_flux_map(
    pid,
    suffix,
    intervention_month="01",
    intervention_year="1999",
    realization="001",
):
    """Load the precomputed cumulative approximate air-sea CO2 flux map
    (see integrate_flux_map.py) for a deficit-tracer experiment."""

    subdir, _ = deficit_tracer_experiment_paths(
        suffix,
        intervention_month=intervention_month,
        intervention_year=intervention_year,
        realization=realization,
    )

    output_file = f"{analysis_base}/{subdir}/flux_map_{pid}.nc"
    return xr.open_dataset(output_file, decode_timedelta=False).flux


def open_deficit_tracer_curve(
    pid,
    suffix,
    intervention_month="01",
    intervention_year="1999",
    realization="001",
):
    subdir, _ = deficit_tracer_experiment_paths(
        suffix,
        intervention_month=intervention_month,
        intervention_year=intervention_year,
        realization=realization,
    )

    output_file = f"{analysis_base}/{subdir}/integrated_{pid}.nc"
    return xr.open_dataset(output_file, decode_timedelta=False)

# ------------------
# Flux computations
# -----------------------------
def compute_forcing(ds, flux, cumulative=True):
    """Compute DIC forcing integrated over time."""
    dt = compute_dt_seconds(ds)
    forcing = ds[flux] * dt # nmol/cm^2/s * s = nmol/cm^2
    if cumulative:
        forcing = forcing.cumsum(_time_dim(ds))
    return forcing.where(ds.KMT > 0)

def compute_air_sea_flux(ds, flux1, flux2, cumulative=True):
    """Compute change in DIC inventory from two flux variables."""
    dt = compute_dt_seconds(ds)
    delta_dic = (ds[flux1] - ds[flux2]) * dt # mmol/m^3 cm/s * s = mmol/m^3 cm
    if cumulative:
        delta_dic = delta_dic.cumsum(_time_dim(ds))
    return delta_dic.where(ds.KMT > 0)

def compute_spatial_integral(da, ds):
    """Spatially integrate a DataArray over the ocean grid."""
    return (da * ds.TAREA).sum(["nlat", "nlon"])

