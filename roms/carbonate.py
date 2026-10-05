import numpy as np
import xarray as xr
import PyCO2SYS as pyco2
from roms_tools import Grid, ROMSOutput
from pathlib import Path
from typing import Tuple

def load_bgc_phys(grid: Grid, bgc_path: str, phys_path: str = None) -> xr.Dataset:
    """Load BGC dataset and optionally add physical variables."""
    roms_output = ROMSOutput(grid=grid, path=bgc_path, use_dask=True)
    ds = roms_output.ds

    if phys_path is not None:
        phys_output = ROMSOutput(
            grid=grid,
            path=phys_path,
            adjust_depth_for_sea_surface_height=True,
            use_dask=True,
        )
        assert (ds.ocean_time == phys_output.ds.ocean_time).all()
        ds["temp"] = phys_output.ds["temp"]
        ds["salt"] = phys_output.ds["salt"]

    # Mask spatial variables
    mask = grid.ds.mask_rho
    spatial_vars = [v for v in ds.data_vars if set(ds[v].dims) & {"eta_rho", "xi_rho"}]
    for v in spatial_vars:
        ds[v] = ds[v].where(mask)

    # Load all relevant variables into memory
    for var in ["ALK_ALT_CO2", "DIC_ALT_CO2", "salt", "temp", "PO4", "SiO3"]:
        if var in ds:
            ds[var].load()

    return ds



def compute_sensitivities(
    ds: xr.Dataset,
    ds_ctrl: xr.Dataset,
    alk_varname: str,
    dic_varname: str,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute carbonate sensitivities using PyCO2SYS.

    beta = dDIC / dCO2
    eta  = dDIC / dALK

    Assumes `ds` contains only surface variables
    (e.g., ds = ds.isel(s_rho=-1)).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing surface ALK and DIC.
    ds_ctrl : xr.Dataset
        Control dataset providing salt, temperature, PO4, SiO3.
    alk_varname : str
        Name of alkalinity variable in `ds`.
    dic_varname : str
        Name of DIC variable in `ds`.

    Returns
    -------
    beta : xr.DataArray
        dDIC / dCO2 sensitivity.
    eta : xr.DataArray
        dDIC / dALK sensitivity.
    """
    required_ctrl_vars = ("salt", "temp", "PO4", "SiO3")
    for var in required_ctrl_vars:
        if var not in ds_ctrl:
            raise KeyError(f"`ds_ctrl` is missing required variable '{var}'")

    if alk_varname not in ds or dic_varname not in ds:
        raise KeyError("ALK or DIC variable not found in `ds`")

    rho_factor = 1000.0 / 1025.0  # mmol/m3 → µmol/kg

    ALK = ds[alk_varname] * rho_factor
    DIC = ds[dic_varname] * rho_factor

    csys = pyco2.sys(
        par1=ALK,
        par2=DIC,
        par1_type=1,
        par2_type=2,
        salinity=ds_ctrl.salt,
        temperature=ds_ctrl.temp,
        total_silicate=ds_ctrl.SiO3 * rho_factor,
        total_phosphate=ds_ctrl.PO4 * rho_factor,
    )

    iso_q = csys["isocapnic_quotient"]

    beta = (
        csys["dic"]
        - (csys["HCO3"] + 2.0 * csys["CO3"]) / iso_q
    ) / csys["CO2"]

    eta = 1.0 / iso_q

    return beta, eta


_EXPERIMENT_BASE_DIRS = {"DOR": "roms_marbl_dic", "OAE": "roms_marbl_alk"}
_DEFAULT_GRID_FILE = "roms_marbl_dic/expCTRL/pacmed12_grd.nc"


def load_experiment_alk_dic(
    date: str,
    exp: str,
    scenario: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load surface ALK and DIC (mmol/m3, native model units) from a single-date
    truth-experiment snapshot, masked to ocean cells and flattened.

    Assumes the current working directory is the `pacific` experiment
    directory (as in alk-dic-diagram.ipynb), since paths are resolved as
    `{roms_marbl_dic,roms_marbl_alk}/exp{exp}/OUTPUT/JOINED/pacmed_bgc.{date}*.nc`.

    Parameters
    ----------
    date : str
        Date string matching the file naming, e.g. "20000701" (the file's
        trailing run timestamp is globbed with "*").
    exp : str
        Experiment name, e.g. "JP10".
    scenario : str
        "DOR" (roms_marbl_dic) or "OAE" (roms_marbl_alk).

    Returns
    -------
    alk, dic : np.ndarray
        Flattened, masked surface ALK and DIC values in mmol/m3.
    """
    if scenario not in _EXPERIMENT_BASE_DIRS:
        raise ValueError(f"scenario must be 'DOR' or 'OAE', got {scenario!r}")
    base_dir = _EXPERIMENT_BASE_DIRS[scenario]

    grid = Grid.from_file(_DEFAULT_GRID_FILE, theta_s=6.0, theta_b=6.0, hc=250.0, N=100)

    # roms_marbl_alk's amp=8 site dirs (exp ending in "8": BC8/EC8/JP8/VI8) are currently
    # named exp<SITE>8_perlmutter (Anvil-sourced copies without the suffix aren't reliable yet).
    suffix = "_perlmutter" if base_dir == "roms_marbl_alk" and exp.endswith("8") else ""
    bgc_path = f"{base_dir}/exp{exp}{suffix}/OUTPUT/JOINED/pacmed_bgc.{date}*.nc"
    roms_output = ROMSOutput(grid=grid, path=bgc_path, use_dask=True)
    ds = roms_output.ds

    if "s_rho" in ds.dims:
        ds = ds.isel(s_rho=-1)

    mask = grid.ds.mask_rho
    for var in ("ALK", "DIC"):
        ds[var] = ds[var].where(mask)

    alk = ds["ALK"].values.ravel()
    dic = ds["DIC"].values.ravel()
    valid = np.isfinite(alk) & np.isfinite(dic)
    return alk[valid], dic[valid]


def load_ctrl_alk_dic(date: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load surface ALK and DIC (mmol/m3, native model units) from a single-date
    control-run snapshot, masked to ocean cells and flattened.

    Assumes the current working directory is the `pacific` experiment
    directory (as in alk-dic-diagram.ipynb), since the path is resolved as
    `roms_marbl_dic/expCTRL/SURFACE/pacmed_bgc.{date}*.nc`.

    Parameters
    ----------
    date : str
        Date string matching the file naming, e.g. "20000701" (the file's
        trailing run timestamp is globbed with "*").

    Returns
    -------
    alk, dic : np.ndarray
        Flattened, masked surface ALK and DIC values in mmol/m3.
    """
    grid = Grid.from_file(_DEFAULT_GRID_FILE, theta_s=6.0, theta_b=6.0, hc=250.0, N=100)

    bgc_path = f"roms_marbl_dic/expCTRL/SURFACE/pacmed_bgc.{date}*.nc"
    roms_output = ROMSOutput(grid=grid, path=bgc_path, use_dask=True)
    ds = roms_output.ds

    mask = grid.ds.mask_rho
    for var in ("ALK_ALT_CO2", "DIC_ALT_CO2"):
        ds[var] = ds[var].where(mask)

    alk = ds["ALK_ALT_CO2"].values.ravel()
    dic = ds["DIC_ALT_CO2"].values.ravel()
    valid = np.isfinite(alk) & np.isfinite(dic)
    return alk[valid], dic[valid]
