import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def _add_coastlines_and_gridlines(ax, lon_min, lon_max, lat_min, lat_max):
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle='--',
        x_inline=False,
        y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False

    def get_step(vmin, vmax):
        span = vmax - vmin
        if span > 60: return 20
        if span > 20: return 10
        if span > 5:  return 5
        return 2

    x_step = get_step(lon_min, lon_max)
    y_step = get_step(lat_min, lat_max)
    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(lon_min / x_step) * x_step, lon_max + x_step, x_step)
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(lat_min / y_step) * y_step, lat_max + y_step, y_step)
    )
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


# Single source of truth for each site's regional zoom box, so it only needs to be edited
# here rather than in every notebook that plots a zoomed map. Keyed by both the two-letter
# site code and the full site name, since different notebooks use one or the other.
_ZOOM_COORDS_BY_CODE = {
    "JP": {"lon_min": 132.5, "lon_max": 182.5, "lat_min": 15, "lat_max": 55},
    "VI": {"lon_min": -150, "lon_max": -119, "lat_min": 39, "lat_max": 60},
    "BC": {"lon_min": -130, "lon_max": -97, "lat_min": 10, "lat_max": 40},
    "EC": {"lon_min": -120, "lon_max": -80, "lat_min": -20, "lat_max": 20},
}
_CODE_TO_NAME = {"JP": "Japan", "VI": "Vancouver Island", "BC": "Baja California", "EC": "Ecuador"}
ZOOM_COORDS = {
    **_ZOOM_COORDS_BY_CODE,
    **{name: _ZOOM_COORDS_BY_CODE[code] for code, name in _CODE_TO_NAME.items()},
}


def zoom_box_indices(grid, loc, zoom_coords=None):
    """Bounding (eta_rho, xi_rho) index slices covering a site's regional zoom box (ZOOM_COORDS
    by default), for subsetting grid/output fields before an expensive per-point computation
    (e.g. a full vertical-profile calculation) instead of computing over the whole domain.

    `grid.ds.lon_rho` on this grid runs in [-280, -46) rather than [-180, 180) or [0, 360), so
    the box's lon_min/lon_max (given in an arbitrary convention) are first wrapped into that
    same range before masking, otherwise the comparison silently matches nothing.

    Returns (eta_slice, xi_slice); the enclosed region is a rectangle in index space that
    contains -- but may extend slightly beyond -- the requested lon/lat box, since the grid is
    curvilinear rather than lon/lat-aligned.
    """
    if zoom_coords is None:
        zoom_coords = ZOOM_COORDS
    zc = zoom_coords[loc]
    lon, lat = grid.ds.lon_rho.values, grid.ds.lat_rho.values

    def _wrap(x):
        while x > lon.max():
            x -= 360
        while x < lon.min():
            x += 360
        return x

    lon_min, lon_max = sorted([_wrap(zc["lon_min"]), _wrap(zc["lon_max"])])
    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= zc["lat_min"]) & (lat <= zc["lat_max"])
    if not mask.any():
        raise ValueError(f"No grid points found in the zoom box for {loc!r}.")
    eta_idx, xi_idx = np.where(mask)
    return slice(eta_idx.min(), eta_idx.max() + 1), slice(xi_idx.min(), xi_idx.max() + 1)

# Actual std dev (km) of each site's Gaussian release footprint, from the peak-amplitude
# inversion in flux_amplification_decomposition.ipynb (Section 1: "sigma_eff"). This differs
# from the nominal 300 km used to configure the filter because physical grid spacing shrinks
# toward the poles on this lon/lat grid, so the same filter_scale (in grid cells) is narrower
# at some sites than others.
RELEASE_SIGMA_KM = {
    "JP": 308.8, "Japan": 308.8,
    "VI": 265.6, "Vancouver Island": 265.6,
    "BC": 322.6, "Baja California": 322.6,
    "EC": 340.6, "Ecuador": 340.6,
}
_R90_OVER_SIGMA = np.sqrt(-2 * np.log(0.1))  # ~2.146: radius enclosing 90% of a 2D isotropic Gaussian


def _draw_release_markers(ax, loc, release_centers, radius_km=None):
    """Draw an X at the release center and a geodesic circle of given radius.

    Default radius encloses 90% of the release's actual Gaussian footprint, using that
    site's actual sigma (RELEASE_SIGMA_KM) rather than the nominal 300 km.
    """
    if not release_centers or loc not in release_centers:
        return
    if radius_km is None:
        radius_km = RELEASE_SIGMA_KM.get(loc, 300.0) * _R90_OVER_SIGMA
    cx, cy = release_centers[loc]
    ax.plot(cx, cy, 'kx', markersize=8, markeredgewidth=1.5,
            transform=ccrs.PlateCarree(), zorder=5)
    bearings = np.linspace(0, 2 * np.pi, 361)
    d = radius_km * 1000 / 6371000
    lat0, lon0 = np.radians(cy), np.radians(cx)
    lats = np.arcsin(np.sin(lat0) * np.cos(d) + np.cos(lat0) * np.sin(d) * np.cos(bearings))
    lons = lon0 + np.arctan2(
        np.sin(bearings) * np.sin(d) * np.cos(lat0),
        np.cos(d) - np.sin(lat0) * np.sin(lats),
    )
    ax.plot(np.degrees(lons), np.degrees(lats), 'k--', linewidth=1.5,
            transform=ccrs.PlateCarree(), zorder=5)


def gaussian_footprint_mask(grid, loc, release_centers, radius_km=None):
    """Boolean (eta_rho, xi_rho) mask of grid cells within radius_km of a site's release
    center, via great-circle distance -- the same disc drawn as the dashed circle by
    `_draw_release_markers`. Default radius encloses 90% of the release's actual Gaussian
    footprint (RELEASE_SIGMA_KM), same as that function's default. Time-independent (unlike a
    mass-weighted plume footprint, which grows over the simulation), so it can be reused as a
    fixed spatial mask across all months of a climatology.
    """
    if radius_km is None:
        radius_km = RELEASE_SIGMA_KM.get(loc, 300.0) * _R90_OVER_SIGMA
    cx, cy = release_centers[loc]
    lon, lat = grid.ds.lon_rho.values, grid.ds.lat_rho.values
    lat0, lon0 = np.radians(cy), np.radians(cx)
    lat1, lon1 = np.radians(lat), np.radians(lon)
    dlat, dlon = lat1 - lat0, lon1 - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    dist_km = 2 * 6371 * np.arcsin(np.sqrt(a))
    return dist_km <= radius_km


# Reference points (lon, lat) for labeled cities/towns, so a single edit here updates
# every notebook that marks them. Keyed by name rather than site code since a panel may
# want an arbitrary subset (see DEFAULT_ZOOM_CITIES for per-site defaults).
CITY_COORDS = {
    "Vancouver": (-123.1207, 49.2827),
    "Victoria": (-123.3656, 48.4284),
    "Port Hardy": (-127.4926, 50.7192),
}

# Cities to mark by default on a site's zoomed-in panel, keyed the same way as ZOOM_COORDS.
DEFAULT_ZOOM_CITIES = {
    "VI": ["Vancouver", "Victoria", "Port Hardy"],
    "Vancouver Island": ["Vancouver", "Victoria", "Port Hardy"],
}


_CITY_MARKER_COLOR = "yellow"
_CITY_MARKER_SYMBOLS = ["s", "^", "o", "D", "v", "P"]


def _draw_city_markers(ax, cities, markersize=6, fontsize=10, legend_loc="upper right", color=None):
    """Mark named cities/towns with colored markers, keyed by a legend in `legend_loc`.

    Each city gets a distinct marker symbol (all in the same color) so they stay
    distinguishable without relying on color.

    `cities` is either a list of names looked up in CITY_COORDS, or a dict of
    name -> (lon, lat) for ad hoc locations not worth adding there.
    `color` overrides _CITY_MARKER_COLOR for this call only.
    """
    face_color = color if color is not None else _CITY_MARKER_COLOR
    coords = cities if isinstance(cities, dict) else {name: CITY_COORDS[name] for name in cities}
    for i, (name, (lon, lat)) in enumerate(coords.items()):
        marker = _CITY_MARKER_SYMBOLS[i % len(_CITY_MARKER_SYMBOLS)]
        ax.plot(lon, lat, marker=marker, markersize=markersize, linestyle='none',
                markerfacecolor=face_color, markeredgecolor='black', markeredgewidth=1.0,
                transform=ccrs.PlateCarree(), zorder=5, label=name)
    ax.legend(loc=legend_loc, fontsize=fontsize, framealpha=0.9)


def _plot_colorbar(fig, ax, p, da, vmax, vmin, shrink=0.7, label="", pad=0.15,
                   orientation="horizontal", extend=None):
    """extend : {"neither", "min", "max", "both"}, optional
        By default (None), auto-computed from whether da's actual min/max fall outside
        vmin/vmax. Pass explicitly to override -- e.g. when vmin or vmax is a fixed
        boundary (not a data-driven clip) that should never grow an extend arrow.
    """
    if extend is None:
        extend = "neither"
        if vmax is not None and vmin is not None:
            if da.max() > vmax and da.min() < vmin:
                extend = "both"
            elif da.max() > vmax:
                extend = "max"
            elif da.min() < vmin:
                extend = "min"
    cbar = fig.colorbar(p, ax=ax, orientation=orientation, shrink=shrink,
                        extend=extend, label=label, pad=pad)
    if not isinstance(p.norm, mcolors.SymLogNorm):
        cbar.locator = mticker.MaxNLocator(nbins=4)
        cbar.update_ticks()
    return cbar


def _plot_field(fig, ax, lon, lat, field, field_zoom, cmap, vmin_cb, vmax_cb,
                nr_digits, fontsize=9, cbar_label="", text_color="black", extend=None):
    """Render pcolormesh + vertical colorbar + min/max text annotation.

    field      : full 2-D array passed to pcolormesh (clipped by vmin_cb/vmax_cb)
    field_zoom : masked/zoomed array used only for the min/max text
    cbar_label : optional label drawn next to the colorbar (e.g. units, "%")
    text_color : color of the min/max annotation text. Default "black"; pass "white"
                 for panels with a dark colormap where black text is hard to read.
    extend     : passed through to _plot_colorbar; see its docstring. Default (None)
                 auto-computes from field's full (unzoomed) min/max vs. vmin_cb/vmax_cb.
    """
    p = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vmin_cb, vmax=vmax_cb)
    _plot_colorbar(fig, ax, p, field, vmax_cb, vmin_cb,
                   orientation="vertical", shrink=0.85, pad=0.02, label=cbar_label,
                   extend=extend)
    vmin_data = float(field_zoom.min(skipna=True))
    vmax_data = float(field_zoom.max(skipna=True))
    ax.text(0.5, 0.05,
            f"Min: {vmin_data:.{nr_digits}f}  Max: {vmax_data:.{nr_digits}f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=fontsize, color=text_color)


def _finish_panel(ax, lon_min, lon_max, lat_min, lat_max, loc=None, release_centers=None, cities=None, city_color=None):
    """Apply coastlines, gridlines, and (optionally) release/city markers to a panel."""
    _add_coastlines_and_gridlines(ax, lon_min, lon_max, lat_min, lat_max)
    _draw_release_markers(ax, loc, release_centers)
    if cities:
        _draw_city_markers(ax, cities, color=city_color)


def compute_zoom_box(center, half_width_km):
    """Return (lon_min, lon_max, lat_min, lat_max) of a box `half_width_km` around `center` = (lon, lat)."""
    cx, cy = center
    dlat = half_width_km / 111.0
    dlon = half_width_km / (111.0 * np.cos(np.radians(cy)))
    return cx - dlon, cx + dlon, cy - dlat, cy + dlat


def _draw_zoom_box(ax, lon_min, lon_max, lat_min, lat_max, color="black", linewidth=1.2, linestyle="dotted"):
    """Outline a lon/lat box on `ax`, e.g. to indicate a subregion zoomed into elsewhere."""
    box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(box_lons, box_lats, color=color, linewidth=linewidth, linestyle=linestyle,
            transform=ccrs.PlateCarree(), zorder=6)


def _diff_limits(diff_zoom, cb_overrides, key):
    """Return (vmin_cb, vmax_cb) for a difference panel.

    Symmetric around zero by default; values from cb_overrides[key] take priority.
    """
    vabs = float(max(
        abs(float(diff_zoom.min(skipna=True))),
        abs(float(diff_zoom.max(skipna=True))),
    ))
    if key in cb_overrides:
        return cb_overrides[key].get("vmin", -vabs), cb_overrides[key].get("vmax", vabs)
    return -vabs, vabs


def get_symlog_levels(vmin, vmax, linthresh, n_neg=3, n_pos=3):
    levels = []
    if vmin < -linthresh:
        neg_max = 10 ** np.floor(np.log10(abs(vmin)))
        neg_levels = -np.logspace(np.log10(linthresh), np.log10(neg_max), n_neg)[::-1]
        levels.extend(neg_levels)
    levels.append(0.0)
    if vmax > linthresh:
        pos_max = 10 ** np.ceil(np.log10(vmax))
        pos_levels = np.logspace(np.log10(linthresh), np.log10(pos_max), n_pos)
        levels.extend(pos_levels)
    return np.array(levels)


def get_label_string(da):
    max_val = np.abs(da).max().compute().item()
    formatted = np.format_float_positional(max_val, precision=3, unique=True, fractional=False, trim="-")
    label_string = f"Max Magnitude = {formatted}"
    return label_string
