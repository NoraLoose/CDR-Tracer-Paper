"""Analytic CDR-tracer vs truth curves — a playground for β, η, mixing depth.

Everything here is closed-form; there is no ODE solver and no model output.

The model
---------
One surface box of effective mixing depth ``h``, piston velocity ``k``, intervention
rate ``S`` applied for ``0 ≤ t ≤ T``.  Working with the *disequilibrium*
``G ≡ η·A − D`` (which is what the flux is proportional to, ``F = h·a·G``) collapses
the system to one linear ODE, ``dG/dt + aG = η·(dA/dt)``, with

    a = (k/β)/h            surface relaxation rate

giving

    F(t) = η·S · (1 − exp(−a·min(t,T))) · exp(−a·max(t−T, 0))
             \_______  spin-up  _______/  \___ relaxation ___/

Normalised so the total intervention is 1 (``S·T = 1``), the time-integral of ``F``
*is* the efficiency curve and plateaus at ``η``.

**DOR is the same expressions with η = 1** (there the deficit *is* the
disequilibrium), so its efficiency plateaus at 1.

What to expect when you play
----------------------------
A perturbation depresses surface CO₂*, so ``β_true > β_control`` and therefore
``a_c > a_t``: the CDR tracer relaxes *faster* than the truth.  Consequences you can
check by moving the sliders:

* **Instantaneous flux always crosses.**  The CDR tracer starts higher and decays
  faster, so the curves must intersect.  The crossover sits near ``t* ≈ 1/a_c = h·β/k``
  and is nearly independent of the size of the β bias (the ε cancels — see ``t_star``
  vs ``tau_c`` in the printout).
* **Cumulative efficiency never crosses**, as long as ``β_true ≥ β_control`` *and*
  ``η_c ≥ η_t``.  Slower removal retains more inventory at all times, so the truth has
  accumulated less uptake at all times.  Setting ``eta_t > eta_c`` is the only way to
  make the cumulative cross in this model — the sliders let you see what that takes.
* **Deployment length matters.**  Peak flux is ``η·S·(1 − exp(−a·T))``, so a deployment
  short compared with ``1/a`` never approaches the balancing level ``η·S``.

See ``antitracer-flux-sign-crossover.md`` and ``antitracer_crossover_math.ipynb``.
"""

import numpy as np
import matplotlib.pyplot as plt

DAY = 86400.0

# palette (categorical slots 1-2, light surface) + chart chrome
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CDR, TRUTH, NEG = "#2a78d6", "#eb6834", "#e34948"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

__all__ = ["flux", "cumulative", "rate", "curves", "plot", "sweep", "explore", "DAY"]


# ----------------------------------------------------------------------------- core
def rate(beta, k=4.0e-3, h_m=15.0):
    """Surface relaxation rate a = (k/β)/h.  k in cm/s, h in metres.  Returns 1/s."""
    return (k / beta) / (h_m * 100.0)


def flux(t, a, eta, T):
    """F(t), normalised so the total intervention is 1 (S = 1/T).  t, T in seconds."""
    spin_up = 1.0 - np.exp(-a * np.minimum(t, T))
    relax = np.exp(-a * np.maximum(t - T, 0.0))
    return eta * (1.0 / T) * spin_up * relax


def cumulative(t, a, eta, T):
    """Time-integral of flux() — the efficiency curve.  Plateaus at eta."""
    t1 = np.minimum(t, T)
    C = eta * (1.0 / T) * (t1 - (1.0 - np.exp(-a * t1)) / a)
    t2 = np.maximum(t - T, 0.0)
    C += eta * (1.0 / T) * (1.0 - np.exp(-a * T)) * (1.0 - np.exp(-a * t2)) / a
    return C


def _first_crossing(t_days, y, tol=0.0):
    """First time y goes from > tol to < -tol (downward crossing), else None."""
    below = y < -tol
    if not below.any() or below[0]:
        return None
    return float(t_days[np.argmax(below)])


def curves(beta_c=15.0, beta_t=17.25, eta_c=0.85, eta_t=0.82,
           k=4.0e-3, h_m=15.0, T_days=30.0, years=2.0, n=4001):
    """Analytic CDR-tracer and truth curves for one parameter set.

    Parameters
    ----------
    beta_c, beta_t : control (prescribed) and true buffer factor, dDIC/dCO2.  O(5-30).
    eta_c, eta_t   : control and true dDIC/dALK = 1/Q.  O(0.8-0.9).  Use 1.0, 1.0 for DOR.
    k              : piston velocity, cm/s.
    h_m            : effective mixing depth, metres.
    T_days         : deployment length.
    years          : length of the record to generate.

    Returns a dict.  Errors follow the convention used in the analysis notebooks:
    ``err_flux`` and ``err_eff`` are **CDR tracer − truth**, so a positive value means
    the CDR tracer is over-estimating.
    """
    T = T_days * DAY
    a_c, a_t = rate(beta_c, k, h_m), rate(beta_t, k, h_m)
    t = np.linspace(0.0, years * 365 * DAY, n)
    td = t / DAY

    F_cdr, F_truth = flux(t, a_c, eta_c, T), flux(t, a_t, eta_t, T)
    C_cdr, C_truth = cumulative(t, a_c, eta_c, T), cumulative(t, a_t, eta_t, T)
    err_flux, err_eff = F_cdr - F_truth, C_cdr - C_truth

    return dict(
        t=t, t_days=td, t_years=td / 365.0,
        a_c=a_c, a_t=a_t,
        tau_c_days=1.0 / a_c / DAY, tau_t_days=1.0 / a_t / DAY,
        # fluxes reported per unit intervention per day, x1e-3 for readability
        F_cdr=F_cdr * DAY * 1e3, F_truth=F_truth * DAY * 1e3,
        C_cdr=C_cdr, C_truth=C_truth,
        err_flux=err_flux * DAY * 1e3, err_eff=err_eff,
        t_star_flux=_first_crossing(td, err_flux),
        t_star_eff=_first_crossing(td, err_eff, tol=1e-9),
        peak_flux_cdr=float(np.max(F_cdr * DAY * 1e3)),
        aT_c=a_c * T, eps=(beta_t - beta_c) / beta_c,
        params=dict(beta_c=beta_c, beta_t=beta_t, eta_c=eta_c, eta_t=eta_t,
                    k=k, h_m=h_m, T_days=T_days, years=years),
    )


def summary(c):
    """One-line-per-fact printout for a curves() result."""
    p = c["params"]
    ts_f = c["t_star_flux"]
    ts_e = c["t_star_eff"]
    print(f"beta {p['beta_c']:.1f} -> {p['beta_t']:.1f} (eps={c['eps']:+.2f})   "
          f"eta {p['eta_c']:.3f} -> {p['eta_t']:.3f}   "
          f"k={p['k']:.1e} cm/s   h={p['h_m']:.0f} m   T={p['T_days']:.0f} d")
    print(f"  equilibration 1/a : CDR {c['tau_c_days']:7.1f} d      truth {c['tau_t_days']:7.1f} d")
    print(f"  a*T (deployment)  : {c['aT_c']:.3f}  -> peak flux reaches "
          f"{100*(1-np.exp(-c['aT_c'])):.0f}% of eta*S")
    if ts_f is None:
        print("  INSTANTANEOUS flux crossover    : none in record")
    else:
        print(f"  INSTANTANEOUS flux crossover    : {ts_f:.0f} d = {ts_f/30:.2f} months"
              f"   ({ts_f*DAY*c['a_c']:.2f} x 1/a_c)")
    if ts_e is None:
        print("  CUMULATIVE efficiency crossover : none"
              "   (expected whenever beta_t>=beta_c and eta_c>=eta_t)")
    else:
        print(f"  CUMULATIVE efficiency crossover : {ts_e:.0f} d = {ts_e/365:.2f} yr")
    print(f"  efficiency plateau : CDR {p['eta_c']:.3f}   truth {p['eta_t']:.3f}")


# ------------------------------------------------------------------------- plotting
def _style(ax, ylab, title, xlab="days since start of deployment"):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_ylabel(ylab, color=INK2, fontsize=9.5)
    ax.set_xlabel(xlab, color=INK2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=9, fontweight="bold")


def plot(c=None, **kw):
    """Four-panel view of one parameter set.  Pass a curves() dict or curves() kwargs.

    Errors are plotted as **CDR tracer − truth** (positive = CDR tracer too high).
    """
    if c is None:
        c = curves(**kw)
    td = c["t_days"]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2), facecolor=SURF)
    fig.subplots_adjust(hspace=0.46, wspace=0.26, top=0.855, bottom=0.095,
                        left=0.095, right=0.975)

    ax = axs[0, 0]
    _style(ax, "ΔFCO₂  (10⁻³ per unit intervention, day⁻¹)", "Instantaneous ΔFCO₂")
    ax.plot(td, c["F_cdr"], color=CDR, lw=2, zorder=3)
    ax.plot(td, c["F_truth"], color=TRUTH, lw=2, zorder=3)

    ax = axs[0, 1]
    _style(ax, "cumulative uptake  (efficiency)", "Cumulative uptake — efficiency curve")
    ax.plot(td, c["C_cdr"], color=CDR, lw=2, zorder=3)
    ax.plot(td, c["C_truth"], color=TRUTH, lw=2, zorder=3)
    ax.set_ylim(0, max(1.05, c["C_cdr"].max() * 1.15))

    ax = axs[1, 0]
    _style(ax, "CDR tracer − truth  (10⁻³)", "Error in instantaneous ΔFCO₂")
    e = c["err_flux"]
    ax.fill_between(td, 0, e, where=e >= 0, color=CDR, alpha=0.15, lw=0, zorder=1)
    ax.fill_between(td, 0, e, where=e < 0, color=NEG, alpha=0.15, lw=0, zorder=1)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.plot(td, e, color=INK, lw=2, zorder=3)

    ax = axs[1, 1]
    _style(ax, "CDR tracer − truth", "Error in cumulative efficiency")
    e = c["err_eff"]
    ax.fill_between(td, 0, e, where=e >= 0, color=CDR, alpha=0.15, lw=0, zorder=1)
    ax.fill_between(td, 0, e, where=e < 0, color=NEG, alpha=0.15, lw=0, zorder=1)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.plot(td, e, color=INK, lw=2, zorder=3)

    for ax, key in ((axs[0, 0], "t_star_flux"), (axs[1, 0], "t_star_flux"),
                    (axs[0, 1], "t_star_eff"), (axs[1, 1], "t_star_eff")):
        if c[key] is not None:
            ax.axvline(c[key], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)

    h1, = axs[0, 0].plot([], [], color=CDR, lw=2, label="CDR tracer  (control β, η)")
    h2, = axs[0, 0].plot([], [], color=TRUTH, lw=2, label="truth  (perturbed β, η)")
    fig.legend(handles=[h1, h2], loc="upper left", bbox_to_anchor=(0.095, 0.952),
               ncol=2, frameon=False, fontsize=10, labelcolor=INK2)
    p = c["params"]
    ts = c["t_star_flux"]
    fig.suptitle(f"β {p['beta_c']:.1f}→{p['beta_t']:.1f}   η {p['eta_c']:.2f}→{p['eta_t']:.2f}   "
                 f"h={p['h_m']:.0f} m   1/a_c={c['tau_c_days']:.0f} d   "
                 f"flux crossover {f'{ts:.0f} d' if ts else 'none'}",
                 x=0.095, y=0.975, ha="left", color=INK, fontsize=13, fontweight="bold")
    return fig


def sweep(param, values, quantity="err_flux", **kw):
    """Overlay one varying parameter.  quantity: err_flux | err_eff | F_cdr | C_cdr."""
    lab = {"err_flux": "CDR tracer − truth  (10⁻³ day⁻¹)",
           "err_eff": "CDR tracer − truth  (efficiency)",
           "F_cdr": "ΔFCO₂ (10⁻³ day⁻¹)", "C_cdr": "efficiency"}[quantity]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURF)
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.11, right=0.97)
    _style(ax, lab, f"Varying {param}")
    cols = [RAMP[int(round(i))] for i in
            np.linspace(1, len(RAMP) - 1, max(len(values), 2))]
    for v, col in zip(values, cols):
        c = curves(**{**kw, param: v})
        ax.plot(c["t_days"], c[quantity], color=col, lw=2, zorder=3,
                label=f"{param} = {v:g}")
    if quantity.startswith("err"):
        ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    return fig


def explore(**defaults):
    """ipywidgets sliders over the same analytic curves.  Needs a live kernel."""
    from ipywidgets import interact, FloatSlider

    d = dict(beta_c=15.0, beta_t=17.25, eta_c=0.85, eta_t=0.82,
             k=4.0e-3, h_m=15.0, T_days=30.0, years=2.0)
    d.update(defaults)

    def _show(beta_c, beta_t, eta_c, eta_t, k, h_m, T_days, years):
        c = curves(beta_c=beta_c, beta_t=beta_t, eta_c=eta_c, eta_t=eta_t,
                   k=k, h_m=h_m, T_days=T_days, years=years)
        summary(c)
        plot(c)
        plt.show()

    return interact(
        _show,
        beta_c=FloatSlider(value=d["beta_c"], min=3, max=40, step=0.5, description="β control"),
        beta_t=FloatSlider(value=d["beta_t"], min=3, max=40, step=0.5, description="β true"),
        eta_c=FloatSlider(value=d["eta_c"], min=0.70, max=1.00, step=0.005, description="η control"),
        eta_t=FloatSlider(value=d["eta_t"], min=0.70, max=1.00, step=0.005, description="η true"),
        k=FloatSlider(value=d["k"], min=1e-3, max=1.5e-2, step=5e-4, readout_format=".4f",
                      description="k (cm/s)"),
        h_m=FloatSlider(value=d["h_m"], min=5, max=200, step=5, description="h (m)"),
        T_days=FloatSlider(value=d["T_days"], min=1, max=365, step=1, description="T (days)"),
        years=FloatSlider(value=d["years"], min=0.5, max=15, step=0.5, description="record (yr)"),
    )
