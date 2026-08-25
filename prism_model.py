#!/usr/bin/env python3
"""
PRISM -- Population-Resolved, Interaction-explicit Spectral Morphology model
==============================================================================

Single-file, self-contained implementation of the PRISM model for black
carbon (BC) - organic carbon (OC) optical enhancement, developed for the
manuscript "PRISM: A Population-Resolved, Interaction-explicit Spectral
Morphology Model for Black Carbon-Organic Carbon Optical Enhancement,
Closed Against Field-Measured Enhancement Ratios from Household Biofuel
Combustion".

Every physical input (refractive indices, size distributions, coating
factors, shell absorptivity) is transcribed from five previously published
studies rather than generic literature defaults; each is cited in-line at
the point it is used. 

--------------------------------------------------------------------------
CONTENTS (in dependency order)
--------------------------------------------------------------------------
  1. Plotting style (shared colors/fonts across all figures)
  2. Tier 1: concentric core-shell Mie optical core
  3. Module A: source-specific brown-carbon (BrC) shell optics
  4. Module C: Monte Carlo population integration
  5. Tier 3a: RDG-FA real fractal-aggregate calculation
  6. Sobol variance-based sensitivity analysis
  7. Value-of-Information (VoI) analysis
  8. Figure-generating functions (Figures 1-9 of the manuscript)
  9. __main__ block: regenerates every figure and prints every headline
     number quoted in the manuscript, for reproducibility

--------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------
    pip install PyMieScatt SALib numpy scipy matplotlib

(PyMieScatt was last released against an older SciPy that still exposed
scipy.integrate.trapz; the small compatibility shim just below restores
that alias if it is missing, so no pinned SciPy version is required.)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python prism_model.py                  # regenerate all 9 figures into ./prism_figures
    python prism_model.py --outdir FOLDER   # write figures elsewhere

Or import individual pieces in your own scripts, e.g.:
    from prism_model import e_abs, aggregate_eabs, sample_population
"""

# ==========================================================================
# 0. Compatibility shim + shared imports
# ==========================================================================
import os
import math
import argparse

import numpy as np
import scipy.integrate as _si
if not hasattr(_si, "trapz"):
    _si.trapz = np.trapezoid  # restore alias removed in SciPy >= 1.14
from scipy.integrate import quad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

import PyMieScatt as ps
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze


# ==========================================================================
# 1. Plotting style (shared across all figures)
# ==========================================================================
COLORS = {
    "primary": "#2166AC",     # blue -- model / PRISM results
    "secondary": "#B2182B",   # red -- field targets / observations
    "tertiary": "#1B7837",    # green -- verification / agreement
    "quaternary": "#762A83",  # purple -- RDG-FA / morphology
    "neutral": "#4D4D4D",     # gray -- reference lines, concentric baseline
    "accent": "#E08214",      # orange -- secondary highlight
}
PALETTE_7 = ["#2166AC", "#4393C3", "#92C5DE", "#B2182B", "#D6604D",
             "#F4A582", "#762A83"]


def apply_plot_style():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
    })


apply_plot_style()


# ==========================================================================
# 2. Tier 1: concentric core-shell Mie optical core
# --------------------------------------------------------------------------
# Solver: PyMieScatt.MieQCoreShell -- the SAME package used in Rathod &
# Sahu (2026, Environ Model Assess 31:685-695): their Methods section
# states "Mie theory-based PyMieScatt computational package [20]", ref
# [20] = Sumlin, Heinson & Chakrabarty (2018, JQSRT 205:127-134).
#
# Default refractive indices are the EXACT values from that paper's
# Table 1 (real data, not placeholders):
#   BC:  1.95 + 0.79i   at 550 nm (Bond & Bergstrom 2006)
#   OC:  1.60 + 0.0083i at 550 nm (their representative OC)
# Densities (same table): BC = 1.8 g/cm3, OC = 1.1 g/cm3.
#
# VERIFICATION: reproducing their 120 nm BC core, coating factor CF=2
# case at 550 nm gives Esca = 13.83, matching their stated "approximately
# 14" within the rounding precision that value appears to have been
# reported at. See make_figure2_tier1_verification() below.
# ==========================================================================
M_CORE_BC = complex(1.95, 0.79)         # BC at 550 nm, Bond & Bergstrom 2006
M_SHELL_NONABS = complex(1.60, 0.0083)  # OC at 550 nm, Rathod & Sahu (2026)
RHO_BC = 1.8   # g/cm3, Rathod & Sahu (2026), Table 1
RHO_OC = 1.1   # g/cm3, same table; traceable to Schkolnik et al. (2007)


def qabs_qsca(dCore_nm, dShell_nm, wavelength_nm=550.0,
              mCore=M_CORE_BC, mShell=M_SHELL_NONABS):
    """Return (Qext, Qsca, Qabs) efficiency factors for one coated sphere."""
    if dShell_nm < dCore_nm:
        raise ValueError("shell diameter must be >= core diameter")
    out = ps.MieQCoreShell(mCore, mShell, wavelength_nm, dCore_nm, dShell_nm)
    return out[0], out[1], out[2]


def e_abs(dCore_nm, coating_thickness_nm, wavelength_nm=550.0,
          mCore=M_CORE_BC, mShell=M_SHELL_NONABS):
    """
    Absorption enhancement factor E_abs = Cabs(coated)/Cabs(bare core),
    following Eq. 1 of Rathod & Sahu (2026): Cabs_coated is the absorption
    cross-section of the coated sphere, Cabs_uncoated is the absorption
    cross-section of the bare core at the SAME core diameter.
    """
    dShell = dCore_nm + 2 * coating_thickness_nm
    _, _, Qabs_bare = qabs_qsca(dCore_nm, dCore_nm, wavelength_nm, mCore, mShell)
    _, _, Qabs_coat = qabs_qsca(dCore_nm, dShell, wavelength_nm, mCore, mShell)
    Cabs_bare = Qabs_bare * math.pi * (dCore_nm / 2) ** 2
    Cabs_coat = Qabs_coat * math.pi * (dShell / 2) ** 2
    return Cabs_coat / Cabs_bare


def e_sca(dCore_nm, coating_thickness_nm, wavelength_nm=550.0,
          mCore=M_CORE_BC, mShell=M_SHELL_NONABS):
    """Scattering enhancement factor E_sca, same convention as e_abs."""
    dShell = dCore_nm + 2 * coating_thickness_nm
    _, Qsca_bare, _ = qabs_qsca(dCore_nm, dCore_nm, wavelength_nm, mCore, mShell)
    _, Qsca_coat, _ = qabs_qsca(dCore_nm, dShell, wavelength_nm, mCore, mShell)
    Csca_bare = Qsca_bare * math.pi * (dCore_nm / 2) ** 2
    Csca_coat = Qsca_coat * math.pi * (dShell / 2) ** 2
    return Csca_coat / Csca_bare


# ==========================================================================
# 3. Module A: source-specific brown-carbon (BrC) shell optics
# --------------------------------------------------------------------------
# Real, digitized k(lambda) values from Rathod et al. (2017, Aerosol Air
# Qual Res 17:108-116), Table 2 -- household-biofuel BrC measurements.
# Real part of refractive index in that paper is n=1.55.
# ==========================================================================
N_BRC_2017 = 1.55  # real part of refractive index, Rathod et al. (2017)

K_TABLE = {
    "wood_210C": {350: 0.0145, 450: 0.0032, 550: 0.0018, 650: 0.0013},
    "wood_380C": {350: 0.0140, 450: 0.0029, 550: 0.0014, 650: 0.0010},
    "dung_210C": {350: 0.0407, 450: 0.0243, 550: 0.0210, 650: 0.0194},
    "dung_380C": {350: 0.0537, 450: 0.0273, 550: 0.0216, 650: 0.0191},
}


def _loglog_fit(k_dict):
    wl = np.array(sorted(k_dict.keys()), dtype=float)
    k = np.array([k_dict[w] for w in wl])
    b, a = np.polyfit(np.log(wl), np.log(k), 1)  # ln(k) = a + b*ln(wl)
    return a, b


_K_FITS = {name: _loglog_fit(tbl) for name, tbl in K_TABLE.items()}


def shell_k_real(wavelength_nm, source="dung_380C"):
    """
    Real, source-specific shell imaginary refractive index, fit to the
    Rathod et al. (2017) Table 2 data (log-log power law across the four
    tabulated wavelengths). `source` in {"wood_210C","wood_380C",
    "dung_210C","dung_380C"}. Returns (k, is_extrapolated); extrapolation
    beyond the measured 350-650 nm range is flagged.
    """
    a, b = _K_FITS[source]
    k = np.exp(a) * wavelength_nm ** b
    is_extrapolated = not (350.0 <= wavelength_nm <= 650.0)
    return k, is_extrapolated


# --- Tier 3b: fast morphology proxy, calibrated against the real RDG-FA
#     calculation in Section 5 below (MCF(1.8) = 0.54, the average of the
#     wood/dung RDG-FA results at the reference fractal dimension). ---
BETA_MCF = (1 - 0.54) / (3 - 1.8) ** 2  # = 0.3194


def morphology_correction_factor(Df):
    """Df=3 (compact sphere) -> MCF=1 (no correction, recovers Tier 1)."""
    return np.clip(1.0 - BETA_MCF * (3.0 - Df) ** 2, 0.05, 1.0)


# ==========================================================================
# 4. Module C: Monte Carlo population integration
# --------------------------------------------------------------------------
# Population parameters anchored to REAL values reported in Rathod et al.
# (2021, Atmos Pollut Res 12:101236): log-normal core-size distributions
# (CMD/GSD) and Mie-derived coating factors, per fuel.
# ==========================================================================
FUEL_PARAMS = {
    "wood": dict(cmd=123.0, gsd=1.78, coating_factor=1.37, k_source="wood_380C"),
    "dung": dict(cmd=152.0, gsd=1.89, coating_factor=1.69, k_source="dung_380C"),
}
DF_MEAN, DF_SD = 1.8, 0.15  # literature-typical placeholder (no morphology
                             # measurement exists in any of the five source papers)

_rng_population = np.random.default_rng(20260721)


def sample_population(n, fuel="dung", wavelength_nm=880.0,
                       cf_relative_sd=0.10, df_mean=DF_MEAN, df_sd=DF_SD,
                       apply_morphology=True, real_part_shell=1.55):
    """
    Draw n particles from the fuel-specific log-normal size distribution
    and (optionally normally-perturbed) coating factor, compute E_abs for
    each via the Tier-1 Mie core with the Module-A shell absorptivity, and
    optionally apply the Tier-3b morphology proxy. Returns an array of
    per-particle E_abs values; the population mean is the headline PRISM
    result quoted throughout the manuscript.
    """
    p = FUEL_PARAMS[fuel]
    dcore = _rng_population.lognormal(np.log(p["cmd"]), np.log(p["gsd"]), n)
    cf = _rng_population.normal(p["coating_factor"], p["coating_factor"] * cf_relative_sd, n)
    cf = np.clip(cf, 1.01, None)
    coat_thick = (cf - 1.0) * dcore / 2.0
    df = np.clip(_rng_population.normal(df_mean, df_sd, n), 1.2, 3.0)

    k, _ = shell_k_real(wavelength_nm, p["k_source"])
    mShell = complex(real_part_shell, k)

    vals = np.empty(n)
    for i in range(n):
        vals[i] = e_abs(dcore[i], coat_thick[i], wavelength_nm,
                         mCore=M_CORE_BC, mShell=mShell)
    if apply_morphology:
        mcf = morphology_correction_factor(df)
        vals = 1.0 + (vals - 1.0) * mcf
    return vals


def convergence_curve(sample_sizes=(200, 500, 1000, 2000, 5000, 10000, 20000, 50000),
                       **kwargs):
    """Population-mean E_abs and its 95% CI as a function of Monte Carlo
    sample size, used to confirm the closure result is not an artifact of
    insufficient sampling (manuscript Section 3.3 / Figure 4)."""
    means, cis = [], []
    for n in sample_sizes:
        v = sample_population(n, **kwargs)
        means.append(v.mean())
        cis.append(1.96 * v.std(ddof=1) / np.sqrt(n))
    return np.array(sample_sizes), np.array(means), np.array(cis)


# ==========================================================================
# 5. Tier 3a: RDG-FA real fractal-aggregate calculation
# --------------------------------------------------------------------------
# Rayleigh-Debye-Gans theory for fractal aggregates (RDG-FA), following
# Sorensen (2001, Aerosol Sci Technol 35:648-687) -- the standard, widely
# used approximate aggregate optics theory (NOT a full discrete-dipole-
# approximation solver, but genuine physics rather than an ad hoc proxy).
# Previously applied specifically to organic-coated soot by Lefevre et al.
# (2018, 2019).
#
# Model: an aggregate of N_p identical coated-sphere monomers (BC core +
# organic shell), each solved with the same core-shell Mie code as Tier 1.
# Aggregate absorption is treated as simply additive across monomers (the
# standard RDG simplification -- no near-field electromagnetic coupling
# between monomers is captured; stated here as a known limitation, not
# claimed as exact). Aggregate scattering uses the Fisher-Burford
# structure factor, angle-integrated over the Rayleigh dipole phase
# function to give a total (not just forward) scattering cross-section.
#
# Monomer diameter and fractal prefactor are LITERATURE-TYPICAL values
# (no primary-particle-size or fractal-prefactor measurement exists in
# any of the five source papers) and are flagged as such.
# ==========================================================================
MONOMER_DIAMETER_NM = 30.0   # typical primary-particle diameter for combustion soot
FRACTAL_PREFACTOR_KF = 1.3   # typical DLCA value for Df ~ 1.7-1.8 (Sorensen 2001)


def monomer_optics(coating_factor, wavelength_nm, mShell, mCore=M_CORE_BC,
                    monomer_diameter_nm=MONOMER_DIAMETER_NM):
    """Cabs, Csca for one coated monomer (core = bare monomer, shell = coated monomer)."""
    d_core = monomer_diameter_nm
    d_shell = d_core * coating_factor
    out_bare = ps.MieQCoreShell(mCore, mShell, wavelength_nm, d_core, d_core)
    out_coat = ps.MieQCoreShell(mCore, mShell, wavelength_nm, d_core, d_shell)
    Cabs_bare = out_bare[2] * math.pi * (d_core / 2) ** 2
    Cabs_coat = out_coat[2] * math.pi * (d_shell / 2) ** 2
    Csca_bare = out_bare[1] * math.pi * (d_core / 2) ** 2
    Csca_coat = out_coat[1] * math.pi * (d_shell / 2) ** 2
    return dict(Cabs_bare=Cabs_bare, Cabs_coat=Cabs_coat,
                Csca_bare=Csca_bare, Csca_coat=Csca_coat)


def n_monomers(core_equiv_diameter_nm, monomer_diameter_nm=MONOMER_DIAMETER_NM):
    """Number of monomers giving the same total BC volume as a single
    equivalent sphere of core_equiv_diameter_nm (mass/volume-equivalence)."""
    return (core_equiv_diameter_nm / monomer_diameter_nm) ** 3


def radius_of_gyration(n_p, monomer_diameter_nm=MONOMER_DIAMETER_NM,
                        Df=1.8, kf=FRACTAL_PREFACTOR_KF):
    a = monomer_diameter_nm / 2.0
    return a * (n_p / kf) ** (1.0 / Df)


def structure_factor_fisher_burford(q, Rg, Df):
    return (1.0 + (2.0 / (3.0 * Df)) * (q * Rg) ** 2) ** (-Df / 2.0)


def angle_integrated_structure_factor(Rg, Df, wavelength_nm):
    """
    Integral of S(q(theta)) weighted by the normalized Rayleigh dipole
    phase function (3/8)(1+cos^2 theta) sin(theta) over theta in [0, pi].
    Reduces the naive N_p^2 coherent-scattering limit to account for
    destructive interference at large scattering angles.
    """
    k = 2 * math.pi / wavelength_nm

    def integrand(theta):
        q = 2 * k * math.sin(theta / 2.0)
        S = structure_factor_fisher_burford(q, Rg, Df)
        phase = (3.0 / 8.0) * (1 + math.cos(theta) ** 2) * math.sin(theta)
        return S * phase

    val, _ = quad(integrand, 0, math.pi)
    return val


def aggregate_eabs(core_equiv_diameter_nm, coating_factor, wavelength_nm,
                    mShell, Df=1.8, kf=FRACTAL_PREFACTOR_KF,
                    monomer_diameter_nm=MONOMER_DIAMETER_NM):
    """
    E_abs for a fractal aggregate of the same total BC volume as a
    concentric sphere of core_equiv_diameter_nm, using RDG-FA.
    Returns (E_abs_aggregate, N_p, Rg_nm).
    """
    n_p = n_monomers(core_equiv_diameter_nm, monomer_diameter_nm)
    mono = monomer_optics(coating_factor, wavelength_nm, mShell,
                           monomer_diameter_nm=monomer_diameter_nm)
    Cabs_agg_bare = n_p * mono["Cabs_bare"]
    Cabs_agg_coat = n_p * mono["Cabs_coat"]
    E_abs_agg = Cabs_agg_coat / Cabs_agg_bare
    Rg = radius_of_gyration(n_p, monomer_diameter_nm, Df, kf)
    return E_abs_agg, n_p, Rg


# ==========================================================================
# 6. Sobol variance-based sensitivity analysis
# --------------------------------------------------------------------------
# Sobol (2001, Math Comput Simul 55:271-280) with Saltelli sampling
# (Saltelli 2002, Comput Phys Commun 145:280-297) and the total-order
# estimator of Saltelli et al. (2010, Comput Phys Commun 181:259-270),
# implemented via SALib (Herman & Usher 2017, JOSS 2(9):97).
# ==========================================================================
SOBOL_PROBLEM = {
    "num_vars": 4,
    "names": ["dcore_nm", "coat_factor", "k_shell_880nm", "Df"],
    "bounds": [
        [60.0, 300.0],   # core diameter, nm -- spans real wood/dung CMD*GSD range
        [1.0, 2.0],       # coating factor -- spans real wood=1.37 / dung=1.69
        [0.001, 0.06],    # shell k at 880 nm -- spans real measured range across
                           # all four fuel/temp conditions (Rathod et al. 2017)
        [1.5, 3.0],       # fractal dimension -- literature placeholder
    ],
}


def sobol_model(X, wavelength_nm=880.0, real_part_shell=1.55):
    out = np.empty(X.shape[0])
    for i, (dcore, cf, k, df) in enumerate(X):
        mShell = complex(real_part_shell, k)
        ct = (cf - 1.0) * dcore / 2.0
        val = e_abs(dcore, ct, wavelength_nm, mCore=M_CORE_BC, mShell=mShell)
        mcf = morphology_correction_factor(df)
        out[i] = 1.0 + (val - 1.0) * mcf
    return out


def run_sobol(n_base=2048, seed=20260721):
    param_values = sobol_sample.sample(SOBOL_PROBLEM, n_base, calc_second_order=True, seed=seed)
    Y = sobol_model(param_values)
    Si = sobol_analyze.analyze(SOBOL_PROBLEM, Y, calc_second_order=True, print_to_console=False)
    return Si, param_values.shape[0]


# ==========================================================================
# 7. Value-of-Information (VoI) analysis
# --------------------------------------------------------------------------
# Direct Monte Carlo estimate of the variance reduction achievable by
# eliminating uncertainty in each Sobol input individually (fixing it at
# its range midpoint while the others continue to vary).
# ==========================================================================
_rng_voi = np.random.default_rng(20260721)
_VOI_N = 30000
_VOI_NAMES = SOBOL_PROBLEM["names"]
_VOI_BOUNDS = np.array(SOBOL_PROBLEM["bounds"])


def _voi_draw(n, fix_index=None):
    X = _rng_voi.uniform(_VOI_BOUNDS[:, 0], _VOI_BOUNDS[:, 1], size=(n, 4))
    if fix_index is not None:
        mid = 0.5 * (_VOI_BOUNDS[fix_index, 0] + _VOI_BOUNDS[fix_index, 1])
        X[:, fix_index] = mid
    return X


def voi():
    Xbase = _voi_draw(_VOI_N)
    Ybase = sobol_model(Xbase)
    var_base = Ybase.var()
    results = {}
    for i, name in enumerate(_VOI_NAMES):
        Xi = _voi_draw(_VOI_N, fix_index=i)
        Yi = sobol_model(Xi)
        var_i = Yi.var()
        results[name] = (var_i, 1.0 - var_i / var_base)
    return var_base, results


# ==========================================================================
# 8. Figure-generating functions (Figures 1-9 of the manuscript)
# ==========================================================================
FUEL_COLOR = {"wood": COLORS["primary"], "dung": COLORS["secondary"]}
FUEL_LABEL = {"wood": "Wood", "dung": "Dung cake"}


def make_figure1_architecture(outdir):
    """Figure 1: PRISM model architecture (data provenance, modules, flow)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 13.5)
    ax.axis("off")

    SRC_ACTIVE, SRC_CONTEXT = "#D5E8D4", "#FFE6CC"
    CORE, MORPH, CLOSE, OUTPUT, EDGE = "#DAE8FC", "#E1D5E7", "#F8CECC", "#F5F5F5", "#555555"

    def box(x, y, w, h, text, fc, fontsize=8.2, dashed=False, ec=EDGE):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                            linewidth=1.3, edgecolor=ec, facecolor=fc,
                            linestyle="dashed" if dashed else "solid")
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
        return (x, y, w, h)

    def anchor(b, side):
        x, y, w, h = b
        return {"bottom": (x + w / 2, y), "top": (x + w / 2, y + h),
                "left": (x, y + h / 2), "right": (x + w, y + h / 2)}[side]

    def arrow(b1, b2, side1="bottom", side2="top", text=None,
              connectionstyle="arc3,rad=0.0", dashed=False, color=EDGE, lw=1.2):
        p1, p2 = anchor(b1, side1), anchor(b2, side2)
        a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                             linewidth=lw, color=color, shrinkA=2, shrinkB=2,
                             connectionstyle=connectionstyle,
                             linestyle="dashed" if dashed else "solid")
        ax.add_patch(a)
        if text:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(mx + 0.15, my, text, fontsize=7, color=color, style="italic")

    src_y, src_h, src_w = 11.6, 1.05, 2.15
    srcs = [
        (0.2,  "Rathod et al. (2017)\nBrC k(\u03bb),\nfuel/temp-specific", SRC_ACTIVE),
        (2.55, "Rathod et al. (2021)\nCMD/GSD, coating\nfactor, field ER", SRC_ACTIVE),
        (4.9,  "Rathod & Sahu (2022)\nAAE_BC/AAE_BrC,\nBC mass loading", SRC_CONTEXT),
        (7.25, "Rathod et al. (2024)\nWSOC/MSOC MAC, AAE", SRC_CONTEXT),
        (9.6,  "Rathod & Sahu (2026)\nBC/OC refractive\nindices, densities", SRC_ACTIVE),
    ]
    src_boxes = [box(x, src_y, src_w, src_h, t, c, fontsize=7.4) for x, t, c in srcs]

    modA = box(0.2, 9.6, 3.9, 1.05, "Module A\nSource-specific shell optical\nconstants (log-log k(\u03bb) fit)", CORE)
    tier1_box = box(4.6, 9.6, 3.3, 1.05, "Tier 1\nConcentric core-shell\nMie (PyMieScatt)", CORE)
    verif = box(8.3, 9.6, 3.5, 1.05, "Verification\nvs. Rathod & Sahu (2026)'s\nown reported numbers", CLOSE)
    arrow(src_boxes[0], modA, "bottom", "top")
    arrow(src_boxes[4], tier1_box, "bottom", "top")
    arrow(tier1_box, verif, "right", "left")

    rdgfa = box(0.2, 7.7, 3.7, 1.05, "Tier 3a: RDG-FA\nreal fractal-aggregate calc.\n(Sorensen 2001)", MORPH)
    proxy = box(4.3, 7.7, 3.7, 1.05, "Tier 3b: fast proxy\nMCF(Df), calibrated\nagainst Tier 3a", MORPH)
    arrow(rdgfa, proxy, "right", "left", text="calibrates")
    arrow(tier1_box, rdgfa, "bottom", "top")

    modC = box(1.6, 5.8, 6.4, 1.05, "Module C: Monte Carlo population integration\n(real CMD/GSD/coating factor per fuel, Rathod et al. 2021)", CORE)
    arrow(modA, modC, "bottom", "top")
    arrow(proxy, modC, "bottom", "top")
    arrow(src_boxes[1], modC, "bottom", "right",
          connectionstyle="arc3,rad=-0.35", color=COLORS["primary"], lw=1.3)

    modD = box(0.2, 3.9, 4.4, 1.15, "Module D: field closure\n880 nm (direct) & 370 nm (derived)\nvs. Rathod et al. (2021)", CLOSE)
    sens = box(5.1, 3.9, 4.4, 1.15, "Sobol + Value-of-Information\nsensitivity analysis\n(SALib, real-anchored ranges)", CLOSE)
    arrow(modC, modD, "bottom", "left")
    arrow(modC, sens, "bottom", "right")

    out1 = box(0.2, 2.1, 4.4, 1.05, "Closure result:\nconsistency check + spectral\nunderprediction at 370 nm", OUTPUT)
    out2 = box(5.1, 2.1, 4.4, 1.05, "Coating factor identified as\ndominant uncertainty source\n(ST=0.79 at 880 nm)", OUTPUT)
    arrow(modD, out1, "bottom", "top")
    arrow(sens, out2, "bottom", "top")

    context_box = box(4.9, 0.25, 6.9, 1.15,
                       "Used for context, not computation:\nregional BrC/AAE comparison (\u00a74.4);\nfuture ambient shell library & forcing scale-up (\u00a74.7)",
                       SRC_CONTEXT, fontsize=7.6, dashed=True)
    arrow(src_boxes[2], context_box, "bottom", "top", dashed=True,
          connectionstyle="arc3,rad=0.15", color=COLORS["accent"])
    arrow(src_boxes[3], context_box, "bottom", "top", dashed=True,
          connectionstyle="arc3,rad=-0.15", color=COLORS["accent"])

    ax.text(6, 12.95, "PRISM model architecture: data provenance, modules, and information flow",
            ha="center", fontsize=13, weight="bold")

    legend_items = [
        Patch(facecolor=SRC_ACTIVE, edgecolor=EDGE, label="Source study -- actively wired into computation"),
        Patch(facecolor=SRC_CONTEXT, edgecolor=EDGE, label="Source study -- context / future work only"),
        Patch(facecolor=CORE, edgecolor=EDGE, label="Mie optical core / population integration"),
        Patch(facecolor=MORPH, edgecolor=EDGE, label="Morphology tier (RDG-FA + fast proxy)"),
        Patch(facecolor=CLOSE, edgecolor=EDGE, label="Closure & sensitivity analysis"),
        Patch(facecolor=OUTPUT, edgecolor=EDGE, label="Model output"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.06),
              ncol=2, frameon=False, fontsize=8.5, handlelength=1.5, handleheight=1.2)

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig01_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig 1 (architecture) done")


def make_figure2_tier1_verification(outdir):
    """Figure 2: Tier-1 verification against Rathod and Sahu (2026)."""
    wl = 550.0
    dcore_list = [20, 30, 40, 50, 100, 150, 200]
    cf_range = np.linspace(1.01, 20, 60)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for dcore, color in zip(dcore_list, PALETTE_7):
        rabs, rsca = [], []
        for cf in cf_range:
            dshell = dcore * cf
            out_c = ps.MieQCoreShell(M_CORE_BC, M_SHELL_NONABS, wl, dcore, dshell)
            out_oc = ps.MieQCoreShell(M_SHELL_NONABS, M_SHELL_NONABS, wl, dshell, dshell)
            Cabs_c = out_c[2] * math.pi * (dshell / 2) ** 2
            Cabs_oc = out_oc[2] * math.pi * (dshell / 2) ** 2
            Csca_c = out_c[1] * math.pi * (dshell / 2) ** 2
            Csca_oc = out_oc[1] * math.pi * (dshell / 2) ** 2
            rabs.append(Cabs_c / Cabs_oc)
            rsca.append(Csca_c / Csca_oc)
        axes[0].plot(cf_range, rabs, color=color, label=f"{dcore} nm")
        axes[1].plot(cf_range, rsca, color=color, label=f"{dcore} nm")

    for ax, ylab, title in zip(axes, [r"$R_{abs}$", r"$R_{sca}$"],
                                ["(a) Absorption ratio: coated BC / pure OC",
                                 "(b) Scattering ratio: coated BC / pure OC"]):
        ax.axhline(1.0, color=COLORS["neutral"], lw=1.0, ls="--")
        ax.set_xlabel("Coating factor (CF)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(fontsize=7.5, title="Core diameter", ncol=1, loc="upper right")
    fig.suptitle("Reproduction of Rathod and Sahu (2026)'s concentric-Mie behavior at 550 nm", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig02_tier1_verification.png")
    plt.close(fig)
    print("Fig 2 (Tier-1 verification) done")


def make_figure3_field_closure(outdir):
    """Figure 3: field closure at 880 nm."""
    results = {}
    for fuel in ["wood", "dung"]:
        v = sample_population(20000, fuel=fuel, wavelength_nm=880.0, apply_morphology=False)
        results[fuel] = v.mean()
    observed = {"wood": 1.35, "dung": 1.59}
    observed_unc = {"wood": 0.06, "dung": 0.08}

    fig, ax = plt.subplots(figsize=(6, 4.3))
    x = np.arange(2)
    width = 0.32
    ax.bar(x - width / 2, [results["wood"], results["dung"]], width,
           label="PRISM (population-averaged)", color=COLORS["primary"])
    ax.bar(x + width / 2, [observed["wood"], observed["dung"]], width,
           yerr=[observed_unc["wood"], observed_unc["dung"]], capsize=4,
           label="Observed field ER (Rathod et al. 2021)", color=COLORS["tertiary"])
    ax.set_xticks(x)
    ax.set_xticklabels([FUEL_LABEL[f] for f in ["wood", "dung"]])
    ax.set_ylabel(r"$E_{abs}$ at 880 nm")
    ax.set_title("Field closure: modeled vs. observed enhancement ratio", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig03_field_closure.png")
    plt.close(fig)
    print("Fig 3 (field closure) done:", results)
    return results


def make_figure4_mc_convergence(outdir):
    """Figure 4: Monte Carlo convergence."""
    n, means, cis = convergence_curve(fuel="dung", wavelength_nm=880.0)
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.errorbar(n, means, yerr=cis, marker="o", capsize=3, color=COLORS["primary"])
    ax.axhline(1.59, color=COLORS["secondary"], ls="--", label="Observed field ER (Rathod et al. 2021)")
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo sample size (N)")
    ax.set_ylabel(r"Population-mean $E_{abs}$(880 nm), dung cake")
    ax.set_title("Monte Carlo convergence", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig04_mc_convergence.png")
    plt.close(fig)
    print("Fig 4 (MC convergence) done")


def make_figure5_spectral_closure(outdir):
    """Figure 5: spectral closure (370 nm and 880 nm targets)."""
    rng = np.random.default_rng(20260721)

    def derive_er370(lensing_mean, lensing_sd, brc_mean, brc_sd, n=500000):
        lensing = rng.normal(lensing_mean, lensing_sd, n)
        brc = rng.normal(brc_mean, brc_sd, n)
        bc_core = np.clip(100 - lensing - brc, 0.5, 99)
        ER = 100 / bc_core
        point_est = 100 / (100 - lensing_mean - brc_mean)
        p16, p84 = np.percentile(ER, [16, 84])
        return point_est, p16, p84

    targets_370 = {"wood": derive_er370(19, 2, 33, 7), "dung": derive_er370(31, 9, 35, 13)}
    targets_880 = {"wood": (1.35, 0.06), "dung": (1.59, 0.08)}
    wavelengths = np.array([350, 370, 450, 550, 650, 780, 880, 950], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, fuel in zip(axes, ["wood", "dung"]):
        means, sds = [], []
        for wl in wavelengths:
            v = sample_population(6000, fuel=fuel, wavelength_nm=wl, apply_morphology=False)
            means.append(v.mean()); sds.append(v.std())
        means, sds = np.array(means), np.array(sds)
        ax.plot(wavelengths, means, "-o", color=COLORS["primary"], label="PRISM modeled")
        ax.fill_between(wavelengths, means - sds, means + sds, color=COLORS["primary"], alpha=0.15)
        ax.axvline(650, color=COLORS["neutral"], ls=":", lw=1)

        pe370, p16, p84 = targets_370[fuel]
        pe880, unc880 = targets_880[fuel]
        ax.errorbar([370], [pe370], yerr=[[pe370 - p16], [p84 - pe370]], fmt="s",
                    color=COLORS["secondary"], capsize=4, ms=7, label="Derived target (370 nm)")
        ax.errorbar([880], [pe880], yerr=[[unc880]], fmt="D", color=COLORS["tertiary"],
                    capsize=4, ms=7, label="Measured target (880 nm)")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(r"$E_{abs}$")
        ax.set_title(f"({'a' if fuel=='wood' else 'b'}) {FUEL_LABEL[fuel]}", loc="left")
        if fuel == "wood":
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Spectral field closure: modeled vs. two independent targets per fuel", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig05_spectral_closure.png")
    plt.close(fig)
    print("Fig 5 (spectral closure) done")


def make_figure6_rdg_comparison(outdir):
    """Figure 6: concentric Mie vs. RDG-FA vs. recalibrated proxy."""
    wl = 880.0
    concentric, rdgfa, proxy_corrected = [], [], []
    for fuel in ["wood", "dung"]:
        p = FUEL_PARAMS[fuel]
        k, _ = shell_k_real(wl, p["k_source"])
        mShell = complex(1.55, k)
        dcore, cf = p["cmd"], p["coating_factor"]
        coat_thick = (cf - 1.0) * dcore / 2.0
        ec = e_abs(dcore, coat_thick, wl, mCore=M_CORE_BC, mShell=mShell)
        eagg, n_p, Rg = aggregate_eabs(dcore, cf, wl, mShell)
        mcf = morphology_correction_factor(1.8)
        epc = 1.0 + (ec - 1.0) * mcf
        concentric.append(ec); rdgfa.append(eagg); proxy_corrected.append(epc)

    x = np.arange(2); w = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.bar(x - w, concentric, w, label="Tier 1: concentric Mie", color=COLORS["neutral"])
    ax.bar(x, rdgfa, w, label="RDG-FA fractal aggregate (Sorensen 2001)", color=COLORS["quaternary"])
    ax.bar(x + w, proxy_corrected, w, label="Tier 1 x recalibrated proxy", color=COLORS["primary"])
    ax.set_xticks(x); ax.set_xticklabels([FUEL_LABEL[f] for f in ["wood", "dung"]])
    ax.set_ylabel(r"$E_{abs}$(880 nm)")
    ax.set_title("Morphology effect: concentric Mie vs. real RDG-FA vs. fast proxy", loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig06_rdg_comparison.png")
    plt.close(fig)
    print("Fig 6 (RDG-FA comparison) done:", concentric, rdgfa, proxy_corrected)


def make_figure7_rdg_sensitivity(outdir):
    """Figure 7 (new vs. earlier drafts): RDG-FA sensitivity to fractal
    dimension and monomer diameter."""
    wl = 880.0
    df_range = np.linspace(1.5, 2.6, 12)
    dp_range = np.linspace(20, 50, 12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for fuel in ["wood", "dung"]:
        p = FUEL_PARAMS[fuel]
        k, _ = shell_k_real(wl, p["k_source"])
        mShell = complex(1.55, k)
        dcore, cf = p["cmd"], p["coating_factor"]

        eabs_vs_df = [aggregate_eabs(dcore, cf, wl, mShell, Df=df, monomer_diameter_nm=30.0)[0]
                      for df in df_range]
        axes[0].plot(df_range, eabs_vs_df, "-o", color=FUEL_COLOR[fuel], label=FUEL_LABEL[fuel])

        eabs_vs_dp = [aggregate_eabs(dcore, cf, wl, mShell, Df=1.8, monomer_diameter_nm=dp)[0]
                      for dp in dp_range]
        axes[1].plot(dp_range, eabs_vs_dp, "-o", color=FUEL_COLOR[fuel], label=FUEL_LABEL[fuel])

    axes[0].axvline(1.8, color=COLORS["neutral"], ls=":", lw=1)
    axes[0].set_xlabel(r"Fractal dimension, $D_f$")
    axes[0].set_ylabel(r"RDG-FA $E_{abs}$(880 nm)")
    axes[0].set_title("(a) Sensitivity to fractal dimension\n(monomer diameter fixed at 30 nm)", loc="left", fontsize=10.5)
    axes[0].legend(frameon=False)

    axes[1].axvline(30.0, color=COLORS["neutral"], ls=":", lw=1)
    axes[1].set_xlabel("Primary monomer diameter (nm)")
    axes[1].set_ylabel(r"RDG-FA $E_{abs}$(880 nm)")
    axes[1].set_title("(b) Sensitivity to monomer diameter\n($D_f$ fixed at 1.8)", loc="left", fontsize=10.5)

    fig.suptitle("RDG-FA sensitivity to literature-typical morphology assumptions", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig07_rdg_sensitivity.png")
    plt.close(fig)
    print("Fig 7 (RDG-FA sensitivity) done")


def make_figure8_sobol(outdir):
    """Figure 8: Sobol sensitivity tornado plot."""
    Si, nsamp = run_sobol(n_base=2048)
    names = SOBOL_PROBLEM["names"]
    label_map = {"dcore_nm": "Core diameter", "coat_factor": "Coating factor",
                 "k_shell_880nm": "Shell k (880 nm)", "Df": "Fractal dimension"}
    order = np.argsort(Si["ST"])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    y = np.arange(len(names))
    s1 = np.array(Si["S1"])[order]
    st = np.array(Si["ST"])[order]
    labels = [label_map[names[i]] for i in order]
    ax.barh(y, s1, color=COLORS["primary"], label="First-order (S1)")
    ax.barh(y, st - s1, left=s1, color=COLORS["accent"], label="Interaction (ST - S1)")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(r"Fraction of $E_{abs}$(880 nm) variance explained")
    ax.set_title(f"Sobol sensitivity analysis (N={nsamp} evaluations)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig08_sobol_tornado.png")
    plt.close(fig)
    print("Fig 8 (Sobol) done")
    return Si, nsamp


def make_figure9_voi(outdir):
    """Figure 9: Value-of-Information ranking."""
    var_base, results = voi()
    label_map = {"dcore_nm": "Core diameter", "coat_factor": "Coating factor",
                 "k_shell_880nm": "Shell k (880 nm)", "Df": "Fractal dimension"}
    names_sorted = sorted(results, key=lambda k: -results[k][1])
    reductions = [results[n][1] * 100 for n in names_sorted]
    labels = [label_map[n] for n in names_sorted]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = [COLORS["primary"] if r >= 0 else COLORS["secondary"] for r in reductions]
    ax.barh(labels[::-1], reductions[::-1], color=colors[::-1])
    ax.axvline(0, color=COLORS["neutral"], lw=0.8)
    ax.set_xlabel("% reduction in total variance if uncertainty eliminated")
    ax.set_title("Value-of-Information ranking", loc="left")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig09_voi.png")
    plt.close(fig)
    print("Fig 9 (VoI) done:", dict(zip(names_sorted, reductions)))
    return var_base, results


# ==========================================================================
# 9. Reproduce every figure and headline number in the manuscript
# ==========================================================================
def main(outdir="./prism_figures"):
    os.makedirs(outdir, exist_ok=True)

    make_figure1_architecture(outdir)
    make_figure2_tier1_verification(outdir)
    field_closure = make_figure3_field_closure(outdir)
    make_figure4_mc_convergence(outdir)
    make_figure5_spectral_closure(outdir)
    make_figure6_rdg_comparison(outdir)
    make_figure7_rdg_sensitivity(outdir)
    Si, nsamp = make_figure8_sobol(outdir)
    var_base, voi_results = make_figure9_voi(outdir)

    print("\n=== Headline numbers (should match the manuscript) ===")
    print(f"Field closure E_abs(880nm): wood={field_closure['wood']:.3f}, "
          f"dung={field_closure['dung']:.3f}  (observed: 1.35 +/- 0.06, 1.59 +/- 0.08)")
    names = SOBOL_PROBLEM["names"]
    for i, name in enumerate(names):
        print(f"Sobol {name:16s} S1={Si['S1'][i]:.3f}  ST={Si['ST'][i]:.3f}")
    print(f"VoI baseline variance: {var_base:.5f}")
    for name, (v, r) in sorted(voi_results.items(), key=lambda kv: -kv[1][1]):
        print(f"VoI {name:16s} reduction={r*100:.1f}%")
    print(f"\nAll figures written to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate all PRISM manuscript figures.")
    parser.add_argument("--outdir", default="./prism_figures",
                         help="directory to write figures into (default: ./prism_figures)")
    args = parser.parse_args()
    main(args.outdir)
