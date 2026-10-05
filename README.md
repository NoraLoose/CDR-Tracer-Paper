# CDR-Tracer-Paper

Notebooks that make the figures in

> Loose, N., Long, M., Bachman, S., Damien, P., Eilerman, S., Karspeck, A., Molemaker, J.,
> Stephenson, D., Tyka, M., & Wyatt, A. *Linearized Carbonate Chemistry for Efficient Modeling of
> Ocean-Based Carbon Dioxide Removal*. Submitted to Journal of Advances in Modeling Earth Systems
> (JAMES).

The model setups and the processing that turns raw model output into the data these notebooks
read are in two companion repositories:

- [CDR-Tracer-ROMS](https://github.com/NoraLoose/CDR-Tracer-ROMS): the CDR tracer method for
  ROMS, and in `paper/` the as-run setup of the Pacific ROMS/MARBL truth and CDR tracer
  experiments
- [CDR-Tracer-CESM](https://github.com/NoraLoose/CDR-Tracer-CESM): the global CESM/MARBL CDR
  tracer atlas workflow

## Layout

```
roms/            Pacific ROMS/MARBL figures (plot.py, carbonate.py: shared helpers)
cesm/            Global CESM/MARBL figures and the box model (analysis.py, cdr_tracer_curves.py,
                 bad_curves_analysis.py: shared helpers)
sensitivities/   β and η from OceanSODA-ETHZ, used for Figs. 2, S3–S5 and the SODA
                 configurations in Table 2
```

Each notebook saves its figures to `figures/` next to it.

## Running

```bash
conda env create -f environment.yml
conda activate cdr-tracer-paper
```

**ROMS notebooks** read model output through relative paths (`roms/exp0/...`,
`roms_marbl_dic/expCTRL/...`, `INPUT/...`). Link the run directories into `roms/` first:

```bash
cd roms
./link_data.sh /path/to/pacific   # default: the NERSC location used for the paper
```

**CESM notebooks** find their data through the paths in `cesm/analysis.py` and
`cesm/bad_curves_analysis.py`. These point to NERSC, except for the OAE truth experiments, which
are read from the public
[OAE Efficiency Atlas on Source Cooperative](https://source.coop/cworthy/oae-efficiency-atlas).

A few notebooks also read external datasets by absolute NERSC path: OceanSODA-ETHZ
(Gregor & Gruber, 2021), the SMYLE-FOSI CESM output, and the CESM carbonate sensitivities written
by step 02 of CDR-Tracer-CESM.

TODO(Nora): add the Zenodo DOI of the processed data needed to run these notebooks without NERSC
access.

## Figures

### Main text

| Figure | Notebook | Saved as (`figures/…`) |
|---|---|---|
| 1 | schematic, not made in code | |
| 2 (a–d) | `roms/carbonate_sensitivities.ipynb` | `carbonate_sensitivity.png` |
| 2 (e,f) | `roms/beta_eta_mean_std.ipynb` | `zonally_averaged_beta_eta_pacific.png` |
| 3 (a,b) | `roms/alk-dic-diagram.ipynb` | `alk_dic_diagram.png` |
| 3 (c–f) | `cesm/efficiency_box_model.ipynb` | `flux_efficiency_box_model.png` |
| 4 | `roms/concentration_differences.ipynb` | `mean_speed_map_*.png`, `vertical_mixing_profiles*.png` |
| 5 | `roms/deltaDIC_deltaALK_truth.ipynb` | `delta_alk_truth_2x3.png` |
| 6 | `roms/beta_eta_second_order.ipynb` | `beta_eta_oae_*_rel_diff_2x4.png` |
| 7 | `roms/curves.ipynb` | `curves_oae_e_vs_etilde.png` |
| 8 | `roms/deltaDIC_deltaALK.ipynb` | `deltadic_deltaalk_truth_diff_VI_JP_*.png` |
| 9 | `roms/pH.ipynb` | `max_delta_pH_dor.png` |
| 10 | `cesm/validation_curves_all.ipynb` | `polygons*.png`, `summary_polygons_efficiency_relative.png` |
| 11 | `cesm/validation_curves_all.ipynb` | `largest_divergences_relative_oae.png` |
| 12 | `cesm/validation_co2_flux.ipynb` | `flux_summary_140_oae_*.png` |
| 13 | `roms/curves_different_beta_eta.ipynb` | `curves_approx_beta_eta_oae_8_relative_error.png` |
| Table 3 | `roms/compute_savings_analysis.ipynb` | (printed table) |

### Supporting information

| Figure | Notebook |
|---|---|
| S1, S2 | `cesm/efficiency_box_model.ipynb` |
| S3, S4 | `roms/beta_eta_mean_std.ipynb` (`*_mean_std.png`) |
| S5 | `roms/beta_eta_mean_std.ipynb` (`SODA_climatology.png`) |
| S6, S7 | `roms/order_of_operations.ipynb` |
| S8 | `roms/surface_forcing.ipynb` |
| S9–S11 | `roms/flux_amplification_decomposition.ipynb` |
| S12–S16 | `roms/concentration_differences.ipynb` |
| S17–S20 | `roms/deltaDIC_deltaALK_truth.ipynb` (`delta_alk_dic_truth_*1000.png`) |
| S21–S23 | `roms/beta_eta_second_order.ipynb` |
| S24 | `roms/linearity_cdr_tracer.ipynb` |
| S25, S26 | `roms/curves.ipynb` (`curves_combined_*.png`) |
| S27–S31 | `cesm/validation_curves_all.ipynb` |
| S32–S35 | `cesm/dalk_vertical_profiles.ipynb` |
| S36 | `cesm/mld_maps.ipynb` |
| S37 | `cesm/seaice_maps.ipynb` |
| S38–S40 | `cesm/validation_co2_flux.ipynb` |
| S41–S43 | `roms/curves_different_beta_eta.ipynb` |
| S44–S54 | `roms/beta_eta_robustness_exps.ipynb` |

The OceanSODA-ETHZ β and η (Figs. 2, S3–S5, and the SODA-mon and SODA-clim configurations) are
computed in `sensitivities/compute_OceanSODA_carbonate_sensitivity.ipynb`. The CESM/MARBL β and η
(Fig. 2 and the CESM configuration) come from step 02 of
[CDR-Tracer-CESM](https://github.com/NoraLoose/CDR-Tracer-CESM).
