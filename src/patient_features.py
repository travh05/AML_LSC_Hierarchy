import numpy as np
import pandas as pd


def features_from_sc_remission(
    adata,
    patient_key="patient_id",
    timepoint_key="timepoint",
    outcome_key="outcome",
    hierarchy_key="hierarchy",
    lsc_flag_key="LSC_high",
    remission_label="remission",
    min_cells=50,
):

    rows = []
    rem = adata[adata.obs[timepoint_key] == remission_label].copy()

    for pid, sub in rem.obs.groupby(patient_key):
        if outcome_key not in sub or sub[outcome_key].nunique() != 1:
            continue

        outcome = sub[outcome_key].iloc[0]
        if outcome not in ["relapse", "no_relapse"]:
            continue

        total = sub.shape[0]
        if total < min_cells:
            continue

        frac_prim = (sub[hierarchy_key] == "Primitive").mean()
        frac_prog = (sub[hierarchy_key] == "Progenitor").mean()
        frac_mat  = (sub[hierarchy_key] == "Mature").mean()
        frac_lsc  = sub[lsc_flag_key].mean() if lsc_flag_key in sub else 0.0

        prim_mask = sub[hierarchy_key] == "Primitive"
        if prim_mask.any():
            prim_lsc_density = sub.loc[prim_mask, lsc_flag_key].mean() if lsc_flag_key in sub else 0.0
        else:
            prim_lsc_density = 0.0

        freq = sub[hierarchy_key].value_counts(normalize=True)
        entropy = -np.sum(freq * np.log2(freq + 1e-9))

        rows.append(dict(
            patient_id=pid,
            outcome=outcome,
            n_cells=total,
            frac_Primitive=frac_prim,
            frac_Progenitor=frac_prog,
            frac_Mature=frac_mat,
            frac_LSC_high=frac_lsc,
            prim_LSC_density=prim_lsc_density,
            entropy_hierarchy=entropy,
        ))

    return pd.DataFrame(rows)


def features_from_bulk_deconv(
    df,
    mixture_col="Mixture",
    primitive_cols=("LSPC-Quiescent", "LSPC-Primed", "LSPC-Cycle"),
    timepoint_map=None,
):

    if timepoint_map is None:
        timepoint_map = {
            "primary": "dx",
            "Dx": "dx",
            "dx": "dx",
            "post_chemo": "post_chemo",
            "post_allo": "post_allo",
            "Rel": "rel",
            "rel": "rel",
            "Relapse": "rel",
        }

    parts = df[mixture_col].astype(str).str.split(".", n=1, expand=True)
    df = df.copy()
    df["patient_id"] = parts[0]
    df["tp_raw"] = parts[1]

    df["timepoint"] = df["tp_raw"].map(timepoint_map).fillna(df["tp_raw"])

    df["frac_Primitive"] = df[list(primitive_cols)].sum(axis=1)

    wide = df.pivot_table(
        index="patient_id",
        columns="timepoint",
        values="frac_Primitive",
        aggfunc="mean",
    )

    features = wide.copy()
    if "dx" in wide.columns and "post_chemo" in wide.columns:
        features["delta_prim_post_chemo"] = wide["post_chemo"] - wide["dx"]
    if "dx" in wide.columns and "post_allo" in wide.columns:
        features["delta_prim_post_allo"] = wide["post_allo"] - wide["dx"]
    if "dx" in wide.columns and "rel" in wide.columns:
        features["delta_prim_rel"] = wide["rel"] - wide["dx"]

    features = features.reset_index().rename(columns={"patient_id": "patient_id"})
    return features
