import scanpy as sc
import numpy as np
import joblib


def map_zeng_celltype_to_hierarchy(ct: str) -> str:

    if not isinstance(ct, str):
        return "Other"
    if ct.startswith("LSPC-") or "HSC" in ct:
        return "Primitive"
    if "GMP" in ct or "Prog" in ct or "ProMono" in ct:
        return "Progenitor"
    if "Mono" in ct or "cDC" in ct or "pDC" in ct:
        return "Mature"
    return "Other"


def load_hierarchy_model(
    model_path: str = "models/hierarchy_lr.joblib",
    genes_path: str = "models/hierarchy_genes.npy",
):

    clf = joblib.load(model_path)
    genes = np.load(genes_path, allow_pickle=True).tolist()
    return clf, genes


def annotate_hierarchy(adata, clf, ref_genes, label_key: str = "hierarchy"):
    adata = adata.copy()

    cell_sums = np.array(adata.X.sum(axis=1)).flatten()
    good_cells = (cell_sums > 0) & np.isfinite(cell_sums)
    adata = adata[good_cells, :].copy()

    if adata.n_obs == 0:
        raise ValueError("No valid cells left after filtering (all zero / NaN).")

    gene_sums = np.array(adata.X.sum(axis=0)).flatten()
    good_genes = (gene_sums > 0) & np.isfinite(gene_sums)
    adata = adata[:, good_genes].copy()

    ref_genes = list(ref_genes)
    n_cells = adata.n_obs
    n_train_genes = len(ref_genes)

    var_index = {g: i for i, g in enumerate(adata.var_names)}

    X = np.zeros((n_cells, n_train_genes), dtype=float)

    present_genes = []
    for j, g in enumerate(ref_genes):
        if g in var_index:
            col = var_index[g]
            if hasattr(adata.X, "toarray"):
                X[:, j] = adata.X[:, col].toarray().ravel()
            else:
                X[:, j] = np.asarray(adata.X[:, col]).ravel()
            present_genes.append(g)

    if len(present_genes) < 300:
        raise ValueError(
            f"Only {len(present_genes)} of {n_train_genes} training genes "
            f"found in this dataset. Check gene symbols / preprocessing."
        )

    preds = clf.predict(X)
    proba = clf.predict_proba(X)

    adata.obs[label_key] = preds
    for i, cls in enumerate(clf.classes_):
        adata.obs[f"{label_key}_{cls}_prob"] = proba[:, i]

    return adata
