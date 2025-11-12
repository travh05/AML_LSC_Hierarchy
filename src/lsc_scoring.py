import scanpy as sc

def score_signature(adata, genes, name):
    """Add a per-cell signature score."""
    genes = [g for g in genes if g in adata.var_names]
    if len(genes) == 0:
        raise ValueError(f"No genes from signature {name} found in adata.")
    sc.tl.score_genes(adata, gene_list=genes, score_name=name)
    return adata
