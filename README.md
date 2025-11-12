# AML-LSC-Hierarchy

A small, reproducible pipeline to project single-cell AML data into a Primitive vs Progenitor vs Mature hierarchy, score stemness in remission, aggregate patient-level features, and train an interpretable relapse model.

## What is included

- `src/` reusable modules  
  - `hierarchy_classifier.py` train and load the hierarchy model and annotate new data  
  - `lsc_scoring.py` compute LSC-style per-cell scores and define LSC-high cells  
  - `patient_features.py` build remission feature tables per patient  
  - `relapse_lsc_model.py` fit a simple logistic regression with transparent weights
- `notebooks/01_train_hierarchy_classifier.ipynb` example training of the hierarchy classifier on a reference atlas
- `models/` saved hierarchy classifier and gene list produced by the notebook
- `gene_sets/AMLCellType_Genesets.gmt` helper gene sets from AMLHierarchies

## What is not included

Large datasets are excluded from version control to keep the repo lightweight. Download them locally before running the pipeline.

- Lambo et al. pediatric AML longitudinal scRNA-seq  
  GEO series GSE235063  
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235063

- van Galen et al. AML atlas used to train the hierarchy model  
  GEO series GSE116256  
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256

- AMLHierarchies reference resources that define the Primitive GMP Mature framework  
  GitHub  
  https://github.com/andygxzeng/AMLHierarchies

> Note  
> Do not commit these datasets to the repo. Add your local data paths to `.gitignore` as shown below.

## Quick start

```bash
# clone your repo
git clone https://github.com/travh05/aml-lsc-hierarchy.git
cd aml-lsc-hierarchy

# recommended conda env on macOS
conda create -n aml-lsc python=3.11 -y
conda activate aml-lsc

# core deps
pip install scanpy anndata pandas numpy scipy scikit-learn scikit-misc joblib

# optional for plotting and notebooks
pip install jupyter matplotlib
```

Create a `.gitignore` to keep big data out of Git

```gitignore
.DS_Store
__pycache__/
.ipynb_checkpoints/
*.h5ad
*.mtx
*.mtx.gz
*.tsv.gz
*.csv.gz
data/
Lambo_GSE235063/
```

## Prepare data

1. Download GSE235063 processed matrices from GEO  
   They ship as triplets per sample  
   `*_processed_matrix.mtx.gz` `*_processed_genes.tsv.gz` `*_processed_barcodes.tsv.gz` plus `*_processed_metadata.tsv.gz`  
   Place them under `Lambo_GSE235063/GSE235063_RAW/` in this repo directory.

2. Ensure a reference atlas is available to train or use the provided saved classifier  
   - Either run the training notebook on van Galen GSE116256  
   - Or use the saved model under `models/hierarchy_lr.joblib` and `models/hierarchy_genes.npy`

## Typical workflow

### 1. Train or load the hierarchy classifier

Open `notebooks/01_train_hierarchy_classifier.ipynb` to reproduce training on the reference atlas.  
Or load the saved model in Python:

```python
from src.hierarchy_classifier import load_hierarchy_model
clf, ref_genes = load_hierarchy_model(
    model_path="models/hierarchy_lr.joblib",
    genes_path="models/hierarchy_genes.npy"
)
```

### 2. Build a unified AnnData from Lambo and annotate hierarchy

```python
import scanpy as sc
from pathlib import Path
from src.hierarchy_classifier import annotate_hierarchy
from src.lsc_scoring import score_signature

# assemble all *_processed_* triplets under this folder
base = Path("Lambo_GSE235063/GSE235063_RAW")

# use your loader script or the example code from our conversation
# result is an AnnData "adata" with obs columns:
#   sample_raw  patient_id  timepoint  Treatment_Outcome

# map outcome to relapse vs no_relapse
import pandas as pd, numpy as np
def map_outcome(v):
    s = str(v).lower()
    if "relapse" in s: return "relapse"
    if "censored" in s or "no relapse" in s or "event-free" in s: return "no_relapse"
    return np.nan
adata.obs["outcome"] = adata.obs["Treatment_Outcome"].apply(map_outcome)

# annotate hierarchy
adata = annotate_hierarchy(adata, clf, ref_genes, label_key="hierarchy")
```

### 3. Add an LSC-style score and define LSC-high primitive cells

```python
lsc17 = [
    "DNMT3B","ZBTB46","NYNRIN","ARHGAP22","LAPTM4B","MMRN1",
    "DPYSL3","KIAA0125","CDK6","CPXM1","SOCS2","SMIM24",
    "EMP1","NGFRAP1","CD34","AKR1C3","GPR56",
]
adata = score_signature(adata, lsc17, "LSC17_score")

prim = adata[adata.obs["hierarchy"]=="Primitive"]
thr = prim.obs["LSC17_score"].quantile(0.9)
adata.obs["LSC_high"] = (
    (adata.obs["hierarchy"]=="Primitive") &
    (adata.obs["LSC17_score"] >= thr)
)
```

### 4. Build remission feature table and train a simple relapse model

```python
from src.patient_features import features_from_sc_remission
from src.relapse_lsc_model import RelapseLSCModel

feat = features_from_sc_remission(
    adata,
    patient_key="patient_id",
    timepoint_key="timepoint",         # "diagnosis" "remission" "relapse"
    outcome_key="outcome",             # "relapse" "no_relapse"
    hierarchy_key="hierarchy",
    lsc_flag_key="LSC_high",
    remission_label="remission",
    min_cells=50,
)

rlsc = RelapseLSCModel()
rlsc.fit(feat)
print(rlsc.explain())  # shows feature weights
```

## Current findings in brief

The pipeline assembled twenty four remission samples from GSE235063 with clinical outcomes. A logistic model trained on remission composition shows negative weights for primitive fraction and hierarchy entropy and a small positive weight for primitive LSC density. Group means match these directions. Results are exploratory given the small class balance of twenty relapse versus four no relapse and the use of a generic LSC17 signature. The code path and data wiring are correct and ready to be re-run on larger or internal cohorts and with refined relapse-focused signatures.

## Acknowledgments

- Lambo et al  
  GSE235063 pediatric AML longitudinal single-cell RNA-seq  
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235063

- van Galen et al  
  GSE116256 AML hierarchy atlas used for classifier training  
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256

- AMLHierarchies framework and resources  
  https://github.com/andygxzeng/AMLHierarchies
