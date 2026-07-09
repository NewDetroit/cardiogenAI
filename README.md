# CardiogenAI — Hybrid QML Cardiotoxicity Pipeline

A two-stage hybrid **Classical–Quantum** machine-learning pipeline that
**triages drug molecules for hERG cardiotoxicity** and then **discovers
structural toxicity sub-groups** among the flagged molecules — directly from
SMILES strings.

```
SMILES strings
   │
   ▼
[1] ChemBERTa Semantic Extraction ── DeepChem/ChemBERTa-77M-MTR
   │                                  → dense (N, 384) chemical embeddings
   ▼
[2] Topological Compression ──────── UMAP(n_neighbors=15, min_dist=0.1,
   │                                       n_components=n_qubits)
   ▼
[3] Quantum Kernel ───────────────── ZZFeatureMap(reps=2, entanglement='linear')
   │                                  fidelity kernel |⟨φ(x)|φ(y)⟩|²
   ▼
[4] Stage 1 — Supervised Triage ──── SVC(kernel='precomputed')  ⇒  QSVC
   │                                  0 = Safe (non-blocker), 1 = Toxic (blocker)
   │                                  class-balanced · CV-tuned C · threshold-calibrated
   │                                  · benchmarked vs. a classical RBF-SVC baseline
   ▼
[5] Stage 2 — Sub-group Discovery ── SpectralClustering(affinity='precomputed')
                                      over the quantum-kernel sub-matrix of the
                                      predicted-toxic molecules → structural
                                      toxicity sub-groups (characterised by MW,
                                      LogP, TPSA, …).
```

## Real model, real data — no mocks

- **Real model.** Step 1 loads the pretrained **ChemBERTa** chemical language
  model (`DeepChem/ChemBERTa-77M-MTR`, a RoBERTa-style transformer pretrained
  on ~77M molecules). There is **no random-initialised fallback** — if the
  weights cannot be fetched, the pipeline raises a clear error rather than
  silently degrading.
- **Real, large, official dataset.** `load_herg_dataset()` loads an official
  hERG blockers benchmark from **[Therapeutics Data Commons
  (TDC)](https://tdcommons.ai/single_pred_tasks/tox/)** — by default
  `herg_karim` (Karim et al. 2021, **~13,445 molecules**); `hERG` (Wang et al.
  2016, ~655 molecules) is also available. Invalid SMILES are dropped with RDKit
  and a **class-balanced** working subsample is drawn for the (O(N²)) quantum
  kernel. A quick fingerprint baseline puts the data's learnable signal at **CV
  ROC-AUC ≈ 0.76**, so it is a genuine benchmark, not a toy. You can also point
  the loader at **any hERG CSV you download** (`local_path=...`); the SMILES and
  label columns are auto-detected.
- **Named-drug sanity panel.** `load_known_drug_panel()` provides 30 curated
  marketed drugs (Terfenadine, Dofetilide, … vs. Aspirin, Metformin, …) with
  RDKit-canonical SMILES for interpretable spot checks.

## What's in Stage 1 (beyond a plain SVC)

The QSVC triage is hardened so it does not collapse into a trivial
"predict-everything-toxic" classifier:

- `class_weight='balanced'` and a **cross-validated `C`** (grid in
  `PipelineConfig.tune_C`);
- **decision-threshold calibration** via Youden's J on out-of-fold training
  scores (proper precomputed-kernel CV that slices the train columns, not just
  the rows);
- **cross-validated ROC-AUC** reporting (mean ± std), not a single split;
- a **classical baseline** (RBF-SVC + logistic regression on the raw 384-d
  ChemBERTa embeddings) so you can see the quantum path's performance *relative
  to a strong classical model* and quantify the cost of UMAP compression.

## Scalable quantum kernel

A fidelity quantum kernel is O(N²). Qiskit's `FidelityQuantumKernel` runs O(N²)
*circuits* (compute–uncompute), which does not scale past a few dozen molecules.
`QuantumProcessor` instead **simulates each feature-map circuit once** with the
statevector simulator (O(N) simulations) and forms all pairwise overlaps by
vectorised inner products — mathematically identical to a noiseless fidelity
kernel, but a ~300×300 kernel takes ~1 s instead of minutes. Set
`use_fidelity_primitive=True` to switch back to Qiskit's primitive (e.g. for a
hardware-backed sampler).

## Run in Google Colab (easiest)

Open **`cardiotoxicity_pipeline_colab.ipynb`** in Colab and choose
**Runtime → Run all**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NewDetroit/cardiogenAI/blob/claude/hybrid-qml-classification-pipeline-ulmbrp/cardiotoxicity_pipeline_colab.ipynb)

Self-contained (writes the module to disk — no clone needed): installs
dependencies, downloads real ChemBERTa + the official TDC hERG dataset, runs all
five stages, and plots the quantum-vs-classical AUC comparison, the kernel
heatmap, and a 2-D molecule map. Colab reaches `huggingface.co` and the TDC
dataset servers by default; a **GPU runtime** is recommended for embedding 800+
molecules.

## Installation (local)

Python 3.10+ required.

```bash
pip install -r requirements.txt
```

Local runs need network access to `huggingface.co` (ChemBERTa) and, for the
default TDC data source, `pip install PyTDC`. Alternatively pass your own hERG
CSV via `load_herg_dataset(local_path=...)` — no network needed for the data.

## Usage

```bash
python cardiotoxicity_pipeline.py
```

Programmatic use:

```python
from cardiotoxicity_pipeline import (
    ToxicityPipeline, PipelineConfig, load_herg_dataset, print_report,
)

# Official hERG benchmark via TDC (pip install PyTDC), class-balanced 800-mol set
smiles, labels, names = load_herg_dataset(source="tdc", tdc_name="herg_karim", n_samples=800)
# ...or a CSV you downloaded from any official source (columns auto-detected):
# smiles, labels, names = load_herg_dataset(local_path="herg.csv", n_samples=800)

config = PipelineConfig(n_qubits=8, n_clusters=3, n_samples=800)
result = ToxicityPipeline(config).run(smiles, labels, names)

print_report(result)              # Stage 1 + baseline table + Stage 2 sub-groups
print(result.roc_auc, result.cv_auc_mean)
print(result.baseline)            # classical RBF-SVC / logreg metrics
print(result.toxic_index)         # molecules predicted Toxic
print(result.cluster_labels)      # sub-group id per predicted-toxic molecule
print(result.cluster_summary)     # per-sub-group physicochemical profile
```

## Key hyperparameters (`PipelineConfig`)

| Field | Default | Meaning |
|---|---|---|
| `model_name` | `DeepChem/ChemBERTa-77M-MTR` | HuggingFace chemical LM |
| `pooling` | `mean` | `mean` (masked), `cls`, or `pooler` token reduction |
| `n_qubits` | `8` | UMAP output dim = qubit count (8 or 16) |
| `n_samples` | `800` | balanced hERG working-set size (↑ = better, slower) |
| `max_quantum_samples` | `1500` | hard cap on molecules in the quantum kernel |
| `tune_C` | `(0.1, 1, 10, 100)` | CV grid for the QSVC `C` (`()` disables) |
| `class_weight` | `balanced` | SVC class weighting |
| `calibrate_threshold` | `True` | pick decision threshold from train CV |
| `cv_folds` | `5` | folds for CV tuning / reporting |
| `run_classical_baseline` | `True` | also fit RBF-SVC + logreg on raw embeddings |
| `use_fidelity_primitive` | `False` | use Qiskit `FidelityQuantumKernel` instead of exact statevector |
| `n_clusters` | `3` | Stage 2 sub-group count |

## Architecture

| Class / function | Responsibility |
|---|---|
| `load_herg_dataset` | Load an official hERG benchmark (TDC) or your CSV; balanced subsample |
| `load_known_drug_panel` | 30 curated named drugs for interpretable checks |
| `MoleculeEmbedder` | SMILES → ChemBERTa embeddings (masked-mean pooling) |
| `TopologicalCompressor` | UMAP fit/transform + scaling to `[0, π]` angles |
| `QuantumProcessor` | ZZFeatureMap + scalable statevector fidelity kernel |
| `ToxicityPipeline` | Orchestrates Steps 1–5, returns `PipelineResult` |
| `print_report` | Formatted Stage 1 / baseline / Stage 2 report |

A single full **N×N** quantum kernel is computed once; the train-to-train and
test-to-train matrices — and the affinity sub-matrix used for sub-group
clustering — are all sub-blocks of it. The Qiskit compatibility shim picks
`zz_feature_map`/`StatevectorSampler` on Qiskit 2.x and `ZZFeatureMap`/`Sampler`
on 1.x automatically.

## Notes on results

With real ChemBERTa embeddings on the real hERG benchmark, expect the classical
baseline and the quantum QSVC to land in a similar ROC-AUC range (the classical
RBF-SVC on the raw 384-d embeddings is the reference ceiling; UMAP → 8-D
compression is what the quantum kernel trades for its Hilbert-space geometry).
Increase `n_samples` and `n_qubits` (with a GPU) to push performance. Stage 2's
sub-groups are reported with a physicochemical profile (MW, LogP, TPSA,
aromatic rings, H-bond donors/acceptors) so each cluster is chemically
interpretable.

## Official dataset links

| Dataset | Source | Load |
|---|---|---|
| **hERG blockers (Karim et al. 2021, ~13,445)** | [TDC](https://tdcommons.ai/single_pred_tasks/tox/) | `load_herg_dataset(source="tdc", tdc_name="herg_karim")` |
| **hERG blockers (Wang et al. 2016, ~655)** | [TDC](https://tdcommons.ai/single_pred_tasks/tox/) | `load_herg_dataset(source="tdc", tdc_name="hERG")` |
| Underlying bioactivity source | [ChEMBL target CHEMBL240](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL240/) | — |
| Your own CSV (SMILES + label) | any | `load_herg_dataset(local_path="herg.csv")` |

TDC access needs `pip install PyTDC`. The TDC label convention is `1` = hERG
blocker (cardiotoxic), `0` = non-blocker.
