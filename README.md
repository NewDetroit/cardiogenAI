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
[2] Topological Compression ──────── UMAP (supervised) or PCA → n_qubits dims
   │
   ▼
[3] Quantum Kernel ───────────────── custom α/β feature map (reps≥2, full entangle)
   │                                  projected kernel (Huang et al. 2021) or fidelity
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
  hERG dataset from **[Therapeutics Data Commons
  (TDC)](https://tdcommons.ai/single_pred_tasks/tox/)** — by default
  `herg_central` (the ~300k-compound hERGCentral electrophysiology screen, Du et
  al. 2011; use `tdc_label_name='hERG_inhib'`), with `herg_karim` (Karim et al.
  2021) and `hERG` (Wang et al. 2016) also available. Invalid SMILES are dropped
  with RDKit and a **class-balanced** working subsample is drawn for the (O(N²))
  quantum kernel. You can also point the loader at **any hERG CSV you download**
  (`local_path=...`); the SMILES and label columns are auto-detected.
- **Named-drug sanity panel.** `load_known_drug_panel()` provides 30 curated
  marketed drugs (Terfenadine, Dofetilide, … vs. Aspirin, Metformin, …) with
  RDKit-canonical SMILES for interpretable spot checks.

## What's in Stage 1 (beyond a plain SVC)

The QSVC triage is hardened so it does not collapse into a trivial
"predict-everything-toxic" classifier:

- a **leakage-free validation holdout** carved from train (supervised UMAP and
  the feature map are fit only on the remaining "fit" split) used to select
  `C`, the α/β/γ feature-map scales, and the decision threshold;
- `class_weight='balanced'` plus **Youden's-J threshold calibration** on that
  validation split;
- **same-features classical controls** (RBF/linear SVC on the identical
  compressed features) and a **raw-embedding ceiling** (`rbf_384`), so you see
  whether the quantum kernel actually beats a classical kernel on equal footing;
- **`repeated_evaluation`** over many seeds (mean ± 95% CI), not a single split.

## Quantum kernel: projected by default, honestly benchmarked

The feature map is a custom, per-rep circuit — Hadamards, single-qubit phases
`2·α·zₖ`, and coupling phases `2·β·zₘ·zₙ` on wired pairs — with **α and β
decoupled** (β is *not* welded to α², so the entangling term isn't forced to a
vanishing value). Both scales are tuned on a validation holdout.

Two kernels are available (`kernel_type`):

- **`projected`** (default; Huang et al. 2021) — a Gaussian on the
  reduced-density-matrix descriptors of |φ(x)⟩ (single-qubit Bloch vectors, plus
  two-qubit RDMs at `projected_order=2`). Projected kernels avoid the
  exponential **kernel concentration** that makes plain fidelity kernels
  saturate and underperform.
- **`fidelity`** — the exact |⟨φ(x)|φ(y)⟩|² kernel, computed by simulating each
  circuit once with the statevector simulator (O(N) sims + vectorised overlaps),
  so a few-hundred-molecule kernel is seconds, not minutes.

Every quantum result is reported next to **same-features classical controls**
(RBF/linear SVC on the identical compressed features) and a **raw-embedding
ceiling** (`rbf_384`), plus the Huang **geometric difference** `g(K_C‖K_Q)` and
**kernel-target alignment** so you can see whether the quantum kernel is merely
*different* or actually *better*. `repeated_evaluation(...)` re-runs the whole
pipeline over many seeds and reports mean ± 95% CI — a single split is not
trusted.

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

# Official hERG data via TDC (pip install PyTDC), class-balanced working set
smiles, labels, names = load_herg_dataset(
    source="tdc", tdc_name="herg_central", tdc_label_name="hERG_inhib", n_samples=2000)
# ...or herg_karim / hERG, or a CSV you downloaded (columns auto-detected):
# smiles, labels, names = load_herg_dataset(local_path="herg.csv", n_samples=2000)

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
| `n_qubits` | `8` | compressed dim = qubit count (6–10 recommended) |
| `compressor` | `umap` | `umap` (optionally supervised) or `pca` front-end |
| `umap_supervised` | `True` | label-guide UMAP on train only (helps distance kernels) |
| `kernel_type` | `projected` | `projected` (Huang 2021) or `fidelity` |
| `projected_order` | `2` | 1 = single-qubit RDMs; 2 = + two-qubit RDMs |
| `feature_map_reps` | `2` | expressivity starts at reps≥2 |
| `entanglement` | `full` | `linear`/`circular`/`full`/`mutual_info` wiring |
| `tune_alpha` / `tune_beta` / `tune_gamma` | grids | feature-map + kernel scales (β=0 ⇒ no coupling) |
| `n_samples` | `2500` | balanced hERG working-set size (↑ = better, slower) |
| `tune_C` | `(0.1, 1, 10, 100)` | grid for the QSVC `C` |
| `class_weight` | `balanced` | SVC class weighting |
| `calibrate_threshold` | `True` | pick decision threshold on the validation split |
| `geometric_difference` | `True` | report `g(K_C‖K_Q)` quantum-vs-classical divergence |
| `run_classical_baseline` | `True` | same-features + raw-embedding controls |
| `n_clusters` | `3` | Stage 2 sub-group count |

## Architecture

| Class / function | Responsibility |
|---|---|
| `load_herg_dataset` | Load an official hERG benchmark (TDC) or your CSV; balanced subsample |
| `load_known_drug_panel` | 30 curated named drugs for interpretable checks |
| `MoleculeEmbedder` | SMILES → ChemBERTa embeddings (masked-mean pooling) |
| `TopologicalCompressor` | UMAP fit/transform + scaling to `[0, π]` angles |
| `QuantumProcessor` | α/β feature map + projected (or fidelity) statevector kernel |
| `ToxicityPipeline` | Orchestrates Steps 1–5, returns `PipelineResult` |
| `print_report` | Formatted Stage 1 / baseline / Stage 2 report |

Each molecule's feature-map circuit is simulated once with `Statevector`; the
projected RDM descriptors (or fidelity overlaps) then form the kernel, and the
train/test/sub-group blocks are all slices of it. Runs on the local simulator —
no Qiskit primitive or hardware account required.

## Notes on results

On the real hERG data with real ChemBERTa, the projected quantum kernel is
**competitive with the same-features classical controls** (RBF/linear SVC on the
identical compressed features) — but the **raw-384-d ceiling (`rbf_384`) still
wins**, so the compression front-end, not the kernel, is the main bottleneck.
More data helps most: going from ~500 → ~2000 molecules lifted quantum test
ROC-AUC by ~0.10. Read the numbers from `repeated_evaluation` (mean ± 95% CI),
not a single split, and prefer a scaffold split for honest generalization.
Stage 2's sub-groups are reported with a physicochemical profile (MW, LogP,
TPSA, aromatic rings, H-bond donors/acceptors) so each cluster is chemically
interpretable.

## Official dataset links

| Dataset | Source | Load |
|---|---|---|
| **hERGCentral (Du et al. 2011, ~300k screen)** | [TDC](https://tdcommons.ai/single_pred_tasks/tox/) | `load_herg_dataset(source="tdc", tdc_name="herg_central", tdc_label_name="hERG_inhib")` |
| **hERG blockers (Karim et al. 2021, ~13,445)** | [TDC](https://tdcommons.ai/single_pred_tasks/tox/) | `load_herg_dataset(source="tdc", tdc_name="herg_karim")` |
| **hERG blockers (Wang et al. 2016, ~655)** | [TDC](https://tdcommons.ai/single_pred_tasks/tox/) | `load_herg_dataset(source="tdc", tdc_name="hERG")` |
| Underlying bioactivity source | [ChEMBL target CHEMBL240](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL240/) | — |
| Your own CSV (SMILES + label) | any | `load_herg_dataset(local_path="herg.csv")` |

TDC access needs `pip install PyTDC`. The TDC label convention is `1` = hERG
blocker (cardiotoxic), `0` = non-blocker.
