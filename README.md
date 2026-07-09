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
[2] Topological Compression ──────── supervised UMAP → n_qubits dims
   │                                  (fit on train labels; test mapped label-free)
   ▼
[3] Quantum Kernel ───────────────── bandwidth map: H, RZ(2c·z), RZZ(2c²·zₘzₙ)
   │                                  fidelity |⟨φ(x)|φ(y)⟩|²  OR projected kernel
   ▼
[4] Stage 1 — Supervised Triage ──── SVC(kernel='precomputed')  ⇒  QSVC
   │                                  0 = Safe (non-blocker), 1 = Toxic (blocker)
   │                                  class-balanced · bandwidth c + C + threshold
   │                                  chosen on a leakage-free validation holdout
   │                                  · benchmarked vs. classical kernels
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
  **hERGCentral** (`herg_central`, label `hERG_inhib`: a ~306,893-compound
  electrophysiology screen, binary blockade). `herg_karim` (~13,445, if your
  PyTDC exposes it) and `hERG` (Wang, ~655) are also available. A **class-balanced**
  working subsample is drawn for the (O(N²)) quantum kernel and validated with
  RDKit. You can also point the loader at **any hERG CSV you download**
  (`local_path=...`); the SMILES and label columns are auto-detected.
- **Named-drug sanity panel.** `load_known_drug_panel()` provides 30 curated
  marketed drugs (Terfenadine, Dofetilide, … vs. Aspirin, Metformin, …) with
  RDKit-canonical SMILES for interpretable spot checks.

## What's in Stage 1 (beyond a plain SVC)

The QSVC triage is hardened so it neither collapses to a trivial one-class
predictor nor reports leaked numbers:

- `class_weight='balanced'`, with the **encoding bandwidth `c` and `C`** grid-searched
  and the **decision threshold** (Youden's J) all chosen on a **validation holdout**
  embedded without labels — leakage-free;
- honest **validation AUC** and **test AUC** (a k-fold CV *inside* a supervised
  embedding would leak and read ≈ 1.0);
- **layered classical baselines**: RBF/linear SVC on the *same* UMAP features
  (`*_umap`, the apples-to-apples control) and on the raw 384-d ChemBERTa
  embeddings (`*_384`, the ceiling you forgo by compressing) — all trained on *fit*,
  reported on *test*, so the comparison is fair.

## Beating kernel concentration (the hard part)

Fidelity quantum kernels **concentrate exponentially**: as the effective Hilbert
space grows, every off-diagonal `|⟨φ(x)|φ(y)⟩|²` shrinks toward 0 and the kernel
degenerates to the identity — the QSVC then predicts a single class regardless of
data quality. This pipeline fixes it on two fronts:

**1. Constrain the circuit.**
- **Bandwidth `c`** — the feature map encodes `RZ(2c·z_k)` and `RZZ(2c²·z_m·z_n)`,
  and `c` is grid-searched (`tune_bandwidth`). Small `c` keeps phases from wandering
  across `[0,2π)`, which is what causes the destructive interference. Measured mean
  off-diagonal fidelity on 8 qubits: `c=0.05 → 0.997`, `c=0.7 → 0.43`, `c=2.0 → 0.005`
  — so `c` tunes the kernel from all-ones to identity, and selection lands in the
  usable middle.
- **No `(π−z)` offsets** — the Qiskit `ZZFeatureMap` default inflates phase variance;
  we use plain `z_m·z_n`.
- **`reps=1`, linear/circular entanglement, few qubits (6–10)** — each repetition and
  every extra qubit compounds the spread.

**2. Give the embedding labels + measure health.**
- **Supervised UMAP** (`umap_supervised`, `umap_target_weight`) shapes the compression
  with class structure (fit on train, test mapped label-free).
- Every run **logs the off-diagonal distribution and kernel-target alignment (KTA)** and
  warns on a near-identity (concentrated) or near-ones (too-weak) kernel.

**3. Projected quantum kernel** (`kernel_type="projected"`, Huang et al. 2021):
instead of state overlap, build a Gaussian kernel on single-qubit **Bloch vectors**
of `|φ(z)⟩`. This sidesteps concentration structurally (not by tuning), halves circuit
depth (no `U†` inversion), and is far more shot/noise-tolerant — the architecture to
prefer for real hardware.

**Leakage-free selection.** Because supervised UMAP uses labels, model selection runs on
a **fit/val/test** split: UMAP fits on *fit*, and bandwidth/`C`/threshold are chosen on a
*val* set embedded without labels. The reported **validation AUC** and **test AUC** are
honest (a k-fold CV inside a supervised embedding would leak and read ~1.0).

## Scalable quantum kernel

A fidelity quantum kernel is O(N²). Qiskit's `FidelityQuantumKernel` runs O(N²)
*circuits* (compute–uncompute), which does not scale past a few dozen molecules.
`QuantumProcessor` instead **simulates each feature-map circuit once** with the
statevector simulator (O(N) simulations) and forms all pairwise overlaps by
vectorised inner products — mathematically identical to a noiseless fidelity
kernel, but a ~300×300 kernel takes ~1 s instead of minutes. The projected
kernel reuses the same statevectors.

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

# Official hERGCentral (hERG_inhib) via TDC (pip install PyTDC), class-balanced set
smiles, labels, names = load_herg_dataset(
    source="tdc", tdc_name="herg_central", tdc_label_name="hERG_inhib", n_samples=408)
# ...or a CSV you downloaded from any official source (columns auto-detected):
# smiles, labels, names = load_herg_dataset(local_path="herg.csv", n_samples=408)

config = PipelineConfig(n_qubits=8, n_clusters=3, n_samples=408)
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
| `n_qubits` | `8` | UMAP output dim = qubit count (keep **6–10**) |
| `umap_supervised` | `True` | shape the embedding with class labels (fit set only) |
| `umap_target_weight` | `0.5` | supervised-UMAP label weight (0=unsup, 1=full) |
| `feature_map_reps` | `1` | feature-map repetitions (each one compounds phase spread) |
| `entanglement` | `linear` | `linear`, `circular`, or `full` |
| `kernel_type` | `fidelity` | `fidelity` or `projected` (Bloch-vector Gaussian) |
| `projected_gamma` | `1.0` | Gaussian width for the projected kernel |
| `encoding_scale` | `0.4` | bandwidth `c` fallback if `tune_bandwidth` is empty |
| `tune_bandwidth` | `(.05,.1,.2,.4,.7,1)` | bandwidth grid (searched by validation AUC) |
| `tune_C` | `(0.1, 1, 10, 100)` | grid for the QSVC `C` |
| `class_weight` | `balanced` | SVC class weighting |
| `calibrate_threshold` | `True` | pick decision threshold on the validation holdout |
| `val_size` | `0.2` | fraction of train held out for leakage-free selection |
| `n_samples` | `408` | balanced working-set size |
| `max_quantum_samples` | `1000` | hard cap on molecules in the quantum kernel |
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
| **hERGCentral (~306,893; default)** | [TDC](https://tdcommons.ai/single_pred_tasks/tox/) | `load_herg_dataset(source="tdc", tdc_name="herg_central", tdc_label_name="hERG_inhib")` |
| hERG blockers (Karim et al. 2021, ~13,445) | [TDC](https://tdcommons.ai/single_pred_tasks/tox/) | `load_herg_dataset(source="tdc", tdc_name="herg_karim")` |
| hERG blockers (Wang et al. 2016, ~655) | [TDC](https://tdcommons.ai/single_pred_tasks/tox/) | `load_herg_dataset(source="tdc", tdc_name="hERG")` |
| Underlying bioactivity source | [ChEMBL target CHEMBL240](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL240/) | — |
| Your own CSV (SMILES + label) | any | `load_herg_dataset(local_path="herg.csv")` |

TDC access needs `pip install PyTDC`. The TDC label convention is `1` = hERG
blocker (cardiotoxic), `0` = non-blocker.
