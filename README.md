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
[3] Quantum Kernel ───────────────── feature map: H, RZ(2α·z), RZZ(2β·zₘzₙ), reps≥2
   │                                  projected (2-RDM Gaussian) kernel, or fidelity
   ▼
[4] Stage 1 — Supervised Triage ──── SVC(kernel='precomputed')  ⇒  QSVC
   │                                  0 = Safe (non-blocker), 1 = Toxic (blocker)
   │                                  class-balanced · (α, β, γ, C) + threshold
   │                                  chosen on a leakage-free validation holdout
   │                                  · geometric-difference screen · repeated-split CI
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

- `class_weight='balanced'`, with the phase scales **(α, β)**, the projected-kernel width
  **γ**, and **`C`** grid-searched and the **decision threshold** (Youden's J) all chosen on
  a **validation holdout** embedded without labels — leakage-free;
- honest **validation AUC** and **test AUC** (a k-fold CV *inside* a supervised
  embedding would leak and read ≈ 1.0);
- **layered classical baselines**: RBF/linear SVC on the *same* UMAP features
  (`*_umap`, the apples-to-apples control) and on the raw 384-d ChemBERTa
  embeddings (`*_384`, the ceiling you forgo by compressing) — all trained on *fit*,
  reported on *test*, so the comparison is fair.

## Designing a quantum kernel that isn't secretly classical

A naive fidelity kernel on this problem is boxed in three ways, and the fix is a
**design** change, not tuning:

1. **The r=1 degeneracy.** At one repetition with small phases, `K_F ≈ 1 − c²‖x−y‖²`
   — the "quantum" SVM is literally a classical squared-distance machine (and the
   projected kernel reduces to an RBF on angles). So the default uses **`reps=2`**: a
   second Hadamard+phase block breaks the closed form and produces non-trivial
   amplitudes and nonzero `Z` marginals — where expressivity actually starts.
2. **The bandwidth trap.** For an `r=1` fidelity kernel, larger phases → concentration
   (`K→I`), smaller → collapse to all-ones; there is no good middle. So the map uses
   **decoupled phase scales** `2α·z_k` (single-qubit) and `2β·z_m·z_n` (coupling),
   tuned **independently** (not `β=α²`, which welds the coupling to a tiny number),
   with **no `(π−z)` offsets**.
3. **A front-end that favors the competition.** Supervised UMAP pulls same-label points
   together in *Euclidean* distance — exactly what the classical RBF/linear controls
   consume. Set `compressor="pca"` or `umap_supervised=False` for a front-end that
   doesn't pre-optimize the space for distance kernels.

**Projected kernel with 2-RDMs (default).** `kernel_type="projected"` builds a Gaussian
kernel on **reduced-density-matrix descriptors** of `|φ(z)⟩` — single-qubit Bloch vectors
and (`projected_order=2`) the **two-qubit RDMs on the entangling pairs** (Huang et al.
2021). It's structurally immune to global concentration (so it runs at O(1) phases),
halves depth (no `U†`), and is hardware-friendlier. The RDM features are exact
(verified against Qiskit `partial_trace`).

**Wiring.** `entanglement="full"` (all pairs, trivial in simulation) or `"mutual_info"`
(edges between the most correlated compressed coordinates) instead of an arbitrary linear
chain.

**Honest evaluation, built in.**
- **Leakage-free selection:** a **fit/val/test** split — supervised compression fits on
  *fit*; `(α, β, γ, C)` and the threshold are chosen on a *val* set embedded without
  labels; *test* is reported once. (A k-fold CV *inside* a supervised embedding leaks and
  reads ≈ 1.0.)
- **Geometric difference `g(K_C‖K_Q)`** (Huang et al.) screens whether *any* labeling of
  this data could favor the quantum kernel over a classical RBF — logged every run.
- **`repeated_evaluation(...)`** reports test AUC as **mean ± 95% CI** across seeds
  (a single ~200-point split has SE ≈ 0.03, so one number proves nothing).
- **Same-feature classical controls** (`rbf_umap`, `linsvc_umap`) so any quantum result
  is measured against a classical kernel on identical inputs.

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
| `n_qubits` | `8` | compressed dim = qubit count (keep **6–10**) |
| `compressor` | `umap` | `umap` or `pca` (PCA = label-agnostic front-end) |
| `umap_supervised` | `True` | shape UMAP with labels (helps distance-kernel controls) |
| `feature_map_reps` | `2` | repetitions (≥2 needed to break the r=1 degeneracy) |
| `entanglement` | `full` | `full`, `linear`, `circular`, or `mutual_info` |
| `kernel_type` | `projected` | `projected` (RDM Gaussian) or `fidelity` |
| `projected_order` | `2` | `1` = single-qubit RDMs; `2` = + two-qubit RDMs |
| `tune_alpha` | `(0.5, 1, 2)` | single-qubit phase-scale grid |
| `tune_beta` | `(0, .5, 1, 2)` | coupling phase-scale grid (`0` = no entanglement) |
| `tune_gamma` | `(0.5, 1, 2)` | projected-kernel width (× median heuristic) |
| `tune_C` | `(0.1, 1, 10, 100)` | grid for the QSVC `C` |
| `geometric_difference` | `True` | log `g(K_C‖K_Q)` each run |
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
