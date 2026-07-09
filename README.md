# CardiogenAI — Hybrid QML Cardiotoxicity Pipeline

A two-stage hybrid **Classical–Quantum** machine-learning pipeline that
**triages drug molecules for cardiotoxicity** and then **discovers structural
toxicity mechanisms** among the flagged molecules — directly from SMILES
strings.

```
SMILES strings
   │
   ▼
[1] ChemBERTa Semantic Extraction ── DeepChem/ChemBERTa-77M-MTR
   │                                  → dense (N, 384) chemical embeddings
   ▼
[2] Topological Compression ──────── UMAP(n_neighbors=5, min_dist=0.1,
   │                                       n_components=n_qubits)
   ▼
[3] Quantum Kernel ───────────────── ZZFeatureMap(reps=2, entanglement='linear')
   │                                  + FidelityQuantumKernel (local simulator)
   ▼
[4] Stage 1 — Supervised Triage ──── SVC(kernel='precomputed')  ⇒  QSVC
   │                                  0 = Safe, 1 = Toxic (cardiotoxic)
   ▼
[5] Stage 2 — Mechanism Discovery ── SpectralClustering(affinity='precomputed')
                                      over the quantum-kernel sub-matrix of the
                                      predicted-toxic molecules → candidate
                                      toxicity-mechanism clusters.
```

## Real, not mock

- **Real model.** Step 1 loads the pretrained **ChemBERTa** chemical language
  model (`DeepChem/ChemBERTa-77M-MTR`, a RoBERTa-style transformer pretrained
  on ~77M molecules). There is **no random-initialised fallback** — if the
  weights cannot be fetched, the pipeline raises a clear error rather than
  silently degrading, so every embedding reflects genuine pretrained
  chemistry.
- **Real data.** The demo panel in `load_cardiotoxicity_dataset()` is 30
  marketed drugs with **RDKit-canonical SMILES** and **pharmacology-grounded
  labels**:
  - **Toxic (1):** hERG / I_Kr blockers, QT-prolonging / torsadogenic agents,
    and drugs withdrawn or restricted for cardiovascular risk — e.g.
    Terfenadine, Astemizole, Cisapride, Dofetilide, Sotalol, Amiodarone,
    Quinidine, Haloperidol, Thioridazine, Pimozide, Ibutilide, Bepridil.
  - **Safe (0):** drugs with no significant clinical cardiotoxicity signal —
    e.g. Aspirin, Ibuprofen, Acetaminophen, Caffeine, Metformin, Naproxen,
    Fexofenadine (the non-cardiotoxic metabolite of Terfenadine), Amoxicillin,
    Lisinopril.

## Why two stages?

- **Stage 1 (QSVC triage)** separates *Toxic* from *Safe* molecules using a
  fidelity quantum kernel — the SVM operates in the Hilbert space induced by
  the ZZ feature map rather than in Euclidean space.
- **Stage 2 (Spectral Clustering)** reuses the *same quantum similarity
  structure* to group the flagged molecules into candidate
  **toxicity-mechanism** clusters (e.g. class-III antiarrhythmics vs.
  antipsychotic hERG blockers vs. antihistamine blockers) — with no labels.

## Installation

Python 3.10+ required.

```bash
pip install -r requirements.txt
```

## Usage

Run the end-to-end demo on the curated real-drug panel:

```bash
python cardiotoxicity_pipeline.py
```

Programmatic use:

```python
from cardiotoxicity_pipeline import (
    ToxicityPipeline, PipelineConfig, load_cardiotoxicity_dataset,
)

smiles, labels, names = load_cardiotoxicity_dataset()
config = PipelineConfig(n_qubits=8, n_clusters=3)
result = ToxicityPipeline(config).run(smiles, labels, names)

print(result.report)          # Stage 1 triage classification report
print(result.toxic_index)     # dataset ids predicted Toxic
print(result.cluster_labels)  # Stage 2 mechanism-cluster id per toxic molecule
```

## Key hyperparameters (`PipelineConfig`)

| Field | Default | Meaning |
|---|---|---|
| `model_name` | `DeepChem/ChemBERTa-77M-MTR` | HuggingFace chemical LM |
| `pooling` | `mean` | `mean` (masked), `cls`, or `pooler` token reduction |
| `n_qubits` | `8` | UMAP output dim = qubit count (8 or 16) |
| `umap_n_neighbors` | `5` | UMAP local neighborhood size |
| `umap_min_dist` | `0.1` | UMAP minimum embedding distance |
| `feature_map_reps` | `2` | ZZFeatureMap repetitions |
| `entanglement` | `linear` | ZZFeatureMap entanglement pattern |
| `n_clusters` | `3` | Stage 2 mechanism-cluster count |

## Architecture

| Class | Responsibility |
|---|---|
| `MoleculeEmbedder` | SMILES → ChemBERTa embeddings (masked-mean pooling) |
| `TopologicalCompressor` | UMAP fit/transform + scaling to `[0, π]` angles |
| `QuantumProcessor` | ZZFeatureMap + fidelity kernel matrix evaluation |
| `ToxicityPipeline` | Orchestrates Steps 1–5, returns `PipelineResult` |

A single full **N×N** quantum kernel is computed once; the train-to-train and
test-to-train matrices used by the QSVC — and the affinity sub-matrix used for
mechanism clustering — are all sub-blocks of it.

All quantum computation runs locally through Qiskit's Sampler primitive
(`StatevectorSampler` on Qiskit 2.x, `Sampler` on 1.x — a small compatibility
shim picks the right one). Point the `FidelityQuantumKernel` at a
hardware-backed sampler to run on real devices.

## Network requirement

Step 1 downloads the ChemBERTa weights from the Hugging Face Hub on first run
(then reads from the local HF cache). The host must be able to reach
`huggingface.co`, or have the model pre-cached (`HF_HOME`) or mirrored
(`HF_ENDPOINT`). In fully air-gapped environments, pre-download the model
where you have connectivity and copy the HF cache across.
