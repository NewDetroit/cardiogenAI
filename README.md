# Hybrid QML Classification & Sub-Cluster Discovery Pipeline

A two-stage hybrid **Classical–Quantum** machine-learning pipeline for complex
pattern classification and hidden sub-category discovery in unstructured text
(system logs, documents, semantic queries).

```
Raw Text
   │
   ▼
[1] LLM Semantic Extraction ── bert-base-uncased pooler output (N, 768)
   │
   ▼
[2] Topological Compression ── UMAP(n_neighbors=5, min_dist=0.1,
   │                                n_components=n_qubits)
   ▼
[3] Quantum Kernel ─────────── ZZFeatureMap(reps=2, entanglement='linear')
   │                           + FidelityQuantumKernel (local Sampler)
   ▼
[4] Stage 1 (Supervised) ───── SVC(kernel='precomputed')  ⇒ QSVC
   │                           0 = Background/Normal, 1 = Target/Anomaly
   ▼
[5] Stage 2 (Unsupervised) ─── SpectralClustering(affinity='precomputed')
                               over the quantum kernel sub-matrix of the
                               points predicted as Target/Anomaly
```

## Why two stages?

- **Stage 1 (QSVC)** separates the *Target/Anomaly* class from background
  noise using a fidelity quantum kernel — the SVM operates in the Hilbert
  space induced by the ZZ feature map rather than in Euclidean space.
- **Stage 2 (Spectral Clustering)** reuses the *same quantum similarity
  structure* to discover hidden sub-categories **within** the predicted
  anomaly class (e.g. auth attacks vs. memory faults vs. network failures),
  without any labels.

## Installation

Python 3.10+ required.

```bash
pip install -r requirements.txt
```

## Usage

Run the end-to-end demo (generates a synthetic log dataset with one normal
class and three hidden anomaly sub-categories, then executes all five steps):

```bash
python hybrid_qml_pipeline.py
```

Programmatic use:

```python
from hybrid_qml_pipeline import HybridPipeline, PipelineConfig

config = PipelineConfig(n_qubits=8, n_clusters=3)
pipeline = HybridPipeline(config)
result = pipeline.run(train_texts, y_train, test_texts, y_test)

print(result.report)            # Stage 1 classification report
print(result.anomaly_indices)   # test indices predicted as Target/Anomaly
print(result.cluster_labels)    # Stage 2 sub-cluster id per anomaly
```

## Key hyperparameters (`PipelineConfig`)

| Field | Default | Meaning |
|---|---|---|
| `model_name` | `bert-base-uncased` | HuggingFace embedding model |
| `n_qubits` | `8` | UMAP output dim = qubit count (8 or 16 recommended) |
| `umap_n_neighbors` | `5` | UMAP local neighborhood size |
| `umap_min_dist` | `0.1` | UMAP minimum embedding distance |
| `feature_map_reps` | `2` | ZZFeatureMap repetitions |
| `entanglement` | `linear` | ZZFeatureMap entanglement pattern |
| `n_clusters` | `3` | Stage 2 sub-cluster count |

## Architecture

| Class | Responsibility |
|---|---|
| `DataEmbedder` | Text → transformer pooler embeddings |
| `TopologicalCompressor` | UMAP fit/transform + scaling to `[0, π]` angles |
| `QuantumProcessor` | ZZFeatureMap + fidelity kernel matrix evaluation |
| `HybridPipeline` | Orchestrates Steps 1–5, returns `PipelineResult` |

All quantum computation runs locally through Qiskit's Sampler primitive
(`StatevectorSampler` on Qiskit 2.x, `Sampler` on 1.x — a small
compatibility layer picks the right one) — no QPU account needed. Point the
`FidelityQuantumKernel` at a hardware-backed sampler to run on real devices.

**Offline note:** if `bert-base-uncased` cannot be fetched from the
HuggingFace hub (air-gapped environments), `DataEmbedder` automatically
falls back to a corpus-trained WordPiece tokenizer paired with a
randomly-initialised compact `BertModel`, so the demo still runs
end-to-end. With hub access, the real pretrained pooler embeddings are
used.
