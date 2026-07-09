"""
Hybrid Classical-Quantum Pipeline for Pattern Classification & Sub-Cluster Discovery
=====================================================================================

A two-stage hybrid architecture:

    Raw Text
       |
       v
    [Step 1] LLM Semantic Extraction   (HuggingFace transformer -> pooler embeddings)
       |
       v
    [Step 2] Topological Compression   (UMAP -> n_qubits dimensions)
       |
       v
    [Step 3] Quantum Kernel            (ZZFeatureMap + FidelityQuantumKernel)
       |
       v
    [Step 4] Stage 1: Supervised       (SVC with precomputed quantum kernel = QSVC)
       |          Binary classification: 0 = Background/Normal, 1 = Target/Anomaly
       v
    [Step 5] Stage 2: Unsupervised     (SpectralClustering on the quantum kernel
                                        sub-matrix of predicted anomalies)
                 Hidden sub-category discovery within the Target class.

Requirements
------------
    Python 3.10+
    pip install qiskit qiskit-machine-learning qiskit-aer scikit-learn \
                umap-learn transformers torch

All quantum computation runs on a local statevector simulator via Qiskit
primitives (no real QPU access required).

Author: CardiogenAI
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import umap
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hybrid_qml")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Central hyperparameter registry for the full hybrid pipeline."""

    # --- Step 1: LLM embedding ---
    model_name: str = "bert-base-uncased"
    max_token_length: int = 64
    embedding_batch_size: int = 16

    # --- Step 2: UMAP compression ---
    n_qubits: int = 8               # UMAP output dim == number of qubits
    umap_n_neighbors: int = 5
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    # --- Step 3: Quantum feature map / kernel ---
    feature_map_reps: int = 2
    entanglement: str = "linear"

    # --- Step 4: QSVC ---
    svc_C: float = 1.0

    # --- Step 5: Spectral clustering ---
    n_clusters: int = 3

    # --- Reproducibility ---
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Step 1: Semantic Extraction (LLM)
# --------------------------------------------------------------------------- #
class DataEmbedder:
    """Wraps a HuggingFace transformer to turn raw text into dense embeddings.

    Tokenizes input strings and extracts the final-layer pooler output
    (shape: (N_samples, Hidden_Dim), e.g. 768 for bert-base-uncased).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading transformer '%s' on device '%s' ...",
            config.model_name,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of raw strings into a dense (N, Hidden_Dim) array."""
        if not texts:
            raise ValueError("DataEmbedder.embed received an empty text list.")

        batches: list[np.ndarray] = []
        bs = self.config.embedding_batch_size
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.config.max_token_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            if getattr(outputs, "pooler_output", None) is not None:
                pooled = outputs.pooler_output          # (B, Hidden_Dim)
            else:
                # Models without a pooler head: mean-pool the last hidden state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                pooled = summed / mask.sum(dim=1).clamp(min=1e-9)
            batches.append(pooled.cpu().numpy())

        embeddings = np.vstack(batches).astype(np.float64)
        logger.info("Embedded %d texts -> shape %s", len(texts), embeddings.shape)
        return embeddings


# --------------------------------------------------------------------------- #
# Step 2: Topological Compression (UMAP)
# --------------------------------------------------------------------------- #
class TopologicalCompressor:
    """Compresses LLM embeddings down to `n_qubits` dimensions with UMAP,
    then rescales each feature to [0, pi] so it is a valid rotation angle
    for the quantum feature map.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.reducer = umap.UMAP(
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            n_components=config.n_qubits,
            metric=config.umap_metric,
            random_state=config.random_state,
        )
        self.scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        self._fitted = False

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit UMAP + scaler on training embeddings and return compressed data."""
        logger.info(
            "Fitting UMAP: %s -> %d dims (n_neighbors=%d, min_dist=%.2f)",
            X_train.shape,
            self.config.n_qubits,
            self.config.umap_n_neighbors,
            self.config.umap_min_dist,
        )
        reduced = self.reducer.fit_transform(X_train)
        scaled = self.scaler.fit_transform(reduced)
        self._fitted = True
        return scaled.astype(np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new (test) embeddings into the fitted UMAP space."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first.")
        reduced = self.reducer.transform(X)
        scaled = self.scaler.transform(reduced)
        # Guard: test points can land slightly outside the train range
        return np.clip(scaled, 0.0, np.pi).astype(np.float64)


# --------------------------------------------------------------------------- #
# Step 3: Quantum Feature Map & Kernel
# --------------------------------------------------------------------------- #
class QuantumProcessor:
    """Builds the ZZFeatureMap and computes fidelity quantum kernel matrices
    on a local simulator through Qiskit's Sampler primitive.

    K(x, y) = |<phi(x)|phi(y)>|^2   with |phi(.)> prepared by the ZZFeatureMap.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_map = ZZFeatureMap(
            feature_dimension=config.n_qubits,
            reps=config.feature_map_reps,
            entanglement=config.entanglement,
        )
        sampler = Sampler()  # local reference simulator primitive
        fidelity = ComputeUncompute(sampler=sampler)
        self.kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=fidelity,
        )
        logger.info(
            "Quantum kernel ready: ZZFeatureMap(qubits=%d, reps=%d, "
            "entanglement='%s'), circuit depth=%d",
            config.n_qubits,
            config.feature_map_reps,
            config.entanglement,
            self.feature_map.decompose().depth(),
        )

    def train_kernel(self, X_train: np.ndarray) -> np.ndarray:
        """Symmetric train-vs-train kernel matrix, shape (N_train, N_train)."""
        logger.info(
            "Computing train kernel matrix (%d x %d) ...",
            len(X_train),
            len(X_train),
        )
        return self.kernel.evaluate(x_vec=X_train)

    def cross_kernel(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """Test-vs-train kernel matrix, shape (N_test, N_train)."""
        logger.info(
            "Computing test-vs-train kernel matrix (%d x %d) ...",
            len(X_test),
            len(X_train),
        )
        return self.kernel.evaluate(x_vec=X_test, y_vec=X_train)


# --------------------------------------------------------------------------- #
# Steps 4 & 5: Hybrid two-stage pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    """Container for everything the pipeline produces."""

    y_pred: np.ndarray
    accuracy: float
    f1: float
    roc_auc: float
    report: str
    anomaly_indices: np.ndarray            # test-set indices predicted as class 1
    cluster_labels: np.ndarray | None      # sub-cluster id per anomaly, or None
    train_kernel: np.ndarray = field(repr=False, default=None)
    test_kernel: np.ndarray = field(repr=False, default=None)


class HybridPipeline:
    """End-to-end orchestrator:

    text -> LLM embedding -> UMAP -> quantum kernel
         -> Stage 1: QSVC (supervised, binary)
         -> Stage 2: SpectralClustering over predicted anomalies (unsupervised)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.embedder = DataEmbedder(self.config)
        self.compressor = TopologicalCompressor(self.config)
        self.quantum = QuantumProcessor(self.config)
        self.classifier = SVC(
            kernel="precomputed",
            C=self.config.svc_C,
            probability=True,
            random_state=self.config.random_state,
        )

    def run(
        self,
        train_texts: list[str],
        y_train: np.ndarray,
        test_texts: list[str],
        y_test: np.ndarray,
    ) -> PipelineResult:
        """Execute the full five-step pipeline and return all outputs."""
        cfg = self.config

        # ----- Step 1: semantic extraction --------------------------------- #
        logger.info("=== Step 1/5: LLM semantic extraction ===")
        E_train = self.embedder.embed(train_texts)
        E_test = self.embedder.embed(test_texts)

        # ----- Step 2: topological compression ----------------------------- #
        logger.info("=== Step 2/5: UMAP topological compression ===")
        X_train = self.compressor.fit_transform(E_train)
        X_test = self.compressor.transform(E_test)

        # ----- Step 3: quantum kernel matrices ----------------------------- #
        logger.info("=== Step 3/5: Quantum kernel generation ===")
        K_train = self.quantum.train_kernel(X_train)
        K_test = self.quantum.cross_kernel(X_test, X_train)

        # ----- Step 4: Stage 1 supervised QSVC ----------------------------- #
        logger.info("=== Step 4/5: Stage 1 — supervised QSVC ===")
        self.classifier.fit(K_train, y_train)
        y_pred = self.classifier.predict(K_test)
        y_score = self.classifier.decision_function(K_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_score)
        except ValueError:
            auc = float("nan")  # single-class test split
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Background/Normal (0)", "Target/Anomaly (1)"],
            zero_division=0,
        )
        logger.info(
            "QSVC test metrics — accuracy=%.4f  f1=%.4f  roc_auc=%.4f",
            acc,
            f1,
            auc,
        )

        # ----- Step 5: Stage 2 unsupervised sub-clustering ----------------- #
        logger.info("=== Step 5/5: Stage 2 — spectral sub-clustering ===")
        anomaly_idx = np.flatnonzero(y_pred == 1)
        cluster_labels: np.ndarray | None = None

        if len(anomaly_idx) >= cfg.n_clusters:
            # Quantum affinity between the predicted anomalies themselves.
            # The test-vs-test sub-kernel is required (not test-vs-train), so
            # evaluate it on the anomalous test points directly.
            X_anom = X_test[anomaly_idx]
            K_anom = self.quantum.train_kernel(X_anom)
            # Symmetrize + clip: fidelity estimates carry tiny numerical noise
            # and SpectralClustering requires a symmetric non-negative affinity.
            K_anom = np.clip((K_anom + K_anom.T) / 2.0, 0.0, 1.0)

            spectral = SpectralClustering(
                n_clusters=cfg.n_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=cfg.random_state,
            )
            cluster_labels = spectral.fit_predict(K_anom)
            logger.info(
                "Discovered sub-clusters among %d predicted anomalies: %s",
                len(anomaly_idx),
                np.bincount(cluster_labels, minlength=cfg.n_clusters).tolist(),
            )
        else:
            logger.warning(
                "Only %d predicted anomalies (< n_clusters=%d): "
                "skipping spectral clustering.",
                len(anomaly_idx),
                cfg.n_clusters,
            )

        return PipelineResult(
            y_pred=y_pred,
            accuracy=acc,
            f1=f1,
            roc_auc=auc,
            report=report,
            anomaly_indices=anomaly_idx,
            cluster_labels=cluster_labels,
            train_kernel=K_train,
            test_kernel=K_test,
        )


# --------------------------------------------------------------------------- #
# Synthetic demo data: mock system logs with 1 normal class + 3 hidden
# anomaly sub-categories (so Stage 2 has real structure to discover).
# --------------------------------------------------------------------------- #
def make_synthetic_log_dataset(
    n_normal: int = 40,
    n_per_anomaly_type: int = 10,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Generate synthetic system-log strings and binary labels.

    Label 0: routine background logs.
    Label 1: anomalous logs drawn from three hidden sub-categories
             (auth attacks, memory faults, network failures) that the
             pipeline's Stage 2 should rediscover as sub-clusters.
    """
    rng = random.Random(seed)

    normal_templates = [
        "INFO scheduled backup for volume {v} completed in {n} seconds",
        "INFO user session for account {v} refreshed successfully token ttl {n}",
        "INFO health check on service {v} returned status 200 latency {n} ms",
        "INFO cache warmup finished for shard {v} with {n} entries loaded",
        "INFO configuration for module {v} reloaded cleanly version {n}",
    ]
    anomaly_templates = {
        "auth_attack": [
            "ALERT repeated failed login for admin account {v} from ip 10.0.0.{n}",
            "ALERT brute force pattern detected on ssh port for host {v} attempts {n}",
            "ALERT privilege escalation attempt by user {v} blocked at level {n}",
        ],
        "memory_fault": [
            "ERROR out of memory killer terminated process {v} rss {n} mb",
            "ERROR segmentation fault in worker {v} at address 0x{n}f3",
            "ERROR heap corruption detected in allocator arena {v} size {n}",
        ],
        "network_failure": [
            "CRITICAL packet loss above threshold on interface {v} loss {n} percent",
            "CRITICAL bgp session dropped with peer {v} after {n} retries",
            "CRITICAL dns resolution timeout for upstream {v} exceeded {n} ms",
        ],
    }

    texts: list[str] = []
    labels: list[int] = []

    for _ in range(n_normal):
        template = rng.choice(normal_templates)
        texts.append(template.format(v=f"svc-{rng.randint(1, 99)}", n=rng.randint(1, 500)))
        labels.append(0)

    for templates in anomaly_templates.values():
        for _ in range(n_per_anomaly_type):
            template = rng.choice(templates)
            texts.append(
                template.format(v=f"node-{rng.randint(1, 99)}", n=rng.randint(1, 500))
            )
            labels.append(1)

    # Shuffle jointly so the train/test split sees a mix of everything.
    order = list(range(len(texts)))
    rng.shuffle(order)
    texts = [texts[i] for i in order]
    labels_arr = np.array([labels[i] for i in order], dtype=int)
    return texts, labels_arr


# --------------------------------------------------------------------------- #
# Demo entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    config = PipelineConfig(n_qubits=8, n_clusters=3)

    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    logger.info("Generating synthetic log dataset ...")
    texts, labels = make_synthetic_log_dataset()
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=config.random_state,
    )
    logger.info(
        "Dataset: %d train / %d test  (anomaly rate train=%.2f, test=%.2f)",
        len(train_texts),
        len(test_texts),
        y_train.mean(),
        y_test.mean(),
    )

    pipeline = HybridPipeline(config)
    result = pipeline.run(train_texts, y_train, test_texts, y_test)

    print("\n" + "=" * 70)
    print("STAGE 1 — QSVC CLASSIFICATION REPORT (quantum precomputed kernel)")
    print("=" * 70)
    print(result.report)
    print(f"Accuracy : {result.accuracy:.4f}")
    print(f"F1-Score : {result.f1:.4f}")
    print(f"ROC-AUC  : {result.roc_auc:.4f}")

    print("\n" + "=" * 70)
    print("STAGE 2 — QUANTUM SPECTRAL SUB-CLUSTERS OF PREDICTED ANOMALIES")
    print("=" * 70)
    if result.cluster_labels is None:
        print("Not enough predicted anomalies to form sub-clusters.")
    else:
        for test_idx, cluster in zip(result.anomaly_indices, result.cluster_labels):
            snippet = test_texts[test_idx][:64]
            print(f"  test[{test_idx:3d}] -> sub-cluster {cluster} | {snippet}")
        counts = np.bincount(result.cluster_labels, minlength=config.n_clusters)
        print(f"\nCluster sizes: {counts.tolist()}")
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
