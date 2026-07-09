"""
Hybrid Classical-Quantum Pipeline for Drug Cardiotoxicity Prediction
====================================================================

A two-stage hybrid **Classical-Quantum** pipeline that triages molecules for
cardiotoxicity (hERG / I_Kr blockade) and then discovers structural toxicity
sub-groups among the flagged molecules:

    SMILES strings
       |
       v
    [Step 1] LLM Semantic Extraction   ChemBERTa (DeepChem/ChemBERTa-77M-MTR)
       |                               -> dense (N, 384) chemical embeddings
       v
    [Step 2] Topological Compression   UMAP -> n_qubits dimensions
       |                               (n_neighbors=5, min_dist=0.1)
       v
    [Step 3] Quantum Kernel            ZZFeatureMap(reps=2, entanglement='linear')
       |                               fidelity kernel |<phi(x)|phi(y)>|^2
       v
    [Step 4] Stage 1: Supervised       SVC(kernel='precomputed')  ==>  QSVC triage
       |          Binary label: 0 = Safe (non-blocker), 1 = Toxic (hERG blocker)
       |          with class balancing, CV-tuned C, and threshold calibration,
       |          benchmarked against a classical RBF-SVC baseline.
       v
    [Step 5] Stage 2: Unsupervised     SpectralClustering(affinity='precomputed')
                                       over the quantum-kernel sub-matrix of the
                                       predicted-toxic molecules -> structural
                                       toxicity sub-groups.

Real data, real model
----------------------
* **Model:** the pretrained ChemBERTa chemical language model is loaded from the
  Hugging Face Hub (no random-initialised stand-in).
* **Data:** an official, large hERG cardiotoxicity benchmark from Therapeutics
  Data Commons -- ``hERG_Karim`` (Karim et al. 2021, ~13,445 molecules) or
  ``hERG`` (Wang et al. 2016, ~655) -- with binary hERG-blocker labels, or any
  CSV you download (``local_path``). It is (optionally) subsampled to a balanced
  working set. A curated panel of 30 named marketed drugs is also provided for
  interpretable sanity checks.

Because a fidelity quantum kernel is O(N^2), the quantum stages run on a
balanced subsample (configurable). The classical baseline and cross-validation
quantify what the quantum path achieves relative to a strong classical model.

Requirements
------------
    Python 3.10+
    pip install qiskit qiskit-machine-learning qiskit-aer scikit-learn \
                umap-learn transformers torch pandas rdkit

Network: Step 1 downloads ChemBERTa from ``huggingface.co`` (cached after first
run). The hERG data comes from an official source of your choice -- Therapeutics
Data Commons (``pip install PyTDC``) by default, or any CSV you download and
pass via ``local_path`` (see ``load_herg_dataset``).

Author: CardiogenAI
"""

from __future__ import annotations

import logging
import os
import tarfile
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

import umap

from qiskit.quantum_info import Statevector

# --- Qiskit version-compatibility shim ------------------------------------- #
# Qiskit 2.x replaced the V1 ``Sampler`` with ``StatevectorSampler`` and
# deprecated the ``ZZFeatureMap`` class in favour of the ``zz_feature_map``
# function. Prefer the modern API and fall back for Qiskit 1.x installs.
try:  # Qiskit >= 1.3
    from qiskit.circuit.library import zz_feature_map as _zz_feature_map
except ImportError:  # Qiskit < 1.3
    from qiskit.circuit.library import ZZFeatureMap as _zz_feature_map

# The fidelity primitive is used only for the optional hardware-compatible
# kernel path; the default kernel is an exact statevector computation.
try:  # Qiskit 2.x
    from qiskit.primitives import StatevectorSampler as _Sampler
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    _HAS_FIDELITY = True
except ImportError:  # pragma: no cover - Qiskit 1.x
    try:
        from qiskit.primitives import Sampler as _Sampler
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        _HAS_FIDELITY = True
    except ImportError:
        _HAS_FIDELITY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cardiotox_qml")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Central hyperparameter registry for the full hybrid pipeline."""

    # --- Step 1: ChemBERTa embedding ---
    model_name: str = "DeepChem/ChemBERTa-77M-MTR"
    max_token_length: int = 128
    embedding_batch_size: int = 32
    pooling: str = "mean"  # "mean" (masked mean), "cls", or "pooler"

    # --- Step 2: UMAP compression ---
    n_qubits: int = 8               # UMAP output dim == number of qubits (8 or 16)
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    # --- Step 3: Quantum feature map / kernel ---
    feature_map_reps: int = 2
    entanglement: str = "linear"
    use_fidelity_primitive: bool = False  # True -> Qiskit FidelityQuantumKernel

    # --- Step 4: QSVC triage ---
    svc_C: float = 1.0
    tune_C: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)  # CV grid; () to disable
    class_weight: str | None = "balanced"
    calibrate_threshold: bool = True      # pick decision threshold from train CV
    cv_folds: int = 5
    run_classical_baseline: bool = True

    # --- Step 5: Spectral clustering ---
    n_clusters: int = 3

    # --- Data / split / reproducibility ---
    n_samples: int = 800                  # balanced hERG working-set size
    max_quantum_samples: int = 1500       # hard cap on molecules in the quantum kernel
    test_size: float = 0.25
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Step 1: Semantic Extraction (ChemBERTa LLM)
# --------------------------------------------------------------------------- #
class MoleculeEmbedder:
    """Wraps the pretrained ChemBERTa chemical language model to turn SMILES
    strings into dense molecular embeddings of shape (N, Hidden_Dim) -- 384 for
    ``DeepChem/ChemBERTa-77M-MTR``.

    Pooling strategies (config.pooling):
        * "mean"   -- attention-masked mean of the final hidden states
                      (default: deterministic, uses only pretrained weights).
        * "cls"    -- final hidden state of the leading <s>/[CLS] token.
        * "pooler" -- the model's ``pooler_output`` head (falls back to masked
                      mean pooling if the checkpoint ships no pooler weights --
                      ChemBERTa-77M-MTR does not).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None  # loaded lazily on first embed() (or via load())

    def load(self) -> "MoleculeEmbedder":
        """Load the pretrained ChemBERTa tokenizer and encoder from the HF Hub.

        Raises a clear, actionable error if the weights cannot be obtained --
        there is deliberately no random-initialised fallback.
        """
        if self.model is not None:
            return self
        logger.info(
            "Loading ChemBERTa '%s' on device '%s' ...",
            self.config.model_name,
            self.device,
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
        except (OSError, EnvironmentError) as err:
            raise RuntimeError(
                f"Could not load pretrained model '{self.config.model_name}' "
                f"from the Hugging Face Hub ({err}). This pipeline requires the "
                "real ChemBERTa weights. Ensure the host can reach "
                "'huggingface.co', or pre-populate the HF cache (HF_HOME) / set "
                "HF_ENDPOINT to a reachable mirror, then retry."
            ) from err
        self.model.eval()
        logger.info(
            "ChemBERTa ready: hidden_dim=%d, vocab=%d, pooling='%s'",
            self.model.config.hidden_size,
            self.model.config.vocab_size,
            self.config.pooling,
        )
        return self

    @property
    def hidden_dim(self) -> int:
        if self.model is None:
            self.load()
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def embed(self, smiles: list[str]) -> np.ndarray:
        """Embed a list of SMILES strings into a dense (N, Hidden_Dim) array."""
        if not smiles:
            raise ValueError("MoleculeEmbedder.embed received an empty SMILES list.")
        if self.model is None:
            self.load()

        vectors: list[np.ndarray] = []
        bs = self.config.embedding_batch_size
        for start in range(0, len(smiles), bs):
            chunk = smiles[start : start + bs]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.config.max_token_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            pooled = self._pool(outputs, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy())
            if len(smiles) > 200 and (start // bs) % 10 == 0:
                logger.info("  embedded %d / %d ...", min(start + bs, len(smiles)), len(smiles))

        embeddings = np.vstack(vectors).astype(np.float64)
        logger.info("Embedded %d molecules -> shape %s", len(smiles), embeddings.shape)
        return embeddings

    def _pool(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        """Reduce token-level hidden states to one vector per molecule."""
        strategy = self.config.pooling
        last_hidden = outputs.last_hidden_state  # (B, T, H)
        if strategy == "cls":
            return last_hidden[:, 0]
        if strategy == "pooler":
            pooler = getattr(outputs, "pooler_output", None)
            if pooler is not None:
                return pooler
            logger.warning("Checkpoint has no pooler_output; using masked mean pooling.")
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


# --------------------------------------------------------------------------- #
# Step 2: Topological Compression (UMAP)
# --------------------------------------------------------------------------- #
class TopologicalCompressor:
    """Compresses ChemBERTa embeddings to `n_qubits` dimensions with UMAP, then
    rescales each feature into [0, pi] so it is a valid rotation angle for the
    quantum feature map.
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
        logger.info(
            "Fitting UMAP: %s -> %d dims (n_neighbors=%d, min_dist=%.2f, metric=%s)",
            X_train.shape, self.config.n_qubits, self.config.umap_n_neighbors,
            self.config.umap_min_dist, self.config.umap_metric,
        )
        reduced = self.reducer.fit_transform(X_train)
        scaled = self.scaler.fit_transform(reduced)
        self._fitted = True
        return scaled.astype(np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first.")
        reduced = self.reducer.transform(X)
        scaled = self.scaler.transform(reduced)
        return np.clip(scaled, 0.0, np.pi).astype(np.float64)


# --------------------------------------------------------------------------- #
# Step 3: Quantum Feature Map & Kernel
# --------------------------------------------------------------------------- #
class QuantumProcessor:
    """Builds the ZZFeatureMap and computes the fidelity quantum kernel

        K(x, y) = |<phi(x)|phi(y)>|^2

    with |phi(.)> prepared by the ZZFeatureMap.

    By default the kernel is computed **exactly** by simulating each feature
    map circuit once with the statevector simulator and forming pairwise
    overlaps -- O(N) circuit simulations + O(N^2) vectorised inner products.
    This is mathematically identical to a noiseless ``FidelityQuantumKernel``
    but scales to hundreds/thousands of molecules (the compute-uncompute
    primitive runs O(N^2) circuits and does not). Set
    ``use_fidelity_primitive=True`` to use Qiskit's ``FidelityQuantumKernel``.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_map = _zz_feature_map(
            feature_dimension=config.n_qubits,
            reps=config.feature_map_reps,
            entanglement=config.entanglement,
        )
        # Robust parameter ordering (x[0], x[1], ... x[k]) by parsed index so
        # binding is correct even for n_qubits >= 10.
        self._params = sorted(
            self.feature_map.parameters,
            key=lambda p: int(p.name.split("[")[1].rstrip("]")),
        )
        self._fqk = None
        if config.use_fidelity_primitive:
            if not _HAS_FIDELITY:
                raise RuntimeError("FidelityQuantumKernel/fidelity primitive unavailable.")
            self._fqk = FidelityQuantumKernel(
                feature_map=self.feature_map,
                fidelity=ComputeUncompute(sampler=_Sampler()),
            )
        logger.info(
            "Quantum kernel ready: ZZFeatureMap(qubits=%d, reps=%d, "
            "entanglement='%s'), depth=%d, mode=%s",
            config.n_qubits, config.feature_map_reps, config.entanglement,
            self.feature_map.decompose().depth(),
            "fidelity-primitive" if self._fqk else "exact-statevector",
        )

    def _statevectors(self, X: np.ndarray) -> np.ndarray:
        """Simulate the feature map once per row; return (N, 2**n_qubits) complex."""
        dim = 2 ** self.feature_map.num_qubits
        states = np.empty((len(X), dim), dtype=np.complex128)
        for i, x in enumerate(X):
            bound = self.feature_map.assign_parameters(
                {p: float(v) for p, v in zip(self._params, x)}
            )
            states[i] = Statevector.from_instruction(bound).data
        return states

    def kernel(self, X_a: np.ndarray, X_b: np.ndarray | None = None) -> np.ndarray:
        """Fidelity kernel between X_a and X_b (or X_a with itself)."""
        n_b = len(X_a) if X_b is None else len(X_b)
        logger.info("Computing quantum kernel (%d x %d, %s) ...", len(X_a), n_b,
                    "fidelity-primitive" if self._fqk else "exact-statevector")
        if self._fqk is not None:
            return self._fqk.evaluate(x_vec=X_a) if X_b is None \
                else self._fqk.evaluate(x_vec=X_a, y_vec=X_b)
        S_a = self._statevectors(X_a)
        S_b = S_a if X_b is None else self._statevectors(X_b)
        K = np.abs(S_a.conj() @ S_b.T) ** 2
        return np.clip(K, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    # Stage 1 (quantum QSVC, held-out test set)
    y_pred_test: np.ndarray
    y_score_test: np.ndarray
    threshold: float
    accuracy: float
    f1: float
    roc_auc: float
    report: str
    best_C: float
    cv_auc_mean: float
    cv_auc_std: float

    # Classical baseline (raw ChemBERTa embeddings)
    baseline: dict | None

    # Stage 2 (mechanism sub-groups over predicted-toxic molecules)
    toxic_index: np.ndarray
    cluster_labels: np.ndarray | None
    cluster_summary: dict | None

    # Bookkeeping
    order: np.ndarray
    n_train: int
    full_kernel: np.ndarray = field(repr=False, default=None)


# --------------------------------------------------------------------------- #
# Steps 4 & 5: two-stage hybrid pipeline
# --------------------------------------------------------------------------- #
class ToxicityPipeline:
    """End-to-end orchestrator (Steps 1-5) with class balancing, CV-tuned C,
    decision-threshold calibration, a classical baseline, and cross-validated
    reporting."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.embedder = MoleculeEmbedder(self.config)
        self.compressor = TopologicalCompressor(self.config)
        self.quantum = QuantumProcessor(self.config)
        self.classifier: SVC | None = None

    # ---- kernel cross-validation helpers ---------------------------------- #
    def _kernel_oof_and_cv(self, K_tr, y_tr, C):
        """Out-of-fold decision scores + per-fold AUCs for a precomputed kernel.

        Manual CV is required because sklearn's cross_val_* slice only the rows
        of a precomputed kernel, not the train columns.
        """
        cfg = self.config
        skf = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
        oof = np.zeros(len(y_tr))
        aucs = []
        for tr, va in skf.split(np.zeros(len(y_tr)), y_tr):
            svc = SVC(kernel="precomputed", C=C, class_weight=cfg.class_weight)
            svc.fit(K_tr[np.ix_(tr, tr)], y_tr[tr])
            s = svc.decision_function(K_tr[np.ix_(va, tr)])
            oof[va] = s
            aucs.append(roc_auc_score(y_tr[va], s))
        return oof, np.array(aucs)

    def run(self, smiles, labels, names=None) -> PipelineResult:
        cfg = self.config
        labels = np.asarray(labels, dtype=int)
        names = list(names) if names is not None else [f"mol_{i}" for i in range(len(smiles))]

        # ---- cap quantum working set for tractability --------------------- #
        if len(smiles) > cfg.max_quantum_samples:
            idx = _balanced_subsample(labels, cfg.max_quantum_samples, cfg.random_state)
            smiles = [smiles[i] for i in idx]
            names = [names[i] for i in idx]
            labels = labels[idx]
            logger.info("Capped working set to %d molecules for the quantum kernel.", len(smiles))

        # ---- stratified split --------------------------------------------- #
        idx_train, idx_test = train_test_split(
            np.arange(len(smiles)), test_size=cfg.test_size,
            stratify=labels, random_state=cfg.random_state,
        )
        order = np.concatenate([idx_train, idx_test])
        n_train = len(idx_train)
        smiles_ord = [smiles[i] for i in order]
        y_ord = labels[order]
        y_train, y_test = y_ord[:n_train], y_ord[n_train:]
        logger.info(
            "Working set: %d molecules (%d train / %d test), toxic rate train=%.2f test=%.2f",
            len(smiles), n_train, len(idx_test), y_train.mean(), y_test.mean(),
        )

        # ---- Step 1: ChemBERTa embeddings --------------------------------- #
        logger.info("=== Step 1/5: ChemBERTa semantic extraction ===")
        E_all = self.embedder.embed(smiles_ord)
        E_train, E_test = E_all[:n_train], E_all[n_train:]

        # ---- Step 2: UMAP compression ------------------------------------- #
        logger.info("=== Step 2/5: UMAP topological compression ===")
        X_train = self.compressor.fit_transform(E_train)
        X_test = self.compressor.transform(E_test)
        X_all = np.vstack([X_train, X_test])

        # ---- Step 3: quantum kernel (one full N x N; blocks are sub-matrices) #
        logger.info("=== Step 3/5: Quantum kernel generation ===")
        K_full = self.quantum.kernel(X_all)
        K_full = np.clip((K_full + K_full.T) / 2.0, 0.0, 1.0)  # symmetrise
        K_train = K_full[:n_train, :n_train]
        K_test = K_full[n_train:, :n_train]

        # ---- Step 4: Stage 1 supervised QSVC triage ----------------------- #
        logger.info("=== Step 4/5: Stage 1 - supervised QSVC triage ===")
        C_grid = cfg.tune_C or (cfg.svc_C,)
        best_C, best_auc, best_oof = cfg.svc_C, -np.inf, None
        for C in C_grid:
            oof, aucs = self._kernel_oof_and_cv(K_train, y_train, C)
            logger.info("  C=%-6g  CV AUC=%.4f +/- %.4f", C, aucs.mean(), aucs.std())
            if aucs.mean() > best_auc:
                best_C, best_auc, best_oof, best_aucs = C, aucs.mean(), oof, aucs
        logger.info("Selected C=%g (CV AUC=%.4f)", best_C, best_auc)

        self.classifier = SVC(kernel="precomputed", C=best_C,
                              class_weight=cfg.class_weight)
        self.classifier.fit(K_train, y_train)

        # decision-threshold calibration from out-of-fold train scores (Youden's J)
        threshold = 0.0
        if cfg.calibrate_threshold:
            fpr, tpr, thr = roc_curve(y_train, best_oof)
            threshold = float(thr[np.argmax(tpr - fpr)])
            logger.info("Calibrated decision threshold: %.4f (Youden's J on train CV)", threshold)

        y_score_test = self.classifier.decision_function(K_test)
        y_pred_test = (y_score_test >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_score_test)
        except ValueError:
            auc = float("nan")
        report = classification_report(
            y_test, y_pred_test, target_names=["Safe (0)", "Toxic (1)"], zero_division=0,
        )
        logger.info("QSVC test: accuracy=%.4f  f1=%.4f  roc_auc=%.4f", acc, f1, auc)

        # ---- classical baseline on raw embeddings ------------------------- #
        baseline = None
        if cfg.run_classical_baseline:
            baseline = self._classical_baseline(E_train, y_train, E_test, y_test)

        # ---- Step 5: Stage 2 mechanism sub-grouping ----------------------- #
        logger.info("=== Step 5/5: Stage 2 - spectral mechanism discovery ===")
        y_pred_all = (self.classifier.decision_function(K_full[:, :n_train]) >= threshold).astype(int)
        toxic_local = np.flatnonzero(y_pred_all == 1)
        toxic_index = order[toxic_local]
        cluster_labels, cluster_summary = None, None
        if len(toxic_local) >= cfg.n_clusters:
            affinity = np.clip(K_full[np.ix_(toxic_local, toxic_local)], 0.0, 1.0)
            spectral = SpectralClustering(
                n_clusters=cfg.n_clusters, affinity="precomputed",
                assign_labels="kmeans", random_state=cfg.random_state,
            )
            cluster_labels = spectral.fit_predict(affinity)
            cluster_summary = _characterise_clusters(
                [smiles[i] for i in toxic_index], cluster_labels, cfg.n_clusters
            )
            logger.info("Discovered %d sub-groups among %d predicted-toxic molecules: sizes=%s",
                        cfg.n_clusters, len(toxic_local),
                        np.bincount(cluster_labels, minlength=cfg.n_clusters).tolist())
        else:
            logger.warning("Only %d predicted-toxic molecules (< n_clusters=%d): skipping clustering.",
                           len(toxic_local), cfg.n_clusters)

        return PipelineResult(
            y_pred_test=y_pred_test, y_score_test=y_score_test, threshold=threshold,
            accuracy=acc, f1=f1, roc_auc=auc, report=report,
            best_C=best_C, cv_auc_mean=float(best_auc), cv_auc_std=float(best_aucs.std()),
            baseline=baseline, toxic_index=toxic_index, cluster_labels=cluster_labels,
            cluster_summary=cluster_summary, order=order, n_train=n_train, full_kernel=K_full,
        )

    def _classical_baseline(self, E_train, y_train, E_test, y_test) -> dict:
        """Classical RBF-SVC and logistic regression on the raw ChemBERTa
        embeddings, with cross-validated AUC -- the reference ceiling."""
        cfg = self.config
        logger.info("Classical baselines on raw %d-d ChemBERTa embeddings ...", E_train.shape[1])
        out = {}
        models = {
            "rbf_svc": make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=10.0, gamma="scale",
                    class_weight=cfg.class_weight),
            ),
            "logreg": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight=cfg.class_weight),
            ),
        }
        skf = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
        for name, model in models.items():
            cv_auc = cross_val_score(model, E_train, y_train, cv=skf, scoring="roc_auc")
            model.fit(E_train, y_train)
            score = (model.decision_function(E_test)
                     if hasattr(model, "decision_function")
                     else model.predict_proba(E_test)[:, 1])
            pred = (score >= 0).astype(int) if name == "rbf_svc" else model.predict(E_test)
            out[name] = {
                "cv_auc_mean": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std()),
                "test_auc": float(roc_auc_score(y_test, score)),
                "test_acc": float(accuracy_score(y_test, pred)),
                "test_f1": float(f1_score(y_test, pred, zero_division=0)),
            }
            logger.info("  %-8s CV AUC=%.4f+/-%.4f | test AUC=%.4f acc=%.4f f1=%.4f",
                        name, out[name]["cv_auc_mean"], out[name]["cv_auc_std"],
                        out[name]["test_auc"], out[name]["test_acc"], out[name]["test_f1"])
        return out


# --------------------------------------------------------------------------- #
# Cluster characterisation (physicochemical fingerprint per sub-group)
# --------------------------------------------------------------------------- #
def _characterise_clusters(smiles, labels, n_clusters) -> dict:
    """Summarise each mechanism cluster by mean RDKit physicochemical
    descriptors (MW, LogP, TPSA, aromatic rings, H-bond donors/acceptors) when
    RDKit is available; otherwise report only sizes."""
    summary = {c: {"size": int(np.sum(labels == c))} for c in range(n_clusters)}
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return summary
    props = {c: [] for c in range(n_clusters)}
    for smi, c in zip(smiles, labels):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        props[int(c)].append((
            Descriptors.MolWt(m), Crippen.MolLogP(m), Descriptors.TPSA(m),
            Lipinski.NumAromaticRings(m), Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
        ))
    for c in range(n_clusters):
        if props[c]:
            arr = np.array(props[c])
            summary[c].update({
                "MolWt": round(float(arr[:, 0].mean()), 1),
                "LogP": round(float(arr[:, 1].mean()), 2),
                "TPSA": round(float(arr[:, 2].mean()), 1),
                "AromaticRings": round(float(arr[:, 3].mean()), 2),
                "HDonors": round(float(arr[:, 4].mean()), 2),
                "HAcceptors": round(float(arr[:, 5].mean()), 2),
            })
    return summary


# --------------------------------------------------------------------------- #
# Data loading -- official hERG cardiotoxicity datasets
# --------------------------------------------------------------------------- #
# Recommended official source: Therapeutics Data Commons (TDC).
#     https://tdcommons.ai/single_pred_tasks/tox/#herg-blockers-karim-et-al
#     pip install PyTDC
#     from tdc.single_pred import Tox
#     Tox(name="hERG")        -> Wang et al. 2016   (~655 drugs)
#     Tox(name="herg_karim")  -> Karim et al. 2021  (~13,445 drugs)  # lowercase!
# get_data() returns a DataFrame with columns: Drug_ID, Drug (SMILES), Y (label;
# 1 = hERG blocker / cardiotoxic, 0 = non-blocker).
#
# Alternatively, pass ``local_path`` pointing at any CSV you downloaded (from
# TDC, ChEMBL target CHEMBL240, a Kaggle mirror, etc.). The SMILES and label
# columns are auto-detected (or name them with ``smiles_col`` / ``label_col``).

# Common column-name aliases used for auto-detection in a user-supplied CSV.
_SMILES_ALIASES = ("smiles", "drug", "canonical_smiles", "smi", "mol", "structure")
_LABEL_ALIASES = ("y", "activity", "label", "class", "target", "toxic",
                   "blocker", "herg", "active", "outcome")


def _default_cache_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".cache", "cardiogenai")
    os.makedirs(d, exist_ok=True)
    return d


def _balanced_subsample(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Return indices for a class-balanced random subsample of size ~n."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = max(1, n // len(classes))
    picks = []
    for c in classes:
        c_idx = np.flatnonzero(labels == c)
        picks.append(rng.choice(c_idx, size=min(per, len(c_idx)), replace=False))
    idx = np.concatenate(picks)
    rng.shuffle(idx)
    return idx


def _pick_column(columns, given, aliases, kind):
    """Resolve a column name from an explicit choice or alias list."""
    lower = {c.lower(): c for c in columns}
    if given is not None:
        if given in columns:
            return given
        if given.lower() in lower:
            return lower[given.lower()]
        raise ValueError(f"{kind} column '{given}' not found in {list(columns)}")
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    # last resort: substring match
    for c in columns:
        if any(a in c.lower() for a in aliases):
            return c
    return None


def _read_dataframe(path: str):
    """Read a tabular file (CSV / CSV.gz / TSV / .tab / .txt / .tar.xz) -> DataFrame."""
    import pandas as pd

    low = path.lower()
    if low.endswith((".tar.xz", ".tar.gz", ".tgz")):
        mode = "r:xz" if low.endswith(".tar.xz") else "r:gz"
        with tarfile.open(path, mode) as tar:
            member = next(m for m in tar.getmembers()
                          if m.name.lower().endswith((".csv", ".tsv", ".tab", ".txt")))
            with tar.extractfile(member) as fh:
                sep = "\t" if member.name.lower().endswith((".tsv", ".tab")) else ","
                return pd.read_csv(fh, sep=sep)
    if low.endswith((".tsv", ".tab")):
        return pd.read_csv(path, sep="\t")
    if low.endswith(".txt"):
        return pd.read_csv(path, sep=None, engine="python")  # sniff delimiter
    return pd.read_csv(path)  # pandas transparently handles .csv/.gz/.zip


def _from_local(path, smiles_col, label_col):
    """Load SMILES + binary labels from a user-supplied CSV file."""
    logger.info("Loading hERG data from local file: %s", path)
    df = _read_dataframe(path)
    s_col = _pick_column(df.columns, smiles_col, _SMILES_ALIASES, "SMILES")
    l_col = _pick_column(df.columns, label_col, _LABEL_ALIASES, "label")
    if s_col is None:
        raise ValueError(f"Could not find a SMILES column in {list(df.columns)}; "
                         "pass smiles_col=...")
    if l_col is None:
        if len(df.columns) == 2:  # 2-column file: the non-SMILES column is the label
            l_col = [c for c in df.columns if c != s_col][0]
        else:
            raise ValueError(f"Could not find a label column in {list(df.columns)}; "
                             "pass label_col=...")
    df = df[[s_col, l_col]].dropna()
    return df[s_col].astype(str).tolist(), df[l_col].astype(int).to_numpy()


def _from_tdc(tdc_name, cache_dir):
    """Load an official hERG dataset via Therapeutics Data Commons (PyTDC).

    TDC dataset names are matched case-insensitively but must exist in the
    installed PyTDC version. The canonical hERG-Karim name is the lowercase
    ``herg_karim`` (per the TDC dataset card); older/other versions vary, so we
    also auto-discover the registered Tox names and try sensible candidates
    before giving up.
    """
    try:
        from tdc.single_pred import Tox
    except ImportError as err:
        raise RuntimeError(
            "The default hERG source needs Therapeutics Data Commons. Either\n"
            "  pip install PyTDC\n"
            "or download an official hERG CSV and pass it via "
            "load_herg_dataset(local_path='herg.csv'). "
            "Official dataset: https://tdcommons.ai/single_pred_tasks/tox/"
        ) from err

    cache_dir = cache_dir or _default_cache_dir()
    candidates = [tdc_name, tdc_name.lower(), "herg_karim", "hERG_Karim", "hERG"]

    # Auto-discover the exact names this PyTDC version registers for Tox, and
    # prefer a hERG-Karim match. Falls back silently if the util API differs.
    valid = None
    try:
        from tdc.utils import retrieve_dataset_names
        valid = retrieve_dataset_names("Tox")
        pref = [v for v in valid if "herg" in v.lower() and "karim" in v.lower()]
        pref += [v for v in valid if "herg" in v.lower() and "central" not in v.lower()]
        candidates = pref + candidates
    except Exception:  # noqa: BLE001 - discovery is best-effort
        pass

    last_err = None
    for name in dict.fromkeys(candidates):  # de-dupe, preserve order
        try:
            logger.info("Loading official TDC dataset '%s' ...", name)
            data = Tox(name=name, path=cache_dir)
            df = data.get_data()  # columns: Drug_ID, Drug (SMILES), Y (0/1)
            return df["Drug"].astype(str).tolist(), df["Y"].astype(int).to_numpy()
        except Exception as err:  # noqa: BLE001 - try the next candidate name
            last_err = err

    raise RuntimeError(
        f"Could not load an hERG dataset from TDC. Registered Tox names: "
        f"{valid}. Tried {list(dict.fromkeys(candidates))}. Last error: {last_err}. "
        "You can instead download an hERG CSV and pass "
        "load_herg_dataset(local_path='herg.csv')."
    )


def load_herg_dataset(
    n_samples: int | None = 800,
    balanced: bool = True,
    *,
    source: str = "tdc",
    tdc_name: str = "herg_karim",
    local_path: str | None = None,
    smiles_col: str | None = None,
    label_col: str | None = None,
    cache_dir: str | None = None,
    random_state: int = 42,
    validate: bool = True,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load an official hERG cardiotoxicity dataset.

    Label convention: 1 = hERG blocker (cardiotoxic), 0 = non-blocker.

    Sources
    -------
    local_path : str
        Path to a CSV / CSV.gz you downloaded from any official source. The
        SMILES and binary-label columns are auto-detected (aliases include
        'smiles'/'drug' and 'y'/'activity'/'label'); override with
        ``smiles_col`` / ``label_col``. Takes precedence over ``source``.
    source="tdc" (default) : Therapeutics Data Commons (needs ``pip install PyTDC``).
        ``tdc_name="herg_karim"`` -> Karim et al. 2021 (~13,445 molecules);
        ``tdc_name="hERG"``       -> Wang et al. 2016 (~655 molecules).
        The exact registered name is auto-discovered, so casing variants are OK.

    Parameters
    ----------
    n_samples : int or None
        If given, return a (class-balanced if ``balanced``) random subsample of
        this size -- appropriate for the O(N^2) quantum kernel. ``None`` returns
        the full dataset.

    Returns (smiles, labels, names).
    """
    if local_path is not None:
        smiles, labels = _from_local(local_path, smiles_col, label_col)
    elif source == "tdc":
        smiles, labels = _from_tdc(tdc_name, cache_dir)
    else:
        raise ValueError(f"Unknown source '{source}'. Use source='tdc' or pass local_path=...")

    if validate:
        smiles, labels = _filter_valid_smiles(smiles, labels)

    logger.info("Loaded hERG dataset: %d molecules (blockers=%d, non-blockers=%d)",
                len(smiles), int(labels.sum()), int((labels == 0).sum()))

    if n_samples is not None and n_samples < len(smiles):
        idx = (_balanced_subsample(labels, n_samples, random_state) if balanced
               else np.random.default_rng(random_state).choice(len(smiles), n_samples, replace=False))
        smiles = [smiles[i] for i in idx]
        labels = labels[idx]
        logger.info("Subsampled to %d molecules (balanced=%s).", len(smiles), balanced)

    names = [f"hERG_{i:05d}" for i in range(len(smiles))]
    return smiles, labels, names


def _filter_valid_smiles(smiles, labels):
    """Drop chemically invalid SMILES using RDKit (if installed)."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return smiles, labels
    keep = [i for i, s in enumerate(smiles) if Chem.MolFromSmiles(s) is not None]
    if len(keep) != len(smiles):
        logger.info("Dropped %d invalid SMILES.", len(smiles) - len(keep))
    return [smiles[i] for i in keep], labels[keep]


def load_known_drug_panel() -> tuple[list[str], np.ndarray, list[str]]:
    """A small curated panel of 30 named marketed drugs (RDKit-canonical SMILES)
    with pharmacology-grounded cardiotoxicity labels -- useful as an
    interpretable sanity check / holdout alongside the large hERG dataset.

    1 = cardiotoxic (hERG blocker / torsadogen / CV-withdrawn); 0 = safe.
    """
    toxic = {
        "Terfenadine":  "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
        "Astemizole":   "COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1",
        "Cisapride":    "COc1cc(C(=O)NC2CCN(CCCOc3ccc(F)cc3)CC2)ccc1N",
        "Sertindole":   "CN1CCN(CCn2ccc3cc(F)ccc32)CC1",
        "Dofetilide":   "CN(CCc1ccc(NS(C)(=O)=O)cc1)CCc1ccc(NS(C)(=O)=O)cc1",
        "Sotalol":      "CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1",
        "Quinidine":    "C=C[C@H]1CN2CC[C@H]1C[C@H]2[C@@H](O)c1ccnc2ccc(OC)cc12",
        "Amiodarone":   "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
        "Haloperidol":  "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
        "Thioridazine": "CSc1ccc2c(c1)N(CCC1CCCCN1C)c1ccccc1S2",
        "Pimozide":     "O=c1[nH]c2ccccc2n1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1",
        "Bepridil":     "CC(C)COCC(Cn1ccnc1)N(Cc1ccccc1)c1ccccc1",
        "Ibutilide":    "CCCCCCCN(CC)CCCC(O)c1ccc(NS(C)(=O)=O)cc1",
        "Droperidol":   "O=C(CCCN1CC=CC(n2c(=O)[nH]c3ccccc32)CC1)c1ccc(F)cc1",
        "Vandetanib":   "COc1cc2ncnc(Nc3ccc(Br)cc3F)c2cc1OCC1CCN(C)CC1",
    }
    safe = {
        "Aspirin":         "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen":       "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "Acetaminophen":   "CC(=O)Nc1ccc(O)cc1",
        "Caffeine":        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "Metformin":       "CN(C)C(=N)NC(=N)N",
        "Naproxen":        "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
        "Fexofenadine":    "CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
        "Loratadine":      "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
        "Ascorbic_acid":   "O=C1O[C@H]([C@@H](O)CO)C(O)=C1O",
        "Diphenhydramine": "CN(C)CCOC(c1ccccc1)c1ccccc1",
        "Amoxicillin":     "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        "Lisinopril":      "NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O",
        "Metronidazole":   "Cc1ncc([N+](=O)[O-])n1CCO",
        "Omeprazole":      "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "Salbutamol":      "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    }
    names, smiles, labels = [], [], []
    for n, s in toxic.items():
        names.append(n); smiles.append(s); labels.append(1)
    for n, s in safe.items():
        names.append(n); smiles.append(s); labels.append(0)
    return smiles, np.array(labels, dtype=int), names


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def print_report(result: PipelineResult) -> None:
    cfg_line = "=" * 74
    print("\n" + cfg_line)
    print("STAGE 1 - QSVC CARDIOTOXICITY TRIAGE (quantum precomputed kernel)")
    print(cfg_line)
    print(result.report)
    print(f"Selected C        : {result.best_C:g}")
    print(f"Train CV ROC-AUC  : {result.cv_auc_mean:.4f} +/- {result.cv_auc_std:.4f}")
    print(f"Decision threshold: {result.threshold:.4f}")
    print(f"Test Accuracy     : {result.accuracy:.4f}")
    print(f"Test F1-Score     : {result.f1:.4f}")
    print(f"Test ROC-AUC      : {result.roc_auc:.4f}")

    if result.baseline:
        print("\n" + cfg_line)
        print("CLASSICAL BASELINE (raw 384-d ChemBERTa embeddings)")
        print(cfg_line)
        print(f"{'model':10s} {'CV ROC-AUC':>18s} {'test AUC':>10s} {'test acc':>10s} {'test f1':>9s}")
        print(f"{'quantum':10s} {result.cv_auc_mean:>10.4f}+/-{result.cv_auc_std:<5.4f} "
              f"{result.roc_auc:>10.4f} {result.accuracy:>10.4f} {result.f1:>9.4f}")
        for name, m in result.baseline.items():
            print(f"{name:10s} {m['cv_auc_mean']:>10.4f}+/-{m['cv_auc_std']:<5.4f} "
                  f"{m['test_auc']:>10.4f} {m['test_acc']:>10.4f} {m['test_f1']:>9.4f}")

    print("\n" + cfg_line)
    print("STAGE 2 - QUANTUM SPECTRAL TOXICITY SUB-GROUPS (predicted-toxic molecules)")
    print(cfg_line)
    if result.cluster_summary is None:
        print("Not enough predicted-toxic molecules to form sub-groups.")
    else:
        for c in sorted(result.cluster_summary):
            s = result.cluster_summary[c]
            extra = ""
            if "MolWt" in s:
                extra = (f" | MW={s['MolWt']}  LogP={s['LogP']}  TPSA={s['TPSA']}  "
                         f"ArRings={s['AromaticRings']}  HBD={s['HDonors']}  HBA={s['HAcceptors']}")
            print(f"  Sub-group {c}: n={s['size']:4d}{extra}")
    print("\nPipeline finished successfully.")


# --------------------------------------------------------------------------- #
# Demo entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    config = PipelineConfig(n_qubits=8, n_clusters=3, n_samples=800)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    logger.info("Loading real hERG cardiotoxicity dataset ...")
    smiles, labels, names = load_herg_dataset(
        n_samples=config.n_samples, random_state=config.random_state
    )

    result = ToxicityPipeline(config).run(smiles, labels, names)
    print_report(result)


if __name__ == "__main__":
    main()
