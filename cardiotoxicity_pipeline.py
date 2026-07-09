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
    [Step 2] Topological Compression   supervised UMAP -> n_qubits dimensions
       |                               (fit on train labels; test mapped label-free)
       v
    [Step 3] Quantum Kernel            bandwidth feature map (H, RZ(2c*z),
       |                               RZZ(2c^2*z_m*z_n); NO (pi-z) offsets)
       |                               fidelity |<phi(x)|phi(y)>|^2  OR projected
       |                               (Bloch-vector Gaussian) kernel
       v
    [Step 4] Stage 1: Supervised       SVC(kernel='precomputed')  ==>  QSVC triage
       |          Binary label: 0 = Safe (non-blocker), 1 = Toxic (hERG blocker).
       |          Class balancing; bandwidth c + C + threshold chosen on a clean
       |          validation holdout (leakage-free); classical baselines on the
       |          same features and on raw ChemBERTa.
       v
    [Step 5] Stage 2: Unsupervised     SpectralClustering(affinity='precomputed')
                                       over the quantum-kernel sub-matrix of the
                                       predicted-toxic molecules -> structural
                                       toxicity sub-groups.

Real data, real model
----------------------
* **Model:** the pretrained ChemBERTa chemical language model is loaded from the
  Hugging Face Hub (no random-initialised stand-in).
* **Data:** an official hERG cardiotoxicity dataset from Therapeutics Data
  Commons -- by default **hERGCentral** (``herg_central``, label ``hERG_inhib``:
  a ~306,893-molecule electrophysiology screen, binary blockade) -- or
  ``herg_karim`` / ``hERG``, or any CSV you download (``local_path``). It is
  subsampled to a balanced working set for the O(N^2) quantum kernel. A curated
  panel of 30 named marketed drugs is also provided for interpretable checks.

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

import umap

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
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
    # n_qubits controls the Hilbert-space dimension (2**n_qubits). Large values
    # cause *kernel concentration*: fidelities -> 0 for all x != y and the kernel
    # degenerates to the identity matrix. Keep n_qubits small (6-10).
    n_qubits: int = 8               # UMAP output dim == number of qubits
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    # Supervised UMAP shapes the embedding with class labels (train only; test is
    # mapped label-free via transform()). target_weight in [0,1]: 0 = unsupervised,
    # 1 = fully supervised (risks train/test mismatch); 0.3-0.6 is a good range.
    umap_supervised: bool = True
    umap_target_weight: float = 0.5

    # --- Step 3: Quantum feature map / kernel ---
    feature_map_reps: int = 1       # each rep compounds phase spread -> use 1
    entanglement: str = "linear"    # "linear", "circular", or "full"
    kernel_type: str = "fidelity"   # "fidelity" or "projected" (Huang et al. 2021)
    projected_gamma: float = 1.0    # Gaussian width for the projected kernel
    # Encoding bandwidth c: phases are 2*c*z (and 2*c^2*z_m*z_n). c < 1 counters
    # concentration (Shaydulin & Wild 2022). tune_bandwidth grid-searches it by
    # kernel-target alignment; () disables and uses encoding_scale directly.
    encoding_scale: float = 0.4
    tune_bandwidth: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
    use_fidelity_primitive: bool = False  # (legacy flag; unused by the custom map)

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
    # NOTE: the exact statevector kernel holds 2**n_qubits complex amplitudes per
    # molecule, so peak memory ~ n_samples * 2**n_qubits * 16 bytes. With the
    # recommended n_qubits in 6-10, a few hundred to ~1000 molecules is fine.
    n_samples: int = 408                  # balanced hERG working-set size
    max_quantum_samples: int = 1000       # hard cap on molecules in the quantum kernel
    test_size: float = 0.25
    val_size: float = 0.2                 # fraction of TRAIN held out for leakage-free
                                          # model selection (supervised UMAP is fit on
                                          # the remaining "fit" split only)
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
    """Compresses ChemBERTa embeddings to `n_qubits` dimensions with UMAP and
    MinMax-scales each feature to [0, 1] (the quantum feature map then applies
    the encoding bandwidth). Optionally uses **supervised** UMAP so the embedding
    carries class structure -- fit on train labels, transform test label-free.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.reducer = umap.UMAP(
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            n_components=config.n_qubits,
            metric=config.umap_metric,
            target_weight=config.umap_target_weight,
            random_state=config.random_state,
        )
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._fitted = False

    def fit_transform(self, X_train: np.ndarray, y_train=None) -> np.ndarray:
        supervised = self.config.umap_supervised and y_train is not None
        logger.info(
            "Fitting %s UMAP: %s -> %d dims (n_neighbors=%d, min_dist=%.2f, "
            "metric=%s%s)",
            "supervised" if supervised else "unsupervised",
            X_train.shape, self.config.n_qubits, self.config.umap_n_neighbors,
            self.config.umap_min_dist, self.config.umap_metric,
            f", target_weight={self.config.umap_target_weight}" if supervised else "",
        )
        reduced = (self.reducer.fit_transform(X_train, y=np.asarray(y_train))
                   if supervised else self.reducer.fit_transform(X_train))
        scaled = self.scaler.fit_transform(reduced)
        self._fitted = True
        return scaled.astype(np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first.")
        reduced = self.reducer.transform(X)
        scaled = self.scaler.transform(reduced)
        return np.clip(scaled, 0.0, 1.0).astype(np.float64)


# --------------------------------------------------------------------------- #
# Step 3: Quantum Feature Map & Kernel
# --------------------------------------------------------------------------- #
def _entangling_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]           # linear
    if entanglement == "circular" and n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    return pairs


class QuantumProcessor:
    """Encodes features with a **bandwidth-controlled** feature map and computes
    a quantum kernel, addressing the exponential concentration of fidelity
    kernels (Shaydulin & Wild 2022; Huang et al. 2021).

    Feature map (per repetition), for bandwidth ``c`` and inputs ``z`` scaled to
    [0, 1]:

        H on every qubit;  P(2 c z_k) on qubit k;
        for each entangling pair (m, n):  CX; P(2 c^2 z_m z_n); CX.

    Note there are **no** ``(pi - z_m)(pi - z_n)`` offsets (the Qiskit
    ``ZZFeatureMap`` default, a notorious concentration amplifier). Small ``c``
    keeps phases from wandering across [0, 2pi), preventing the destructive
    interference that drives the kernel to the identity.

    Kernels:
      * ``kernel_type="fidelity"`` : K = |<phi(x)|phi(y)>|^2 (exact statevector).
      * ``kernel_type="projected"``: Gaussian kernel on single-qubit Bloch
        vectors of |phi(.)> (projected quantum kernel, Huang et al. 2021), which
        sidesteps concentration structurally and is far more shot/noise friendly
        on hardware.
    """

    def __init__(self, config: PipelineConfig, bandwidth: float | None = None):
        self.config = config
        self.n_qubits = config.n_qubits
        self.bandwidth = float(config.encoding_scale if bandwidth is None else bandwidth)
        self._z = ParameterVector("z", self.n_qubits)
        self.feature_map = self._build_feature_map()
        logger.info(
            "Quantum kernel: bandwidth c=%.3g, qubits=%d, reps=%d, entangle='%s', "
            "kernel='%s', depth=%d",
            self.bandwidth, self.n_qubits, config.feature_map_reps,
            config.entanglement, config.kernel_type, self.feature_map.depth(),
        )

    def _build_feature_map(self) -> QuantumCircuit:
        c, z = self.bandwidth, self._z
        qc = QuantumCircuit(self.n_qubits)
        pairs = _entangling_pairs(self.n_qubits, self.config.entanglement)
        for _ in range(self.config.feature_map_reps):
            for k in range(self.n_qubits):
                qc.h(k)
                qc.p(2.0 * c * z[k], k)
            for (m, n) in pairs:
                qc.cx(m, n)
                qc.p(2.0 * c * c * z[m] * z[n], n)
                qc.cx(m, n)
        return qc

    def _statevectors(self, X: np.ndarray) -> np.ndarray:
        """Simulate the feature map once per row; return (N, 2**n_qubits) complex."""
        dim = 2 ** self.n_qubits
        states = np.empty((len(X), dim), dtype=np.complex128)
        for i, x in enumerate(X):
            bound = self.feature_map.assign_parameters(
                {p: float(v) for p, v in zip(self._z, x)}
            )
            states[i] = Statevector.from_instruction(bound).data
        return states

    def _bloch_features(self, states: np.ndarray) -> np.ndarray:
        """Single-qubit Bloch vectors (<X>,<Y>,<Z> per qubit) for each state,
        the classical descriptor behind the projected quantum kernel."""
        n, dim = len(states), states.shape[1]
        idx = np.arange(dim)
        feats = np.empty((n, 3 * self.n_qubits))
        probs = np.abs(states) ** 2
        for k in range(self.n_qubits):
            bit = (idx >> k) & 1                      # qiskit little-endian
            partner = idx ^ (1 << k)
            prod = np.conj(states) * states[:, partner]
            mask0 = bit == 0
            feats[:, 3 * k + 0] = 2.0 * prod[:, mask0].real.sum(axis=1)   # <X>
            feats[:, 3 * k + 1] = 2.0 * prod[:, mask0].imag.sum(axis=1)   # <Y>
            feats[:, 3 * k + 2] = (probs * (1 - 2 * bit)).sum(axis=1)      # <Z>
        return feats

    def kernel(self, X_a: np.ndarray, X_b: np.ndarray | None = None) -> np.ndarray:
        """Quantum kernel between X_a and X_b (or X_a with itself)."""
        n_b = len(X_a) if X_b is None else len(X_b)
        logger.info("Computing %s quantum kernel (%d x %d, c=%.3g) ...",
                    self.config.kernel_type, len(X_a), n_b, self.bandwidth)
        S_a = self._statevectors(X_a)
        S_b = S_a if X_b is None else self._statevectors(X_b)
        if self.config.kernel_type == "projected":
            B_a = self._bloch_features(S_a)
            B_b = B_a if X_b is None else self._bloch_features(S_b)
            # ||rho_k(i)-rho_k(j)||_F^2 = 1/2 ||bloch_i-bloch_j||^2, folded into gamma.
            sq = (
                (B_a ** 2).sum(1)[:, None]
                + (B_b ** 2).sum(1)[None, :]
                - 2.0 * B_a @ B_b.T
            )
            return np.exp(-self.config.projected_gamma * np.clip(sq, 0.0, None))
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

    # Quantum-kernel health
    bandwidth: float = 0.0            # selected encoding bandwidth c
    kta: float = 0.0                  # kernel-target alignment
    offdiag_mean: float = 0.0         # mean off-diagonal kernel entry


# --------------------------------------------------------------------------- #
# Steps 4 & 5: two-stage hybrid pipeline
# --------------------------------------------------------------------------- #
class ToxicityPipeline:
    """End-to-end orchestrator (Steps 1-5). Uses a fit/val/test split so the
    supervised UMAP, encoding bandwidth, SVC C, and decision threshold are all
    chosen on a validation set the embedding never saw (leakage-free), with the
    QSVC benchmarked against classical baselines on the identical features."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.embedder = MoleculeEmbedder(self.config)
        self.compressor = TopologicalCompressor(self.config)
        self.quantum = QuantumProcessor(self.config)
        self.classifier: SVC | None = None

    # ---- kernel cross-validation helpers ---------------------------------- #
    @staticmethod
    def _kta(K, y):
        """Centered kernel-target alignment: how well K's geometry matches the
        labels. KTA = <K, yy^T> / (||K|| ||yy^T||), with y in {-1,+1}."""
        yy = np.outer(y, y).astype(float)
        num = float((K * yy).sum())
        den = float(np.linalg.norm(K) * np.linalg.norm(yy))
        return num / den if den > 0 else 0.0

    def _select_hyperparams(self, X_fit, y_fit, X_val, y_val):
        """Jointly grid-search encoding bandwidth c and SVC C, scoring each on the
        **clean validation set** (embedded without labels). Leakage-free because
        the supervised UMAP only ever saw the fit set. Returns the best config and
        the validation decision scores of the winning (fit-trained) model."""
        cfg = self.config
        c_grid = cfg.tune_bandwidth or (cfg.encoding_scale,)
        C_grid = cfg.tune_C or (cfg.svc_C,)
        y_fit_signed = np.where(y_fit == 1, 1.0, -1.0)
        best = {"auc": -np.inf, "c": c_grid[0], "C": C_grid[0],
                "val_scores": None, "kta": 0.0, "off": 0.0}
        for c in c_grid:
            qp = QuantumProcessor(cfg, bandwidth=c)
            K_ff = np.clip((lambda K: (K + K.T) / 2.0)(qp.kernel(X_fit)), 0.0, 1.0)
            K_vf = qp.kernel(X_val, X_fit)
            off = K_ff[~np.eye(len(K_ff), dtype=bool)]
            kta = self._kta(K_ff, y_fit_signed)
            best_val_here = -np.inf
            for C in C_grid:
                svc = SVC(kernel="precomputed", C=C, class_weight=cfg.class_weight)
                svc.fit(K_ff, y_fit)
                val_scores = svc.decision_function(K_vf)
                try:
                    vauc = roc_auc_score(y_val, val_scores)
                except ValueError:
                    vauc = 0.5
                best_val_here = max(best_val_here, vauc)
                if vauc > best["auc"]:
                    best.update(auc=vauc, c=c, C=C, val_scores=val_scores,
                                kta=kta, off=float(off.mean()))
            logger.info("  c=%-5.3g  KTA=%.3f off-diag mean=%.4g | best val AUC(C)=%.4f",
                        c, kta, off.mean(), best_val_here)
        logger.info("Selected bandwidth c=%.3g, C=%g (val AUC=%.4f)",
                    best["c"], best["C"], best["auc"])
        return best

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

        # ---- stratified train/test split, then a clean validation holdout --- #
        # Supervised UMAP is fit on the FIT split only; VAL and TEST are embedded
        # label-free via transform(), so bandwidth/C/threshold are all chosen on
        # data the embedding never saw (no leakage). The final QSVC is trained on
        # FIT and evaluated on the untouched TEST set.
        idx_train, idx_test = train_test_split(
            np.arange(len(smiles)), test_size=cfg.test_size,
            stratify=labels, random_state=cfg.random_state,
        )
        idx_fit, idx_val = train_test_split(
            idx_train, test_size=cfg.val_size, stratify=labels[idx_train],
            random_state=cfg.random_state,
        )
        order = np.concatenate([idx_fit, idx_val, idx_test])
        n_fit, n_val, n_test = len(idx_fit), len(idx_val), len(idx_test)
        n_train = n_fit + n_val
        smiles_ord = [smiles[i] for i in order]
        y_ord = labels[order]
        y_fit, y_val = y_ord[:n_fit], y_ord[n_fit:n_train]
        y_train, y_test = y_ord[:n_train], y_ord[n_train:]
        logger.info(
            "Working set: %d molecules (fit %d / val %d / test %d), toxic rate "
            "fit=%.2f val=%.2f test=%.2f", len(smiles), n_fit, n_val, n_test,
            y_fit.mean(), y_val.mean(), y_test.mean(),
        )

        # ---- Step 1: ChemBERTa embeddings --------------------------------- #
        logger.info("=== Step 1/5: ChemBERTa semantic extraction ===")
        E_all = self.embedder.embed(smiles_ord)
        E_fit, E_val, E_test = E_all[:n_fit], E_all[n_fit:n_train], E_all[n_train:]

        # ---- Step 2: UMAP compression (supervised, fit on FIT labels only) -- #
        logger.info("=== Step 2/5: UMAP topological compression ===")
        X_fit = self.compressor.fit_transform(E_fit, y_fit)
        X_val = self.compressor.transform(E_val)
        X_test = self.compressor.transform(E_test)
        X_all = np.vstack([X_fit, X_val, X_test])

        # ---- Step 3+4: bandwidth/C selection on VAL, then final kernel ----- #
        logger.info("=== Step 3/5: Quantum kernel + leakage-free selection ===")
        best = self._select_hyperparams(X_fit, y_fit, X_val, y_val)
        best_c, best_C, val_auc = best["c"], best["C"], best["auc"]

        self.quantum = QuantumProcessor(cfg, bandwidth=best_c)
        K_full = self.quantum.kernel(X_all)
        K_full = np.clip((K_full + K_full.T) / 2.0, 0.0, 1.0)  # symmetrise
        K_fit = K_full[:n_fit, :n_fit]                         # fit-vs-fit (training)
        K_test_fit = K_full[n_train:, :n_fit]                  # test-vs-fit

        # Kernel-health diagnostic: the off-diagonal distribution. Healthy kernels
        # spread mass across ~[0.05, 0.9]; a spike at 0 => concentration (near
        # identity), a spike at 1 => the map is too weak to separate anything.
        offdiag = K_full[~np.eye(len(K_full), dtype=bool)]
        pct = np.percentile(offdiag, [5, 25, 50, 75, 95])
        logger.info("Kernel off-diagonal: mean=%.4g pct[5,25,50,75,95]=%s max=%.4g | KTA=%.4f",
                    offdiag.mean(), np.round(pct, 4).tolist(), offdiag.max(), best["kta"])
        if offdiag.mean() < 1e-3:
            logger.warning(
                "Quantum kernel is CONCENTRATED (near-identity): off-diagonal "
                "mean=%.2g. The QSVC will not generalise. Reduce n_qubits (<=10), "
                "feature_map_reps, and/or the bandwidth grid.", offdiag.mean())
        elif offdiag.mean() > 0.98:
            logger.warning(
                "Quantum kernel is near-constant (~all ones): off-diagonal "
                "mean=%.3g. The feature map is too weak; raise the bandwidth.",
                offdiag.mean())

        # ---- Step 4: Stage 1 QSVC (train on FIT, threshold on VAL) --------- #
        logger.info("=== Step 4/5: Stage 1 - supervised QSVC triage ===")
        self.classifier = SVC(kernel="precomputed", C=best_C,
                              class_weight=cfg.class_weight)
        self.classifier.fit(K_fit, y_fit)

        # Threshold: Youden's J on the clean validation scores (model trained on
        # FIT, val embedded label-free). Same model/scale used for test below.
        threshold = 0.0
        if cfg.calibrate_threshold and best["val_scores"] is not None \
                and np.std(best["val_scores"]) > 1e-6:
            fpr, tpr, thr = roc_curve(y_val, best["val_scores"])
            threshold = float(thr[np.argmax(tpr - fpr)])
            logger.info("Calibrated decision threshold: %.4f (Youden's J on validation)", threshold)

        y_score_test = self.classifier.decision_function(K_test_fit)
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
        logger.info("QSVC test: accuracy=%.4f  f1=%.4f  roc_auc=%.4f (val AUC=%.4f)",
                    acc, f1, auc, val_auc)

        # ---- classical baselines (fit-train, val-select, test-report) ------ #
        baseline = None
        if cfg.run_classical_baseline:
            baseline = self._classical_baseline(
                E_fit, y_fit, E_val, y_val, E_test, y_test,
                X_fit=X_fit, X_val=X_val, X_test=X_test)

        # ---- Step 5: Stage 2 mechanism sub-grouping ----------------------- #
        # Triage every molecule with the fit-trained QSVC (columns = fit block).
        logger.info("=== Step 5/5: Stage 2 - spectral mechanism discovery ===")
        y_pred_all = (self.classifier.decision_function(K_full[:, :n_fit]) >= threshold).astype(int)
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
            best_C=best_C, cv_auc_mean=float(val_auc), cv_auc_std=0.0,
            baseline=baseline, toxic_index=toxic_index, cluster_labels=cluster_labels,
            cluster_summary=cluster_summary, order=order, n_train=n_train, full_kernel=K_full,
            bandwidth=float(best_c), kta=float(best["kta"]), offdiag_mean=float(offdiag.mean()),
        )

    def _classical_baseline(self, E_fit, y_fit, E_val, y_val, E_test, y_test,
                            X_fit=None, X_val=None, X_test=None) -> dict:
        """Classical baselines, trained/selected/reported on the SAME fit/val/test
        splits as the QSVC for a fair, leakage-free comparison:

        * ``rbf_umap`` / ``linsvc_umap`` -- classical SVC on the **same** UMAP
          features the quantum kernel sees (apples-to-apples control: quantum ZZ
          kernel vs classical kernel on identical n_qubits-d inputs).
        * ``rbf_384`` / ``logreg_384`` -- on the raw ChemBERTa embeddings, i.e.
          the ceiling you forgo by compressing to n_qubits dimensions.

        The reported ``cv_auc_mean`` is the clean **validation** AUC (comparable
        to the quantum val AUC); ``test_*`` are on the held-out test set.
        """
        cfg = self.config
        out = {}

        def _evaluate(name, model, fit_X, val_X, test_X):
            model.fit(fit_X, y_fit)
            def _score(Z):
                return (model.decision_function(Z) if hasattr(model, "decision_function")
                        else model.predict_proba(Z)[:, 1])
            try:
                val_auc = roc_auc_score(y_val, _score(val_X))
            except ValueError:
                val_auc = 0.5
            score = _score(test_X)
            pred = model.predict(test_X)
            out[name] = {
                "cv_auc_mean": float(val_auc), "cv_auc_std": 0.0,
                "test_auc": float(roc_auc_score(y_test, score)),
                "test_acc": float(accuracy_score(y_test, pred)),
                "test_f1": float(f1_score(y_test, pred, zero_division=0)),
            }
            logger.info("  %-12s val AUC=%.4f | test AUC=%.4f acc=%.4f f1=%.4f",
                        name, out[name]["cv_auc_mean"],
                        out[name]["test_auc"], out[name]["test_acc"], out[name]["test_f1"])

        # Fair, same-features control on the UMAP features (if provided).
        if X_fit is not None and X_val is not None and X_test is not None:
            logger.info("Classical baselines on the same %d-d UMAP features ...", X_fit.shape[1])
            _evaluate("rbf_umap", make_pipeline(
                StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale",
                                      class_weight=cfg.class_weight)), X_fit, X_val, X_test)
            _evaluate("linsvc_umap", make_pipeline(
                StandardScaler(), SVC(kernel="linear", C=1.0,
                                      class_weight=cfg.class_weight)), X_fit, X_val, X_test)

        # Ceiling on the raw ChemBERTa embeddings.
        logger.info("Classical baselines on raw %d-d ChemBERTa embeddings ...", E_fit.shape[1])
        _evaluate("rbf_384", make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale",
                                  class_weight=cfg.class_weight)), E_fit, E_val, E_test)
        _evaluate("logreg_384", make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000,
                                                 class_weight=cfg.class_weight)), E_fit, E_val, E_test)
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
#     https://tdcommons.ai/single_pred_tasks/tox/
#     pip install PyTDC
#     from tdc.single_pred import Tox
#     # hERGCentral -- huge electrophysiology screen (~306,893 drugs), multi-label:
#     Tox(name="herg_central", label_name="hERG_inhib")  # binary blockade (default)
#     Tox(name="herg_central", label_name="hERG_at_1uM")  # % inhibition (regression)
#     Tox(name="hERG")        -> Wang et al. 2016   (~655 drugs)
#     Tox(name="herg_karim")  -> Karim et al. 2021  (~13,445 drugs, if exposed)
# get_data() returns a DataFrame with columns: Drug_ID, Drug (SMILES), Y (label;
# for hERG_inhib: 1 = hERG blocker / cardiotoxic, 0 = non-blocker).
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


def _resolve_tdc_label(dataset_name, label_name):
    """Match a requested TDC ``label_name`` (case-insensitively) against the
    labels the dataset actually exposes. Only multi-label datasets (e.g.
    ``herg_central``, ``tox21``) accept a label_name."""
    if label_name is None:
        return None
    try:
        from tdc.utils import retrieve_label_name_list
        avail = retrieve_label_name_list(dataset_name.lower())
        for lab in avail:
            if lab.lower() == label_name.lower():
                return lab
        logger.warning("Label '%s' not found for '%s'; available: %s. "
                       "Passing it through unchanged.", label_name, dataset_name, avail)
    except Exception:  # noqa: BLE001 - best-effort; let TDC validate
        pass
    return label_name


def _from_tdc(tdc_name, cache_dir, label_name=None):
    """Load an official hERG dataset via Therapeutics Data Commons (PyTDC).

    Handles single-label sets (``herg``, ``herg_karim``) and multi-label sets
    (``herg_central``, which needs a ``label_name`` such as ``hERG_inhib``).
    TDC names are matched case-insensitively; the exact registered name is
    auto-discovered so casing/variants are tolerated.
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
    req = tdc_name.lower().replace("-", "_")
    candidates = [tdc_name, tdc_name.lower()]

    # Auto-discover the exact registered names; prefer ones matching the request.
    valid = None
    try:
        from tdc.utils import retrieve_dataset_names
        valid = retrieve_dataset_names("Tox")
        logger.info("TDC registered Tox datasets: %s", valid)
        exact = [v for v in valid if v.lower().replace("-", "_") == req]
        partial = [v for v in valid if req in v.lower() or v.lower() in req]
        # If the user asked for Karim but this build lacks it, do NOT silently
        # fall back to the tiny Wang set -- surface that clearly instead.
        candidates = exact + partial + candidates
    except Exception:  # noqa: BLE001 - discovery is best-effort
        pass

    last_err = None
    for name in dict.fromkeys(candidates):  # de-dupe, preserve order
        try:
            resolved_label = _resolve_tdc_label(name, label_name)
            kwargs = {"name": name, "path": cache_dir}
            if resolved_label is not None:
                kwargs["label_name"] = resolved_label
            logger.info("Loading official TDC dataset '%s'%s ...", name,
                        f" (label='{resolved_label}')" if resolved_label else "")
            df = Tox(**kwargs).get_data()  # columns: Drug_ID, Drug (SMILES), Y
            smiles = df["Drug"].astype(str).tolist()
            labels = df["Y"].astype(int).to_numpy()
            logger.info("TDC '%s': %d molecules (label positives=%d).",
                        name, len(smiles), int(labels.sum()))
            if "karim" in req and len(smiles) < 1000:
                logger.warning(
                    "Requested Karim (~13,445) but loaded only %d molecules -- this "
                    "PyTDC build does not expose 'herg_karim' by name. Consider "
                    "tdc_name='herg_central' (label 'hERG_inhib') or local_path=.",
                    len(smiles))
            return smiles, labels
        except Exception as err:  # noqa: BLE001 - try the next candidate name
            last_err = err

    raise RuntimeError(
        f"Could not load TDC dataset for '{tdc_name}'. Registered Tox names: "
        f"{valid}. Tried {list(dict.fromkeys(candidates))}. Last error: {last_err}. "
        "You can instead download an hERG CSV and pass "
        "load_herg_dataset(local_path='herg.csv')."
    )


def load_herg_dataset(
    n_samples: int | None = 408,
    balanced: bool = True,
    *,
    source: str = "tdc",
    tdc_name: str = "herg_central",
    tdc_label_name: str | None = "hERG_inhib",
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
        ``tdc_name="herg_central"`` + ``tdc_label_name="hERG_inhib"`` (default) ->
            hERGCentral binary blockade task (~306,893 molecules; label 1 if the
            compound blocks hERG, i.e. hERG_at_10uM < -50).
        ``tdc_name="herg_karim"`` -> Karim et al. 2021 (~13,445; if your PyTDC
            build exposes it); ``tdc_name="hERG"`` -> Wang et al. 2016 (~655).
        The exact registered name is auto-discovered, so casing variants are OK.
    tdc_label_name : str or None
        For multi-label TDC sets (hERGCentral: ``hERG_at_1uM``, ``hERG_at_10uM``,
        ``hERG_inhib``). Ignored for single-label sets.

    Parameters
    ----------
    n_samples : int or None
        If given, return a (class-balanced if ``balanced``) random subsample of
        this size -- appropriate for the O(N^2) quantum kernel. ``None`` returns
        the full dataset. (hERGCentral is huge, so keep this modest.)

    Returns (smiles, labels, names).
    """
    if local_path is not None:
        smiles, labels = _from_local(local_path, smiles_col, label_col)
    elif source == "tdc":
        smiles, labels = _from_tdc(tdc_name, cache_dir, label_name=tdc_label_name)
    else:
        raise ValueError(f"Unknown source '{source}'. Use source='tdc' or pass local_path=...")

    logger.info("Loaded hERG dataset: %d molecules (blockers=%d, non-blockers=%d)",
                len(smiles), int(labels.sum()), int((labels == 0).sum()))

    # Subsample FIRST (hERGCentral has ~307k rows -- validating them all would be
    # wastefully slow), then validate the (small) working set.
    if n_samples is not None and n_samples < len(smiles):
        idx = (_balanced_subsample(labels, n_samples, random_state) if balanced
               else np.random.default_rng(random_state).choice(len(smiles), n_samples, replace=False))
        smiles = [smiles[i] for i in idx]
        labels = labels[idx]
        logger.info("Subsampled to %d molecules (balanced=%s).", len(smiles), balanced)

    if validate:
        smiles, labels = _filter_valid_smiles(smiles, labels)
        logger.info("After SMILES validation: %d molecules (blockers=%d).",
                    len(smiles), int(labels.sum()))

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
    print(f"Bandwidth c       : {result.bandwidth:g}  (KTA={result.kta:.4f}, "
          f"kernel off-diag mean={result.offdiag_mean:.4g})")
    print(f"Validation ROC-AUC: {result.cv_auc_mean:.4f}  (leakage-free holdout; used for selection)")
    print(f"Decision threshold: {result.threshold:.4f}")
    print(f"Test Accuracy     : {result.accuracy:.4f}")
    print(f"Test F1-Score     : {result.f1:.4f}")
    print(f"Test ROC-AUC      : {result.roc_auc:.4f}")

    if result.baseline:
        print("\n" + cfg_line)
        print("QUANTUM vs CLASSICAL  (*_umap = same UMAP features as quantum = fair")
        print("                       control; *_384 = ceiling on raw ChemBERTa embeddings)")
        print(cfg_line)
        print(f"{'model':13s} {'val AUC':>10s} {'test AUC':>10s} {'test acc':>10s} {'test f1':>9s}")
        print(f"{'quantum':13s} {result.cv_auc_mean:>10.4f} "
              f"{result.roc_auc:>10.4f} {result.accuracy:>10.4f} {result.f1:>9.4f}")
        for name, m in result.baseline.items():
            print(f"{name:13s} {m['cv_auc_mean']:>10.4f} "
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
    config = PipelineConfig(n_qubits=8, n_clusters=3, n_samples=408)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    logger.info("Loading official hERGCentral dataset (label 'hERG_inhib') ...")
    smiles, labels, names = load_herg_dataset(
        source="tdc", tdc_name="herg_central", tdc_label_name="hERG_inhib",
        n_samples=config.n_samples, random_state=config.random_state,
    )

    result = ToxicityPipeline(config).run(smiles, labels, names)
    print_report(result)


if __name__ == "__main__":
    main()
