"""
Hybrid Classical-Quantum Pipeline for Drug Cardiotoxicity Prediction
====================================================================

A two-stage hybrid **Classical-Quantum** pipeline that triages molecules for
cardiotoxicity and then discovers structural toxicity mechanisms:

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
       |                               + FidelityQuantumKernel (local simulator)
       v
    [Step 4] Stage 1: Supervised       SVC(kernel='precomputed')  ==>  QSVC triage
       |          Binary label: 0 = Safe, 1 = Toxic (cardiotoxic)
       v
    [Step 5] Stage 2: Unsupervised     SpectralClustering(affinity='precomputed')
                                       over the quantum-kernel sub-matrix of the
                                       molecules the QSVC flags as Toxic, grouping
                                       them by quantum structural similarity into
                                       candidate toxicity-mechanism clusters.

This is a **real** implementation: it loads the pretrained ChemBERTa chemical
language model from the Hugging Face Hub (no random-initialised stand-in) and
operates on a curated panel of **real** marketed drugs whose cardiotoxicity
labels are grounded in pharmacology (hERG / I_Kr blockers, QT-prolonging and
torsadogenic agents, and drugs withdrawn for cardiovascular risk, versus drugs
with no significant clinical cardiotoxicity).

Requirements
------------
    Python 3.10+
    pip install qiskit qiskit-machine-learning qiskit-aer scikit-learn \
                umap-learn transformers torch
    # optional, for SMILES validation / canonicalisation:
    pip install rdkit

Network note: Step 1 downloads the ChemBERTa weights from the Hugging Face
Hub on first run (or reads them from the local HF cache). The host must be
able to reach ``huggingface.co`` (or have the model pre-cached / mirrored via
the ``HF_HOME`` / ``HF_ENDPOINT`` environment variables).

Author: CardiogenAI
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
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

import umap

from qiskit_machine_learning.kernels import FidelityQuantumKernel

# --- Qiskit version-compatibility shim ------------------------------------- #
# Qiskit 2.x replaced the V1 ``Sampler`` primitive with ``StatevectorSampler``
# and deprecated the ``ZZFeatureMap`` class in favour of the ``zz_feature_map``
# function. Prefer the modern API and fall back for Qiskit 1.x installs.
try:  # Qiskit >= 1.3
    from qiskit.circuit.library import zz_feature_map as _zz_feature_map
except ImportError:  # Qiskit < 1.3
    from qiskit.circuit.library import ZZFeatureMap as _zz_feature_map

try:  # Qiskit 2.x + fidelity shipped with qiskit-machine-learning
    from qiskit.primitives import StatevectorSampler as _Sampler
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
except ImportError:  # Qiskit 1.x + qiskit-algorithms fidelity
    from qiskit.primitives import Sampler as _Sampler
    from qiskit_algorithms.state_fidelities import ComputeUncompute

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
    embedding_batch_size: int = 16
    pooling: str = "mean"  # "mean" (masked mean of last hidden state), "cls", or "pooler"

    # --- Step 2: UMAP compression ---
    n_qubits: int = 8               # UMAP output dim == number of qubits (8 or 16)
    umap_n_neighbors: int = 5
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    # --- Step 3: Quantum feature map / kernel ---
    feature_map_reps: int = 2
    entanglement: str = "linear"

    # --- Step 4: QSVC triage ---
    svc_C: float = 1.0

    # --- Step 5: Spectral clustering (mechanism discovery) ---
    n_clusters: int = 3

    # --- Split / reproducibility ---
    test_size: float = 0.3
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Step 1: Semantic Extraction (ChemBERTa LLM)
# --------------------------------------------------------------------------- #
class MoleculeEmbedder:
    """Wraps the pretrained ChemBERTa chemical language model to turn SMILES
    strings into dense molecular embeddings.

    The model (``DeepChem/ChemBERTa-77M-MTR`` by default) is a RoBERTa-style
    transformer pretrained on ~77M molecules. Each SMILES is tokenised and
    encoded; a fixed-size embedding is produced from the final hidden layer.

    Pooling strategies (config.pooling):
        * "mean"   — attention-masked mean of the final hidden states
                     (recommended: deterministic and uses only pretrained
                     weights).
        * "cls"    — final hidden state of the leading <s>/[CLS] token.
        * "pooler" — the model's ``pooler_output`` head (falls back to masked
                     mean pooling if the checkpoint ships no pooler weights).

    Output: dense numpy array of shape (N_samples, Hidden_Dim) — 384 for
    ChemBERTa-77M-MTR.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None  # loaded lazily on first embed() (or via load())

    def load(self) -> "MoleculeEmbedder":
        """Load the pretrained ChemBERTa tokenizer and encoder from the HF Hub.

        Raises a clear, actionable error if the weights cannot be obtained —
        there is deliberately no random-initialised fallback, so embeddings
        always reflect genuine pretrained chemistry.
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

        # Default: attention-masked mean of the final hidden states.
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


# --------------------------------------------------------------------------- #
# Step 2: Topological Compression (UMAP)
# --------------------------------------------------------------------------- #
class TopologicalCompressor:
    """Compresses ChemBERTa embeddings down to `n_qubits` dimensions with UMAP,
    then rescales each feature into [0, pi] so it is a valid rotation angle for
    the quantum feature map.
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
        """Fit UMAP + scaler on training embeddings; return compressed data."""
        logger.info(
            "Fitting UMAP: %s -> %d dims (n_neighbors=%d, min_dist=%.2f, metric=%s)",
            X_train.shape,
            self.config.n_qubits,
            self.config.umap_n_neighbors,
            self.config.umap_min_dist,
            self.config.umap_metric,
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
        # Test points can land slightly outside the fitted train range.
        return np.clip(scaled, 0.0, np.pi).astype(np.float64)


# --------------------------------------------------------------------------- #
# Step 3: Quantum Feature Map & Kernel
# --------------------------------------------------------------------------- #
class QuantumProcessor:
    """Builds the ZZFeatureMap and computes fidelity quantum-kernel matrices on
    a local simulator through Qiskit's Sampler primitive.

        K(x, y) = |<phi(x)|phi(y)>|^2

    with |phi(.)> prepared by the ZZFeatureMap. All train-to-train and
    test-to-train kernels used downstream are sub-blocks of a single computed
    kernel matrix.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_map = _zz_feature_map(
            feature_dimension=config.n_qubits,
            reps=config.feature_map_reps,
            entanglement=config.entanglement,
        )
        sampler = _Sampler()  # local statevector simulator primitive
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

    def full_kernel(self, X: np.ndarray) -> np.ndarray:
        """Symmetric quantum-kernel matrix over all rows of X, shape (N, N)."""
        logger.info("Computing quantum kernel matrix (%d x %d) ...", len(X), len(X))
        return self.kernel.evaluate(x_vec=X)

    def cross_kernel(self, X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
        """Rectangular quantum-kernel matrix between X_a and X_b, shape (|a|,|b|)."""
        logger.info("Computing quantum kernel matrix (%d x %d) ...", len(X_a), len(X_b))
        return self.kernel.evaluate(x_vec=X_a, y_vec=X_b)


# --------------------------------------------------------------------------- #
# Steps 4 & 5: two-stage hybrid pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    """Everything the pipeline produces."""

    # Stage 1 (supervised triage, evaluated on the held-out test set)
    y_pred_test: np.ndarray
    accuracy: float
    f1: float
    roc_auc: float
    report: str

    # Stage 2 (unsupervised mechanism discovery over all predicted-toxic molecules)
    toxic_index: np.ndarray            # indices (into the full ordered dataset)
    cluster_labels: np.ndarray | None  # sub-cluster id per predicted-toxic molecule

    # Bookkeeping
    order: np.ndarray                  # dataset ordering used internally ([train; test])
    n_train: int
    full_kernel: np.ndarray = field(repr=False, default=None)


class ToxicityPipeline:
    """End-to-end orchestrator:

    SMILES -> ChemBERTa embedding -> UMAP -> quantum kernel
           -> Stage 1: QSVC triage (supervised, Safe/Toxic)
           -> Stage 2: SpectralClustering over predicted-toxic molecules
              (unsupervised toxicity-mechanism discovery).
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.embedder = MoleculeEmbedder(self.config)
        self.compressor = TopologicalCompressor(self.config)
        self.quantum = QuantumProcessor(self.config)
        self.classifier = SVC(
            kernel="precomputed",
            C=self.config.svc_C,
            random_state=self.config.random_state,
        )

    def run(
        self,
        smiles: list[str],
        labels: np.ndarray,
        names: list[str] | None = None,
    ) -> PipelineResult:
        """Execute the full five-step pipeline on a labelled SMILES panel."""
        cfg = self.config
        labels = np.asarray(labels, dtype=int)
        names = names or [f"mol_{i}" for i in range(len(smiles))]

        # ---- stratified train/test split (index-based so we can track names) #
        idx_train, idx_test = train_test_split(
            np.arange(len(smiles)),
            test_size=cfg.test_size,
            stratify=labels,
            random_state=cfg.random_state,
        )
        # Fixed internal ordering: [train ... ; test ...]
        order = np.concatenate([idx_train, idx_test])
        n_train = len(idx_train)
        smiles_ord = [smiles[i] for i in order]
        y_ord = labels[order]
        y_train, y_test = y_ord[:n_train], y_ord[n_train:]

        logger.info(
            "Dataset: %d molecules (%d train / %d test), toxic rate "
            "train=%.2f test=%.2f",
            len(smiles),
            n_train,
            len(idx_test),
            y_train.mean(),
            y_test.mean(),
        )

        # ---- Step 1: ChemBERTa semantic extraction ------------------------ #
        logger.info("=== Step 1/5: ChemBERTa semantic extraction ===")
        E_all = self.embedder.embed(smiles_ord)
        E_train, E_test = E_all[:n_train], E_all[n_train:]

        # ---- Step 2: UMAP topological compression ------------------------- #
        logger.info("=== Step 2/5: UMAP topological compression ===")
        X_train = self.compressor.fit_transform(E_train)
        X_test = self.compressor.transform(E_test)
        X_all = np.vstack([X_train, X_test])

        # ---- Step 3: quantum kernel --------------------------------------- #
        # One full (N x N) fidelity kernel; the train-to-train and
        # test-to-train matrices the QSVC needs are sub-blocks of it.
        logger.info("=== Step 3/5: Quantum kernel generation ===")
        K_full = self.quantum.full_kernel(X_all)
        K_train = K_full[:n_train, :n_train]                 # train-to-train
        K_test = K_full[n_train:, :n_train]                  # test-to-train

        # ---- Step 4: Stage 1 supervised QSVC triage ----------------------- #
        logger.info("=== Step 4/5: Stage 1 - supervised QSVC triage ===")
        self.classifier.fit(K_train, y_train)
        y_pred_test = self.classifier.predict(K_test)
        y_score_test = self.classifier.decision_function(K_test)

        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_score_test)
        except ValueError:
            auc = float("nan")  # single-class test split
        report = classification_report(
            y_test,
            y_pred_test,
            target_names=["Safe (0)", "Toxic (1)"],
            zero_division=0,
        )
        logger.info(
            "QSVC test metrics - accuracy=%.4f  f1=%.4f  roc_auc=%.4f",
            acc,
            f1,
            auc,
        )

        # ---- Step 5: Stage 2 unsupervised mechanism discovery ------------- #
        # Triage every molecule with the trained QSVC, then cluster all
        # predicted-toxic molecules using the corresponding sub-matrix of the
        # computed quantum kernel as a precomputed affinity.
        logger.info("=== Step 5/5: Stage 2 - spectral mechanism discovery ===")
        y_pred_all = self.classifier.predict(K_full[:, :n_train])
        toxic_local = np.flatnonzero(y_pred_all == 1)          # positions in `order`
        toxic_index = order[toxic_local]                       # original dataset ids
        cluster_labels: np.ndarray | None = None

        if len(toxic_local) >= cfg.n_clusters:
            affinity = K_full[np.ix_(toxic_local, toxic_local)]
            # Symmetrise + clip: fidelity estimates carry tiny numerical noise
            # and SpectralClustering requires a symmetric non-negative affinity.
            affinity = np.clip((affinity + affinity.T) / 2.0, 0.0, 1.0)
            spectral = SpectralClustering(
                n_clusters=cfg.n_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=cfg.random_state,
            )
            cluster_labels = spectral.fit_predict(affinity)
            logger.info(
                "Discovered %d mechanism clusters among %d predicted-toxic "
                "molecules: sizes=%s",
                cfg.n_clusters,
                len(toxic_local),
                np.bincount(cluster_labels, minlength=cfg.n_clusters).tolist(),
            )
        else:
            logger.warning(
                "Only %d predicted-toxic molecules (< n_clusters=%d): "
                "skipping spectral clustering.",
                len(toxic_local),
                cfg.n_clusters,
            )

        return PipelineResult(
            y_pred_test=y_pred_test,
            accuracy=acc,
            f1=f1,
            roc_auc=auc,
            report=report,
            toxic_index=toxic_index,
            cluster_labels=cluster_labels,
            order=order,
            n_train=n_train,
            full_kernel=K_full,
        )


# --------------------------------------------------------------------------- #
# Real curated cardiotoxicity dataset
# --------------------------------------------------------------------------- #
def load_cardiotoxicity_dataset() -> tuple[list[str], np.ndarray, list[str]]:
    """A curated panel of real marketed drugs with RDKit-canonical SMILES.

    Labels are grounded in cardiotoxicity pharmacology:
        1 = Toxic  — hERG / I_Kr blockers, QT-prolonging / torsadogenic agents,
                     or drugs withdrawn / restricted for cardiovascular risk.
        0 = Safe   — drugs with no significant clinical cardiotoxicity signal.

    Returns (smiles, labels, names).
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
    for name, smi in toxic.items():
        names.append(name); smiles.append(smi); labels.append(1)
    for name, smi in safe.items():
        names.append(name); smiles.append(smi); labels.append(0)
    return smiles, np.array(labels, dtype=int), names


# --------------------------------------------------------------------------- #
# Demo entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    config = PipelineConfig(n_qubits=8, n_clusters=3)

    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    logger.info("Loading real cardiotoxicity drug panel ...")
    smiles, labels, names = load_cardiotoxicity_dataset()
    logger.info(
        "Loaded %d drugs (%d toxic / %d safe).",
        len(smiles),
        int(labels.sum()),
        int((labels == 0).sum()),
    )

    pipeline = ToxicityPipeline(config)
    result = pipeline.run(smiles, labels, names)

    # ---- Stage 1 report --------------------------------------------------- #
    print("\n" + "=" * 72)
    print("STAGE 1 - QSVC CARDIOTOXICITY TRIAGE (quantum precomputed kernel)")
    print("=" * 72)
    print(result.report)
    print(f"Accuracy : {result.accuracy:.4f}")
    print(f"F1-Score : {result.f1:.4f}")
    print(f"ROC-AUC  : {result.roc_auc:.4f}")

    test_ids = result.order[result.n_train:]
    print("\nPer-molecule test-set triage:")
    for local_pos, dataset_id in enumerate(test_ids):
        pred = result.y_pred_test[local_pos]
        truth = labels[dataset_id]
        flag = "OK " if pred == truth else "XX "
        print(
            f"  {flag}{names[dataset_id]:16s} "
            f"pred={'Toxic' if pred else 'Safe ':5s} "
            f"true={'Toxic' if truth else 'Safe'}"
        )

    # ---- Stage 2 report --------------------------------------------------- #
    print("\n" + "=" * 72)
    print("STAGE 2 - QUANTUM SPECTRAL TOXICITY-MECHANISM CLUSTERS")
    print("=" * 72)
    if result.cluster_labels is None:
        print("Not enough predicted-toxic molecules to form mechanism clusters.")
    else:
        by_cluster: dict[int, list[str]] = {}
        for dataset_id, cluster in zip(result.toxic_index, result.cluster_labels):
            by_cluster.setdefault(int(cluster), []).append(names[dataset_id])
        for cluster in sorted(by_cluster):
            members = ", ".join(by_cluster[cluster])
            print(f"  Mechanism cluster {cluster}: {members}")
        counts = np.bincount(result.cluster_labels, minlength=config.n_clusters)
        print(f"\nCluster sizes: {counts.tolist()}")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
