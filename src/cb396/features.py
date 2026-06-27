"""Construção de features: janela deslizante sobre a PSSM."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_TO_ID, WINDOW_SIZE, Paths
from .pssm import parse_ascii_pssm


def make_sliding_windows(
    pssm_matrix: np.ndarray, q3: str, window_size: int = WINDOW_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Gera ``X[L, window*20]`` e ``y[L]`` a partir da PSSM e dos rótulos Q3."""
    if window_size % 2 == 0:
        raise ValueError("window_size deve ser ímpar.")
    L = pssm_matrix.shape[0]
    half = window_size // 2
    if len(q3) != L:
        raise ValueError(f"q3 tem tamanho {len(q3)}, mas PSSM tem {L}")

    padded = np.pad(
        pssm_matrix, ((half, half), (0, 0)), mode="constant", constant_values=0
    )
    X = np.empty((L, window_size * pssm_matrix.shape[1]), dtype=np.float32)
    y = np.empty(L, dtype=np.int64)
    for i in range(L):
        X[i] = padded[i : i + window_size, :].reshape(-1)
        y[i] = LABEL_TO_ID[q3[i]]
    return X, y


def build_feature_matrix(
    df_valid: pd.DataFrame, paths: Paths, window_size: int = WINDOW_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Monta ``X``, ``y`` e ``groups`` a partir das PSSMs de cada proteína.

    Retorna ``(X, y, groups, df_feature_errors)``.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    groups: list[str] = []
    errors: list[dict] = []

    for row in df_valid.itertuples(index=False):
        pssm_path = paths.pssm / f"{row.id}.pssm"
        if not pssm_path.exists():
            errors.append({"id": row.id, "error": "missing_pssm"})
            continue
        try:
            seq, pssm = parse_ascii_pssm(pssm_path)
            if len(seq) != len(row.sequence):
                errors.append(
                    {
                        "id": row.id,
                        "error": "pssm_sequence_length_mismatch",
                        "pssm_len": len(seq),
                        "sequence_len": len(row.sequence),
                    }
                )
                continue
            X_i, y_i = make_sliding_windows(pssm, row.q3, window_size)
            X_parts.append(X_i)
            y_parts.append(y_i)
            groups.extend([row.id] * len(y_i))
        except Exception as exc:  # noqa: BLE001 - registramos e seguimos
            errors.append({"id": row.id, "error": str(exc)})

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y, np.asarray(groups), pd.DataFrame(errors)


def save_features(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, errors: pd.DataFrame, paths: Paths
) -> None:
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    np.save(paths.artifacts / "X_pssm_window13.npy", X)
    np.save(paths.artifacts / "y_q3.npy", y)
    np.save(paths.artifacts / "groups.npy", groups)
    errors.to_csv(paths.artifacts / "feature_errors.csv", index=False)


def load_features(paths: Paths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrega ``X``, ``y`` e ``groups`` já salvos (treino sem Colab/BLAST)."""
    X = np.load(paths.artifacts / "X_pssm_window13.npy")
    y = np.load(paths.artifacts / "y_q3.npy")
    groups = np.load(paths.artifacts / "groups.npy", allow_pickle=True)
    return X, y, groups
