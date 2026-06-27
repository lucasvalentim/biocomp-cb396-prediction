"""Modelos SVM, validação cruzada por proteína e treino do modelo final."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .config import ID_TO_LABEL, N_SPLITS, RANDOM_STATE, Paths


def make_linear_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_rbf_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    cache_size=1024,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _predict_in_batches(model, X_test: np.ndarray, batch: int = 1000) -> np.ndarray:
    """Predict em lotes — evita picos de memória do RBF (ver docs/07)."""
    return np.concatenate(
        [model.predict(X_test[i : i + batch]) for i in range(0, len(X_test), batch)]
    )


def evaluate_group_cv(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    max_train_residues: int | None = None,
    batched_predict: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Avalia um modelo com ``GroupKFold`` (split por proteína).

    ``model_factory`` é uma função sem argumentos que devolve um pipeline novo.
    ``max_train_residues`` amostra o treino (usado no RBF).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    gkf = GroupKFold(n_splits=n_splits)
    all_true, all_pred, fold_rows = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        if max_train_residues and len(train_idx) > max_train_residues:
            train_idx = rng.choice(train_idx, size=max_train_residues, replace=False)

        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = (
            _predict_in_batches(model, X[test_idx])
            if batched_predict
            else model.predict(X[test_idx])
        )
        q3 = accuracy_score(y[test_idx], y_pred)
        if verbose:
            print(
                f"Fold {fold}: Q3={q3:.4f} | train={len(train_idx)} "
                f"test={len(test_idx)} test_prot={len(np.unique(groups[test_idx]))}",
                flush=True,
            )
        all_true.append(y[test_idx])
        all_pred.append(y_pred)
        fold_rows.append(
            {
                "fold": fold,
                "q3": q3,
                "train_residues": len(train_idx),
                "test_residues": len(test_idx),
                "test_proteins": int(len(np.unique(groups[test_idx]))),
            }
        )

    return np.concatenate(all_true), np.concatenate(all_pred), pd.DataFrame(fold_rows)


def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, pd.DataFrame]:
    """Retorna o classification report (str) e a matriz de confusão (DataFrame)."""
    report = classification_report(
        y_true, y_pred, target_names=["H", "E", "C"], digits=4
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    df_cm = pd.DataFrame(
        cm,
        index=["true_H", "true_E", "true_C"],
        columns=["pred_H", "pred_E", "pred_C"],
    )
    return report, df_cm


def save_evaluation(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_folds: pd.DataFrame,
    paths: Paths,
) -> None:
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(paths.artifacts / f"fold_metrics_{name}.csv", index=False)
    pd.DataFrame(
        {
            "y_true_id": y_true,
            "y_pred_id": y_pred,
            "y_true": [ID_TO_LABEL[i] for i in y_true],
            "y_pred": [ID_TO_LABEL[i] for i in y_pred],
        }
    ).to_csv(paths.artifacts / f"predictions_{name}.csv", index=False)
    _, df_cm = metrics_report(y_true, y_pred)
    df_cm.to_csv(paths.artifacts / f"confusion_matrix_{name}.csv")


def train_final_model(X: np.ndarray, y: np.ndarray, paths: Paths):
    """Treina o Linear SVM em todos os dados e salva o ``.joblib``."""
    import joblib

    model = make_linear_svm()
    model.fit(X, y)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    out = paths.artifacts / "final_linear_svm_pssm_window13.joblib"
    joblib.dump(model, out)
    return model, out
