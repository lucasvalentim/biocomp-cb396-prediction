"""CLI do pipeline CB396. Use ``cb396 <comando> --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from . import blastdb, dataset, features, pssm, train
from .config import (
    BLAST_DB_NAME,
    PSIBLAST_EVALUE,
    PSIBLAST_ITERATIONS,
    WINDOW_SIZE,
    LABEL_TO_ID,
    Paths,
)


def _load_valid(paths: Paths) -> pd.DataFrame:
    csv = paths.processed / "cb396_valid_392.csv"
    if not csv.exists():
        sys.exit(f"Dataset não encontrado em {csv}. Rode `cb396 dataset` primeiro.")
    return pd.read_csv(csv)


# ---- comandos ----

def cmd_dataset(paths: Paths, _args) -> None:
    paths.mkdirs()
    files = dataset.download_cb396(paths)
    df_valid, df_skipped = dataset.build_dataset(files)
    dataset.save_dataset(df_valid, df_skipped, paths)
    print(f"Arquivos originais : {len(files)}")
    print(f"Proteínas válidas  : {len(df_valid)}")
    print(f"Proteínas descartadas: {len(df_skipped)}")
    print(f"Resíduos válidos   : {int(df_valid['length'].sum())}")
    print(f"Salvo em           : {paths.processed}")


def cmd_blastdb(paths: Paths, _args) -> None:
    paths.mkdirs()
    blastdb.download_swissprot(paths)
    print(blastdb.db_info(paths))


def cmd_psiblast(paths: Paths, args) -> None:
    paths.mkdirs()
    df_valid = _load_valid(paths)
    logs = [
        pssm.run_psiblast(pid, paths, num_threads=args.threads)
        for pid in tqdm(df_valid["id"].tolist())
    ]
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(paths.artifacts / "psiblast_logs.csv", index=False)
    print(df_logs["status"].value_counts().to_string())


def cmd_features(paths: Paths, _args) -> None:
    paths.mkdirs()
    df_valid = _load_valid(paths)
    X, y, groups, errors = features.build_feature_matrix(df_valid, paths)
    features.save_features(X, y, groups, errors, paths)
    print(f"X: {X.shape} | y: {y.shape} | groups: {groups.shape}")
    print(f"Proteínas usadas: {len(set(groups.tolist()))} | erros: {len(errors)}")


def cmd_evaluate(paths: Paths, args) -> None:
    paths.mkdirs()
    X, y, groups = features.load_features(paths)
    if args.model == "rbf":
        factory, name = train.make_rbf_svm, "rbf_svm"
        y_true, y_pred, df_folds = train.evaluate_group_cv(
            factory, X, y, groups,
            max_train_residues=args.max_train_residues, batched_predict=True,
        )
    else:
        factory, name = train.make_linear_svm, "linear_svm"
        y_true, y_pred, df_folds = train.evaluate_group_cv(factory, X, y, groups)

    from sklearn.metrics import accuracy_score

    report, df_cm = train.metrics_report(y_true, y_pred)
    train.save_evaluation(name, y_true, y_pred, df_folds, paths)
    print(f"\nQ3 geral: {accuracy_score(y_true, y_pred):.4f}\n")
    print(report)
    print(df_cm.to_string())


def cmd_train(paths: Paths, _args) -> None:
    paths.mkdirs()
    X, y, _ = features.load_features(paths)
    _, out = train.train_final_model(X, y, paths)
    print(f"Modelo final salvo em: {out}")


def _import_cnn():
    try:
        from . import cnn
    except ImportError:
        sys.exit("PyTorch não instalado. Rode: uv sync --extra cnn")
    return cnn


def _load_cnn_items(cnn, paths: Paths, normalize: bool):
    if any(paths.pssm.glob("*.pssm")):
        items = cnn.load_protein_tensors(_load_valid(paths), paths, normalize=normalize)
        print(f"Proteínas (PSSMs brutas em data/pssm/): {len(items)}")
    else:
        items = cnn.load_protein_tensors_from_features(paths, normalize=normalize)
        print(f"Proteínas (reconstruídas de X/y/groups): {len(items)}")
    return items


def _run_cnn(paths, args, make_model, cfg, name, label, out_file):
    """CV + modelo final genérico para os modelos baseados em CNN."""
    from sklearn.metrics import accuracy_score

    cnn = _import_cnn()
    paths.mkdirs()
    items = _load_cnn_items(cnn, paths, normalize=not args.no_normalize)

    if not args.skip_cv:
        y_true, y_pred, df_folds = cnn.cross_validate(
            items, make_model,
            n_splits=args.n_splits, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )
        report, df_cm = train.metrics_report(y_true, y_pred)
        train.save_evaluation(name, y_true, y_pred, df_folds, paths)
        print(f"\nQ3 geral ({label}): {accuracy_score(y_true, y_pred):.4f}\n")
        print(report)
        print(df_cm.to_string())

    if not args.no_final:
        out = cnn.train_final(
            items, make_model, cfg, paths, out_file,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )
        print(f"\nModelo {label} final salvo em: {out}")


def cmd_train_cnn(paths: Paths, args) -> None:
    cnn = _import_cnn()
    cfg = cnn.CNNConfig(
        hidden=args.hidden, depth=args.depth, kernel=args.kernel, dropout=args.dropout
    )
    _run_cnn(paths, args, lambda: cnn.SSCNN(cfg), cfg, "cnn", "CNN", "cnn_pssm.pt")


def cmd_train_cnn_bilstm(paths: Paths, args) -> None:
    cnn = _import_cnn()
    cfg = cnn.CNNBiLSTMConfig(
        hidden=args.hidden, depth=args.depth, kernel=args.kernel, dropout=args.dropout,
        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
    )
    _run_cnn(
        paths, args, lambda: cnn.SSCNNBiLSTM(cfg), cfg,
        "cnn_bilstm", "CNN+BiLSTM", "cnn_bilstm_pssm.pt",
    )


def cmd_train_cnn_bilstm_crf(paths: Paths, args) -> None:
    cnn = _import_cnn()
    cfg = cnn.CNNBiLSTMConfig(
        hidden=args.hidden, depth=args.depth, kernel=args.kernel, dropout=args.dropout,
        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
    )
    _run_cnn(
        paths, args, lambda: cnn.SSCNNBiLSTMCRF(cfg), cfg,
        "cnn_bilstm_crf", "CNN+BiLSTM+CRF", "cnn_bilstm_crf_pssm.pt",
    )


def cmd_summary(paths: Paths, _args) -> None:
    df_valid = _load_valid(paths)
    df_skipped = pd.read_csv(paths.processed / "cb396_skipped_4.csv")
    X, _, _ = features.load_features(paths)
    summary = {
        "valid_proteins": len(df_valid),
        "skipped_proteins": len(df_skipped),
        "total_residues": int(df_valid["length"].sum()),
        "residues_with_pssm": int(X.shape[0]),
        "window_size": WINDOW_SIZE,
        "features_per_residue": WINDOW_SIZE * 20,
        "blast_db": BLAST_DB_NAME,
        "psiblast_iterations": PSIBLAST_ITERATIONS,
        "psiblast_evalue": PSIBLAST_EVALUE,
        "labels": LABEL_TO_ID,
    }
    out = paths.artifacts / "experiment_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_all(paths: Paths, args) -> None:
    cmd_dataset(paths, args)
    cmd_blastdb(paths, args)
    cmd_psiblast(paths, args)
    cmd_features(paths, args)
    cmd_evaluate(paths, args)
    cmd_train(paths, args)
    cmd_summary(paths, args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cb396", description=__doc__)
    p.add_argument(
        "--data-dir", default="data", help="diretório base dos dados (padrão: data)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("dataset", help="baixar + parsear + salvar CB396").set_defaults(func=cmd_dataset)
    sub.add_parser("blastdb", help="baixar Swiss-Prot pré-formatado").set_defaults(func=cmd_blastdb)

    sp = sub.add_parser("psiblast", help="rodar PSI-BLAST em todas as proteínas")
    sp.add_argument("--threads", type=int, default=2)
    sp.set_defaults(func=cmd_psiblast)

    sub.add_parser("features", help="construir X/y/groups").set_defaults(func=cmd_features)

    ev = sub.add_parser("evaluate", help="validação cruzada por proteína")
    ev.add_argument("--model", choices=["linear", "rbf"], default="linear")
    ev.add_argument("--max-train-residues", type=int, default=30000, dest="max_train_residues")
    ev.set_defaults(func=cmd_evaluate)

    sub.add_parser("train", help="treinar modelo final (Linear SVM)").set_defaults(func=cmd_train)

    def _add_cnn_args(parser, depth_default):
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--hidden", type=int, default=64)
        parser.add_argument("--depth", type=int, default=depth_default)
        parser.add_argument("--kernel", type=int, default=11)
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
        parser.add_argument("--n-splits", type=int, default=7, dest="n_splits")
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--no-normalize", action="store_true", help="não aplicar sigmoid na PSSM")
        parser.add_argument("--skip-cv", action="store_true", help="pular validação cruzada")
        parser.add_argument("--no-final", action="store_true", help="não treinar/salvar o modelo final")

    cn = sub.add_parser("train-cnn", help="CNN 1D sobre as PSSMs (precisa do extra cnn)")
    _add_cnn_args(cn, depth_default=3)
    cn.set_defaults(func=cmd_train_cnn)

    cb = sub.add_parser("train-cnn-bilstm", help="CNN + BiLSTM sobre as PSSMs (extra cnn)")
    _add_cnn_args(cb, depth_default=2)
    cb.add_argument("--lstm-hidden", type=int, default=128, dest="lstm_hidden")
    cb.add_argument("--lstm-layers", type=int, default=1, dest="lstm_layers")
    cb.set_defaults(func=cmd_train_cnn_bilstm)

    cc = sub.add_parser("train-cnn-bilstm-crf", help="CNN + BiLSTM + CRF (extra cnn)")
    _add_cnn_args(cc, depth_default=2)
    cc.add_argument("--lstm-hidden", type=int, default=128, dest="lstm_hidden")
    cc.add_argument("--lstm-layers", type=int, default=1, dest="lstm_layers")
    cc.set_defaults(func=cmd_train_cnn_bilstm_crf)

    sub.add_parser("summary", help="gerar experiment_summary.json").set_defaults(func=cmd_summary)

    al = sub.add_parser("all", help="pipeline completo (precisa de blast+)")
    al.add_argument("--threads", type=int, default=2)
    al.add_argument("--model", choices=["linear", "rbf"], default="linear")
    al.add_argument("--max-train-residues", type=int, default=30000, dest="max_train_residues")
    al.set_defaults(func=cmd_all)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = Paths(root=Path(args.data_dir))
    args.func(paths, args)


if __name__ == "__main__":
    main()
