"""Download, parsing e validação do CB396 (formato ``.concise``)."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import requests
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from .config import CB396_URL, Paths

_ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")


def download_cb396(paths: Paths) -> list[Path]:
    """Baixa e extrai o CB396; retorna a lista de arquivos ``.concise``."""
    paths.raw.mkdir(parents=True, exist_ok=True)
    tar_path = paths.raw / "396.concise.tar.gz"
    extract_dir = paths.raw / "396_concise"

    if not tar_path.exists():
        resp = requests.get(CB396_URL, timeout=120)
        resp.raise_for_status()
        tar_path.write_bytes(resp.content)

    extract_dir.mkdir(parents=True, exist_ok=True)
    if not any(extract_dir.rglob("*.concise")):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)

    return sorted(p for p in extract_dir.rglob("*.concise") if p.is_file())


def _parse_concise_records(path: Path) -> list[dict]:
    """Lê um ``.concise`` preservando chaves repetidas (várias ``sequence``)."""
    records = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        tokens = [x.strip() for x in value.split(",") if x.strip()]
        records.append({"key": key.strip(), "tokens": tokens, "n": len(tokens)})
    return records


def dssp_to_q3_char(c: str) -> str:
    """Converte um estado DSSP (8 estados) para Q3 (H/E/C)."""
    c = c.upper()
    if c in {"H", "G", "I"}:
        return "H"
    if c in {"E", "B"}:
        return "E"
    return "C"


def _is_valid_aa_sequence(tokens: list[str]) -> bool:
    seq = "".join(tokens).replace("-", "").replace("_", "").replace(".", "").upper()
    return len(seq) > 0 and set(seq).issubset(_ALLOWED_AA)


def parse_cb396_file(path: Path) -> tuple[dict | None, dict | None]:
    """Parser robusto de uma entrada. Retorna ``(row, None)`` ou ``(None, erro)``."""
    records = _parse_concise_records(path)
    dssp = [r for r in records if r["key"].lower() == "dssp"]
    seqs = [
        r
        for r in records
        if r["key"].lower() == "sequence" and _is_valid_aa_sequence(r["tokens"])
    ]

    if not dssp:
        return None, {"reason": "missing_dssp", "path": str(path)}
    if not seqs:
        return None, {"reason": "missing_sequence", "path": str(path)}

    dssp_tokens = dssp[0]["tokens"]
    matching = [r for r in seqs if r["n"] == len(dssp_tokens)]
    seq_tokens = (matching or seqs)[0]["tokens"]

    sequence = (
        "".join(seq_tokens).replace("-", "").replace("_", "").replace(".", "").upper()
    )
    dssp_str = "".join(dssp_tokens).replace(" ", "").upper()

    if len(sequence) != len(dssp_str):
        return None, {
            "reason": "length_mismatch",
            "path": str(path),
            "seq_len": len(sequence),
            "dssp_len": len(dssp_str),
            "sequence_candidate_lengths": [r["n"] for r in seqs],
            "dssp_candidate_lengths": [r["n"] for r in dssp],
        }

    q3 = "".join(dssp_to_q3_char(c) for c in dssp_str)
    return {
        "id": path.stem,
        "sequence": sequence,
        "dssp": dssp_str,
        "q3": q3,
        "length": len(sequence),
        "path": str(path),
    }, None


def build_dataset(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Constrói os DataFrames de proteínas válidas e descartadas."""
    rows, skipped = [], []
    for path in files:
        row, error = parse_cb396_file(path)
        (rows if row else skipped).append(row or error)
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def save_dataset(
    df_valid: pd.DataFrame, df_skipped: pd.DataFrame, paths: Paths
) -> None:
    """Salva CSVs, FASTA consolidado e um FASTA por proteína."""
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.fasta_per_protein.mkdir(parents=True, exist_ok=True)

    df_valid.to_csv(paths.processed / "cb396_valid_392.csv", index=False)
    df_skipped.to_csv(paths.processed / "cb396_skipped_4.csv", index=False)

    records = [
        SeqRecord(Seq(r.sequence), id=r.id, description=f"CB396_valid length={r.length}")
        for r in df_valid.itertuples(index=False)
    ]
    SeqIO.write(records, paths.processed / "cb396_valid_392.fasta", "fasta")

    for rec in records:
        SeqIO.write([rec], paths.fasta_per_protein / f"{rec.id}.fasta", "fasta")
