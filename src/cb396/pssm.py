"""Execução do PSI-BLAST e parsing das PSSMs ASCII."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from .config import (
    BLAST_DB_NAME,
    PSIBLAST_EVALUE,
    PSIBLAST_ITERATIONS,
    Paths,
)


def run_psiblast(protein_id: str, paths: Paths, *, num_threads: int = 2, force: bool = False) -> dict:
    """Roda PSI-BLAST para uma proteína, gerando ``<id>.pssm``.

    Cacheia por arquivo: se a PSSM já existe e ``force`` é falso, pula.
    """
    fasta_path = paths.fasta_per_protein / f"{protein_id}.fasta"
    pssm_path = paths.pssm / f"{protein_id}.pssm"
    out_path = paths.blast_out / f"{protein_id}.txt"

    if pssm_path.exists() and not force:
        return {"id": protein_id, "status": "cached", "returncode": 0, "stderr": ""}

    env = os.environ.copy()
    env["BLASTDB"] = str(paths.blastdb)

    cmd = [
        "psiblast",
        "-query", str(fasta_path),
        "-db", BLAST_DB_NAME,
        "-num_iterations", str(PSIBLAST_ITERATIONS),
        "-evalue", str(PSIBLAST_EVALUE),
        "-out_ascii_pssm", str(pssm_path),
        "-out", str(out_path),
        "-num_threads", str(num_threads),
    ]
    result = subprocess.run(
        cmd, env=env, cwd=str(paths.blastdb), capture_output=True, text=True
    )
    ok = result.returncode == 0 and pssm_path.exists()
    return {
        "id": protein_id,
        "status": "ok" if ok else "error",
        "returncode": result.returncode,
        "stderr": result.stderr[-2000:],
    }


def parse_ascii_pssm(pssm_path: Path) -> tuple[str, np.ndarray]:
    """Lê uma PSSM ASCII. Retorna ``(sequence, matrix[L, 20])``."""
    rows: list[list[int]] = []
    aas: list[str] = []
    with open(pssm_path, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 22 or not parts[0].isdigit():
                continue
            try:
                scores = [int(x) for x in parts[2:22]]
            except ValueError:
                continue
            aas.append(parts[1])
            rows.append(scores)
    if not rows:
        raise ValueError(f"Nenhuma linha PSSM lida em {pssm_path}")
    return "".join(aas), np.asarray(rows, dtype=np.float32)
