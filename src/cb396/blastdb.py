"""Download do banco BLAST Swiss-Prot (pré-formatado, FTP do NCBI)."""

from __future__ import annotations

import subprocess
import tarfile

import requests

from .config import SWISSPROT_URL, Paths


def download_swissprot(paths: Paths) -> None:
    """Baixa e extrai o Swiss-Prot pré-formatado em ``paths.blastdb``.

    Usa o tarball pré-formatado do NCBI em vez de ``update_blastdb.pl`` (que não
    vem no pacote ``ncbi-blast+`` do apt).
    """
    paths.blastdb.mkdir(parents=True, exist_ok=True)
    if any(paths.blastdb.glob("swissprot.p*")):
        return

    tgz = paths.blastdb / "swissprot.tar.gz"
    with requests.get(SWISSPROT_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(tgz, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(paths.blastdb)
    tgz.unlink()


def db_info(paths: Paths) -> str:
    """Retorna a saída de ``blastdbcmd -info`` (valida o banco)."""
    result = subprocess.run(
        ["blastdbcmd", "-db", str(paths.blastdb / "swissprot"), "-info"],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr
