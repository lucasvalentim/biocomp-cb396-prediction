"""Configuração central: caminhos e constantes do pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---- Constantes do experimento ----
CB396_URL = "https://www.compbio.dundee.ac.uk/jpred/downloads/396.concise.tar.gz"
SWISSPROT_URL = "https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz"
BLAST_DB_NAME = "swissprot"

WINDOW_SIZE = 13
RANDOM_STATE = 42
N_SPLITS = 7

PSIBLAST_ITERATIONS = 3
PSIBLAST_EVALUE = 0.001

LABEL_TO_ID = {"H": 0, "E": 1, "C": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


@dataclass
class Paths:
    """Estrutura de diretórios do projeto. Padrão: ``<root>/data``."""

    root: Path = field(default_factory=lambda: Path("data"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def fasta_per_protein(self) -> Path:
        return self.root / "fasta_per_protein"

    @property
    def pssm(self) -> Path:
        return self.root / "pssm"

    @property
    def blast_out(self) -> Path:
        return self.root / "blast_out"

    @property
    def blastdb(self) -> Path:
        return self.root / "blastdb"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def mkdirs(self) -> None:
        for d in (
            self.raw,
            self.processed,
            self.fasta_per_protein,
            self.pssm,
            self.blast_out,
            self.blastdb,
            self.artifacts,
        ):
            d.mkdir(parents=True, exist_ok=True)
