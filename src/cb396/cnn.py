"""CNN 1D para predição de estrutura secundária (Q3) a partir das PSSMs.

Reaproveita as PSSMs já geradas (`data/pssm/`) e o split por proteína. Cada
proteína é um exemplo de comprimento variável (`L × 20`); a CNN prediz os ``L``
resíduos de uma vez (fully-convolutional). Requer o extra ``cnn`` (PyTorch):

    uv sync --extra cnn
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .config import LABEL_TO_ID, N_SPLITS, RANDOM_STATE, WINDOW_SIZE, Paths
from .features import load_features
from .pssm import parse_ascii_pssm

PAD_LABEL = -100  # ignorado pela CrossEntropyLoss (posições de padding)


@dataclass
class CNNConfig:
    in_ch: int = 20
    hidden: int = 64
    depth: int = 3
    kernel: int = 11
    dropout: float = 0.3
    n_classes: int = 3


@dataclass
class CNNBiLSTMConfig:
    in_ch: int = 20
    hidden: int = 64        # canais das convs
    depth: int = 2          # nº de camadas conv
    kernel: int = 11
    dropout: float = 0.3
    lstm_hidden: int = 128  # estado oculto do LSTM (por direção)
    lstm_layers: int = 1
    n_classes: int = 3


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #

ProteinItem = tuple[str, np.ndarray, np.ndarray]  # (id, X[L,20], y[L])


def load_protein_tensors(
    df_valid: pd.DataFrame, paths: Paths, *, normalize: bool = True
) -> list[ProteinItem]:
    """Carrega (id, PSSM[L,20], rótulos[L]) por proteína a partir de ``data/pssm``."""
    items: list[ProteinItem] = []
    for row in df_valid.itertuples(index=False):
        pssm_path = paths.pssm / f"{row.id}.pssm"
        if not pssm_path.exists():
            continue
        seq, pssm = parse_ascii_pssm(pssm_path)
        if len(seq) != len(row.sequence):
            continue
        x = pssm.astype(np.float32)
        if normalize:
            x = 1.0 / (1.0 + np.exp(-x))  # sigmoid nos log-odds da PSSM
        y = np.fromiter((LABEL_TO_ID[c] for c in row.q3), dtype=np.int64)
        items.append((row.id, x, y))
    if not items:
        raise SystemExit("Nenhuma PSSM encontrada em data/pssm/.")
    return items


def load_protein_tensors_from_features(
    paths: Paths, *, normalize: bool = True
) -> list[ProteinItem]:
    """Reconstrói (id, PSSM[L,20], rótulos[L]) por proteína a partir de ``X/y/groups``.

    A coluna central de cada janela (a posição do próprio resíduo) é a linha PSSM
    original — então a PSSM por proteína é recuperável sem os ``.pssm`` brutos.
    """
    X, y, groups = load_features(paths)
    half = WINDOW_SIZE // 2
    center = slice(half * 20, (half + 1) * 20)  # colunas 120:140

    items: list[ProteinItem] = []
    for pid in pd.unique(groups):  # ordem de primeira aparição
        mask = groups == pid
        x = X[mask][:, center].astype(np.float32)
        if normalize:
            x = 1.0 / (1.0 + np.exp(-x))
        items.append((str(pid), x, y[mask].astype(np.int64)))
    return items


class ProteinDataset(Dataset):
    def __init__(self, items: list[ProteinItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        _id, x, y = self.items[i]
        return torch.from_numpy(x), torch.from_numpy(y)


def _collate(batch):
    """Padding por batch → X (B, C, Lmax); rótulos de padding = PAD_LABEL."""
    xs, ys = zip(*batch)
    L = max(x.shape[0] for x in xs)
    C = xs[0].shape[1]
    X = torch.zeros(len(xs), C, L, dtype=torch.float32)
    Y = torch.full((len(xs), L), PAD_LABEL, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        n = x.shape[0]
        X[i, :, :n] = x.transpose(0, 1)
        Y[i, :n] = y
    return X, Y


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #


def _conv_body(in_ch, hidden, depth, kernel, dropout) -> nn.Sequential:
    """Pilha de Conv1d(same) → BatchNorm → ReLU → Dropout."""
    layers: list[nn.Module] = []
    ch = in_ch
    for _ in range(depth):
        layers += [
            nn.Conv1d(ch, hidden, kernel, padding=kernel // 2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        ch = hidden
    return nn.Sequential(*layers)


class SSCNN(nn.Module):
    """Pilha de convoluções 1D com BatchNorm/ReLU/Dropout + cabeça 1×1."""

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.body = _conv_body(cfg.in_ch, cfg.hidden, cfg.depth, cfg.kernel, cfg.dropout)
        self.head = nn.Conv1d(cfg.hidden, cfg.n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 20, L) -> (B, 3, L)
        return self.head(self.body(x))


class SSCNNBiLSTM(nn.Module):
    """CNN (features locais) → BiLSTM (contexto global) → classificador por resíduo.

    O comprimento real de cada proteína é inferido das colunas não nulas da
    entrada (o padding do batch é preenchido com zeros), e o BiLSTM usa
    ``pack_padded_sequence`` para ignorar o padding.
    """

    def __init__(self, cfg: CNNBiLSTMConfig):
        super().__init__()
        self.body = _conv_body(cfg.in_ch, cfg.hidden, cfg.depth, cfg.kernel, cfg.dropout)
        self.lstm = nn.LSTM(
            input_size=cfg.hidden,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(2 * cfg.lstm_hidden, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 20, L) -> (B, 3, L)
        L = x.shape[-1]
        lengths = (x.abs().sum(dim=1) > 0).sum(dim=1).clamp(min=1)  # (B,)
        h = self.body(x).transpose(1, 2)  # (B, L, hidden)
        packed = pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        out = self.head(self.dropout(out))  # (B, L, 3)
        return out.transpose(1, 2)  # (B, 3, L)


class CRF(nn.Module):
    """CRF linear-chain (forward/Viterbi), batch-first, com máscara.

    Implementação enxuta (estilo pytorch-crf): assume que o padding está sempre no
    fim da sequência (é o nosso caso). Emissões com shape ``(B, L, T)``.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        for p in (self.start_transitions, self.end_transitions, self.transitions):
            nn.init.uniform_(p, -0.1, 0.1)

    def nll(self, emissions, tags, mask):
        """Negative log-likelihood média do batch."""
        numerator = self._score(emissions, tags, mask)
        denominator = self._partition(emissions, mask)
        return (denominator - numerator).mean()

    def _score(self, emissions, tags, mask):
        B, L, _ = emissions.shape
        m = mask.float()
        ar = torch.arange(B, device=emissions.device)
        score = self.start_transitions[tags[:, 0]] + emissions[ar, 0, tags[:, 0]]
        for i in range(1, L):
            score = score + self.transitions[tags[:, i - 1], tags[:, i]] * m[:, i]
            score = score + emissions[ar, i, tags[:, i]] * m[:, i]
        seq_ends = mask.long().sum(1) - 1
        score = score + self.end_transitions[tags[ar, seq_ends]]
        return score

    def _partition(self, emissions, mask):
        L = emissions.shape[1]
        score = self.start_transitions + emissions[:, 0]  # (B, T)
        for i in range(1, L):
            nxt = score.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1)
            nxt = torch.logsumexp(nxt, dim=1)  # (B, T)
            score = torch.where(mask[:, i].unsqueeze(1), nxt, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)  # (B,)

    def decode(self, emissions, mask):
        B, L, _ = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for i in range(1, L):
            nxt = score.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1)
            nxt, idx = nxt.max(dim=1)  # (B, T), (B, T)
            score = torch.where(mask[:, i].unsqueeze(1), nxt, score)
            history.append(idx)
        score = score + self.end_transitions
        seq_ends = mask.long().sum(1) - 1
        best: list[list[int]] = []
        for b in range(B):
            last = score[b].argmax().item()
            tags = [last]
            for hist in reversed(history[: seq_ends[b]]):
                last = hist[b][tags[-1]].item()
                tags.append(last)
            tags.reverse()
            best.append(tags)
        return best


class SSCNNBiLSTMCRF(nn.Module):
    """CNN → BiLSTM → emissões por resíduo → CRF (decodificação estruturada).

    O CRF modela as transições entre H/E/C, capturando a "gramática" dos segmentos
    (hélices/folhas têm comprimento mínimo e não alternam resíduo a resíduo).
    """

    def __init__(self, cfg: CNNBiLSTMConfig):
        super().__init__()
        self.body = _conv_body(cfg.in_ch, cfg.hidden, cfg.depth, cfg.kernel, cfg.dropout)
        self.lstm = nn.LSTM(
            input_size=cfg.hidden,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.emit = nn.Linear(2 * cfg.lstm_hidden, cfg.n_classes)
        self.crf = CRF(cfg.n_classes)

    def _emissions(self, x):
        L = x.shape[-1]
        lengths = (x.abs().sum(dim=1) > 0).sum(dim=1).clamp(min=1)
        h = self.body(x).transpose(1, 2)
        packed = pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        emissions = self.emit(self.dropout(out))  # (B, L, n_classes)
        mask = torch.arange(L, device=x.device)[None, :] < lengths[:, None]
        return emissions, mask

    def nll(self, x, tags):  # tags (B, L) com PAD_LABEL no padding
        emissions, mask = self._emissions(x)
        return self.crf.nll(emissions, tags.clamp(min=0), mask)

    @torch.no_grad()
    def decode(self, x):
        emissions, mask = self._emissions(x)
        return self.crf.decode(emissions, mask)


# --------------------------------------------------------------------------- #
# Treino / avaliação
# --------------------------------------------------------------------------- #


def _train_loop(model, loader, opt, device, epochs):
    """Modelos com método ``nll`` (CRF) usam a própria loss; os demais usam CE."""
    lossf = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    has_nll = hasattr(model, "nll")
    model.train()
    for _ in range(epochs):
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            loss = model.nll(X, Y) if has_nll else lossf(model(X), Y)
            loss.backward()
            opt.step()


@torch.no_grad()
def _predict_items(model, items, device):
    """Modelos com ``decode`` (CRF/Viterbi) decodificam; os demais usam argmax."""
    model.eval()
    has_decode = hasattr(model, "decode")
    y_true, y_pred = [], []
    for _id, x, y in items:
        X = torch.from_numpy(x).transpose(0, 1).unsqueeze(0).to(device)  # (1,20,L)
        if has_decode:
            pred = np.asarray(model.decode(X)[0], dtype=np.int64)
        else:
            pred = model(X)[0].argmax(0).cpu().numpy()
        y_true.append(y)
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)


def _build(make_model, lr: float, device: str):
    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    return model, opt


def cross_validate(
    items: list[ProteinItem],
    make_model,
    *,
    n_splits: int = N_SPLITS,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """GroupKFold por proteína. ``make_model`` devolve um modelo novo a cada fold."""
    torch.manual_seed(RANDOM_STATE)
    ids = np.array([it[0] for it in items])
    idx = np.arange(len(items))
    gkf = GroupKFold(n_splits=n_splits)

    all_true, all_pred, fold_rows = [], [], []
    for fold, (tr, te) in enumerate(gkf.split(idx, groups=ids), start=1):
        train_items = [items[i] for i in tr]
        test_items = [items[i] for i in te]
        loader = DataLoader(
            ProteinDataset(train_items),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        model, opt = _build(make_model, lr, device)
        _train_loop(model, loader, opt, device, epochs)
        yt, yp = _predict_items(model, test_items, device)
        q3 = accuracy_score(yt, yp)
        if verbose:
            print(
                f"Fold {fold}: Q3={q3:.4f} | train_prot={len(tr)} "
                f"test_prot={len(te)} test_res={len(yt)}",
                flush=True,
            )
        all_true.append(yt)
        all_pred.append(yp)
        fold_rows.append(
            {
                "fold": fold,
                "q3": q3,
                "train_proteins": len(tr),
                "test_proteins": len(te),
                "test_residues": int(len(yt)),
            }
        )

    return np.concatenate(all_true), np.concatenate(all_pred), pd.DataFrame(fold_rows)


def train_final(
    items: list[ProteinItem],
    make_model,
    cfg,
    paths: Paths,
    out_name: str,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cpu",
) -> Path:
    """Treina em todas as proteínas e salva pesos + config em ``out_name``."""
    torch.manual_seed(RANDOM_STATE)
    loader = DataLoader(
        ProteinDataset(items),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    model, opt = _build(make_model, lr, device)
    _train_loop(model, loader, opt, device, epochs)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    out = paths.artifacts / out_name
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, out)
    return out
